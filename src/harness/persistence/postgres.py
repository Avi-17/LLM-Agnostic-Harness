"""PostgreSQL connectivity.

The only module that builds connection strings or opens connections.
Other components receive a ready ConnectionPool and never see credentials.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from harness.config import DatabaseSettings

APPLICATION_NAME = "ai_harness"

# Connection settings LangGraph's Postgres checkpointer and store expect when
# you hand them your own connections:
#   autocommit         - their .setup() migrations include statements
#                        (e.g. CREATE INDEX CONCURRENTLY) that can't run in a transaction
#   row_factory        - their queries read result columns by name
#   prepare_threshold  - disables server-side prepared statements, as LangGraph's
#                        docs recommend; avoids conflicts behind connection poolers
CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}


class DatabaseUnavailable(RuntimeError):
    """Could not reach or log into PostgreSQL. Message is safe to print."""


@dataclass(frozen=True)
class DatabaseStatus:
    database: str
    user: str
    server_version: str
    pgvector_version: str | None


def describe_target(db: DatabaseSettings) -> str:
    return f"{db.user}@{db.host}:{db.port}/{db.name}"


def build_conninfo(db: DatabaseSettings, *, connect_timeout: int = 5) -> str:
    """Build a libpq connection string with correct escaping.

    The result CONTAINS THE PASSWORD: pass it to psycopg, never print or log it.
    """
    return make_conninfo(
        host=db.host,
        port=str(db.port),
        dbname=db.name,
        user=db.user,
        password=db.password.get_secret_value(),
        connect_timeout=str(connect_timeout),
        application_name=APPLICATION_NAME,
    )


@contextmanager
def open_pool(
    db: DatabaseSettings,
    *,
    min_size: int = 1,
    max_size: int = 5,
    timeout: float = 10.0,
) -> Iterator[ConnectionPool]:
    """Open a connection pool, failing fast with a readable error."""
    conninfo = build_conninfo(db)
    target = describe_target(db)

    # Probe once directly: a single connect gives a clear libpq error
    # ("password authentication failed...") instead of a generic pool timeout.
    try:
        with psycopg.connect(conninfo):
            pass
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailable(f"Cannot connect to {target}: {str(exc).strip()}") from None

    pool = ConnectionPool(
        conninfo=conninfo,
        min_size=min_size,
        max_size=max_size,
        kwargs=CONNECTION_KWARGS,
        name=APPLICATION_NAME,
        open=False,
    )
    try:
        pool.open(wait=True, timeout=timeout)
    except PoolTimeout:
        pool.close()
        raise DatabaseUnavailable(
            f"Connection pool for {target} did not become ready within {timeout:.0f}s"
        ) from None

    try:
        yield pool
    finally:
        pool.close()


def check_database(pool: ConnectionPool) -> DatabaseStatus:
    """Read-only health check: who/where we are, and whether pgvector is enabled."""
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT current_database()               AS database,
                   current_user                     AS user_name,
                   current_setting('server_version') AS server_version
            """
        ).fetchone()
        vector = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()

    return DatabaseStatus(
        database=row["database"],
        user=row["user_name"],
        server_version=row["server_version"],
        pgvector_version=vector["extversion"] if vector else None,
    )