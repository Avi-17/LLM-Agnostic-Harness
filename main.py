import os
from urllib.parse import quote
import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_database_url() -> str:
    load_dotenv()
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    database = quote(os.environ["DB_NAME"], safe="")
    user = quote(os.environ["DB_USER"], safe="")
    password = quote(os.environ["DB_PASSWORD"], safe="")

    return (
        f"postgresql://{user}:{password}"
        f"@{host}:{port}/{database}"
    )

def main() -> None:
    database_url = build_database_url()

    # 1. Verify the database connection and pgvector extension.
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT current_database(), current_user, version() 
            """
        ).fetchone()

        vector = conn.execute(
            """
            SELECT extversion from pg_extension
            where extname ='vector'
            """
        ).fetchone()

    print("Database:", row[0])
    print("User:", row[1])
    print("PostgreSQL:", row[2].split(",")[0])
    print("pgvector:", vector[0] if vector else "NOT ENABLED")

    # 2. Initialize LangGraph checkpoint tables.
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()

    print("Checkpoint storage initialized.")

    # Configure semantic search.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        output_dimensionality=1536,
    )

    index_config = {
        "dims": 1536,
        "embed": embeddings,
        "fields": ["text"],
    }

    with PostgresStore.from_conn_string(
            database_url,
            index=index_config,
    ) as store:
        store.setup()

        namespace = ("ai_harness", "development")

        # Save test memories. Each text field will be embedded.
        store.put(
            namespace,
            "database",
            {
                "text": (
                    "PostgreSQL with pgvector stores vector embeddings "
                    "and supports similarity search."
                )
            },
        )

        store.put(
            namespace,
            "checkpoints",
            {
                "text": (
                    "LangGraph checkpoints persist workflow state "
                    "so interrupted executions can resume."
                )
            },
        )

        store.put(
            namespace,
            "project",
            {
                "text": (
                    "This AI harness uses Python and LangGraph "
                    "to orchestrate agent workflows."
                )
            },
        )

        # Search by meaning rather than exact keywords.
        results = store.search(
            namespace,
            query="How can an interrupted agent continue its work?",
            limit=3,
        )

        print("\nSemantic search results:")

        for item in results:
            print(f"\nKey: {item.key}")
            print(f"Score: {item.score}")
            print(f"Memory: {item.value['text']}")

    print("All database tests passed.")

if __name__ == "__main__":
    main()