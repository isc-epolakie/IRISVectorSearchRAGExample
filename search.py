import iris
from typing import List
from dotenv import load_dotenv
import os
from openai import OpenAI

# --- Configuration ---
PARQUET_FILE_PATH = "your_embeddings.parquet" 
IRIS_HOST = "localhost"
IRIS_PORT = 8881
IRIS_NAMESPACE = "VECTOR"
IRIS_USERNAME = "superuser"
IRIS_PASSWORD = "sys"
TABLE_NAME = "AIDemo.Embeddings" # Must match the table created in IRIS
EMBEDDING_DIMENSIONS = 1536
MODEL = "text-embedding-3-small"

load_dotenv(dotenv_path="venv/dev.env")

def get_embedding(text: str, model: str, client) -> List[float]:
    # Normalize newlines and coerce to str
    payload = [("" if text is None else str(text)).replace("\n", " ") for _ in range(1)]
    resp = client.embeddings.create(model=model, input=payload, encoding_format="float")
    return resp.data[0].embedding

def search_embeddings(search: str, top_k: int):
    print("-------RAG--------")
    print(f"Searching IRIS vector store for: ", search)
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    # 2. Establish connection to InterSystems IRIS
    connection = None
    try:
        conn_string = f"{IRIS_HOST}:{IRIS_PORT}/{IRIS_NAMESPACE}"
        connection = iris.connect(conn_string, IRIS_USERNAME, IRIS_PASSWORD)
        cursor = connection.cursor()
        print("Successfully connected to InterSystems IRIS.")

        # Embed query for searching
        #emb_raw = str(test_embedding) # FOR TESTING
        emb_raw = get_embedding(search, model=MODEL, client=client)
        emb_raw = str(emb_raw)
        #print("EMB_RAW:", emb_raw)

        emb_values = []
        for x in emb_raw.replace('[', '').replace(']', '').split(','):
            try:
                emb_values.append(str(float(x.strip())))
            except ValueError:
                continue
        emb_str = ", ".join(emb_values)

        # Prepare the SQL SELECT statement
        search_sql = f"""
        SELECT TOP {top_k} ID, chunk_text FROM {TABLE_NAME}
        ORDER BY VECTOR_DOT_PRODUCT((embedding), TO_VECTOR(('{emb_str}'), FLOAT)) DESC
        """

        cursor.execute(search_sql)

        results = []
        row = cursor.fetchone()
        while row is not None:
            results.append(row[:])
            row = cursor.fetchone()
    
    except iris.DBAPIError as e:
        print(f"A database error occurred: {e}")
        if connection:
            connection.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if connection:
            connection.close()
            print("Database connection closed.")
        print("------------RAG Finished-------------")
        return results

if __name__ == "__main__":
    # Example Usage:
    print(search_embeddings("What settings does a business service have?"))
