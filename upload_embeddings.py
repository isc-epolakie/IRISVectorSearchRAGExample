
# size 1536
# %%
import iris  # The InterSystems IRIS Python DB-API driver 
import pandas as pd
import numpy as np
import json
from pathlib import Path


# --- Configuration ---
PARQUET_FILE_PATH = "your_embeddings.parquet" 
IRIS_HOST = "localhost"
IRIS_PORT = 8881
IRIS_NAMESPACE = "VECTOR"
IRIS_USERNAME = "superuser"
IRIS_PASSWORD = "sys"
TABLE_NAME = "AIDemo.Embeddings" # Must match the table created in IRIS
EMBEDDING_DIMENSIONS = 1536 # Must match the dimensions for the embeddings you used

def upload_embeddings_to_iris(parquet_path: str):
    """
    Reads a Parquet file with 'chunk_text' and 'embedding' columns 
    and uploads them to an InterSystems IRIS vector database table.
    """
    
    # 1. Load data from the Parquet file using pandas
    try:
        df = pd.read_parquet(parquet_path)
        if 'chunk_text' not in df.columns or 'embedding' not in df.columns:
            print("Error: Parquet file must contain 'chunk_text' and 'embedding' columns.")
            return
    except FileNotFoundError:
        print(f"Error: The file at {parquet_path} was not found.")
        return
    

    # Ensure embeddings are in a format compatible with TO_VECTOR function (list of floats)
    # Parquet often saves numpy arrays as lists
    if isinstance(df['embedding'].iloc[0], np.ndarray):
        df['embedding'] = df['embedding'].apply(lambda x: x.tolist())

    print(f"Loaded {len(df)} records from {parquet_path}.")

    # 2. Establish connection to InterSystems IRIS
    connection = None
    try:
        conn_string = f"{IRIS_HOST}:{IRIS_PORT}/{IRIS_NAMESPACE}"
        connection = iris.connect(conn_string, IRIS_USERNAME, IRIS_PASSWORD)
        cursor = connection.cursor()
        print("Successfully connected to InterSystems IRIS.")

        # Create embedding table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS  {TABLE_NAME} (
            ID INTEGER IDENTITY PRIMARY KEY,
            chunk_text VARCHAR(2500), embedding VECTOR(FLOAT, {EMBEDDING_DIMENSIONS})
            )"""
        )

        # 3. Prepare the SQL INSERT statement
        # InterSystems IRIS uses the TO_VECTOR function for inserting vector data via SQL
        insert_sql = f"""
        INSERT INTO {TABLE_NAME} (chunk_text, embedding) 
        VALUES (?, TO_VECTOR(?))
        """
        
        # 4. Iterate and insert data
        count = 0
        for index, row in df.iterrows():
            text = row['chunk_text']
            # Convert the list of floats to a JSON string, which is required by TO_VECTOR when using DB-API
            vector_json_str = json.dumps(row['embedding']) 
            
            cursor.execute(insert_sql, (text, vector_json_str))
            count += 1
            if count % 100 == 0:
                print(f"Inserted {count} rows...")
        
        # Commit the transaction
        connection.commit()
        print(f"Data upload complete. Total rows inserted: {count}.")

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

# --- Example Usage ---

if __name__ == "__main__":
    # Note: Replace 'your_embeddings.parquet' with the actual path to your file.
    # The script assumes the file exists and the table is set up.
    current_file_path = Path(__file__).resolve()
    upload_embeddings_to_iris(current_file_path.parent / "BusinessService_embedded.parquet")