import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pathlib import Path
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path="venv/dev.env")

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Chunk text prioritizing paragraph and sentence boundaries using
    RecursiveCharacterTextSplitter. Returns a list of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        # Prioritize larger semantic units first, then fall back to smaller ones
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)

    
def chunk_file(path: Path, chunk_size: int, chunk_overlap: int, encoding_name: str = "cl100k_base") -> list[dict]:
    """
    Read a file, split its contents into token-aware chunks, and return metadata for each chunk.

    Returns a list of dicts with keys:
    - filename
    - relative_path
    - absolute_path
    - chunk_index
    - chunk_text
    - token_count
    - modified_time
    - size_bytes
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"Failed to read file {p}: {e}")

    # Prepare tokenizer for accurate token counts
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        raise ValueError(f"Invalid encoding name '{encoding_name}': {e}")

    # Create chunks using provided chunker
    chunks = chunk_text_by_tokens(text, chunk_size, chunk_overlap)

    # File metadata
    stat = p.stat()
    from datetime import datetime, timezone
    modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    absolute_path = str(p.resolve())
    try:
        relative_path = str(p.resolve().relative_to(Path.cwd()))
    except Exception:
        relative_path = p.name

    # Build rows
    rows: list[dict] = []
    for idx, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        rows.append({
            "filename": p.name,
            "relative_path": relative_path,
            "absolute_path": absolute_path,
            "chunk_index": idx,
            "chunk_text": chunk,
            "token_count": token_count,
            "modified_time": modified_time,
            "size_bytes": stat.st_size,
        })

    return rows


def load_documentation_to_parquet(input_dir=None, output_file=None, chunk_size=300, chunk_overlap=50, encoding_name="cl100k_base"):
    """Load all .txt and .md files documentation folder into a single Parquet file with chunked content.

    Parameters:
    - input_dir (str or Path, optional): Directory to read files from.
    - output_file (str or Path, optional): Output parquet path.

    Returns:
    - Path to the written Parquet file (as a `pathlib.Path`).

    Raises:
    - FileNotFoundError: if the input directory does not exist.
    - ValueError: if no matching files were found.
    - RuntimeError: if parquet writing fails.
    """

    # Determine the input and output paths
    current_file_path = Path(__file__).resolve()
    if input_dir is None:
        raise FileNotFoundError(f"No input directory provided. Please specify an input directory.")
    else:
        input_dir = Path(input_dir).resolve()

    if output_file is None:
        raise FileNotFoundError(f"No output file provided. Please specify an output file.")
    else:
        output_file = Path(output_file).resolve()

    # Check if the input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist or is not a directory")

    # Collect data from all .txt and .md files into chunks
    rows = []
    for file_type in ["*.txt", "*.md"]:
        for path in sorted(input_dir.rglob(file_type)):
            try:
                rows.extend(chunk_file(path, chunk_size, chunk_overlap, encoding_name))
            except Exception as e:
                print(f"Failed to process {path}: {e}")

    if not rows:
        raise ValueError(f"No files found in {input_dir} matching patterns: *.txt, *.md")

    # Create a DataFrame from the collected data
    df = pd.DataFrame(rows)

    # Write the DataFrame to a Parquet file using pyarrow or fastparquet backend
    try:
        df.to_parquet(output_file, index=False)
    except Exception as e:
        print(f"Failed to write parquet file: {e}")

    return output_file

def embed_and_save_parquet(input_parquet_path: str, output_parquet_path: str):
    """
    Loads a Parquet file, creates embeddings for the 'chunk_text' column using 
    OpenAI's small embedding model, and saves the result to a new Parquet file.

    Args:
        input_parquet_path (str): Path to the input Parquet file containing 'chunk_text'.
        output_parquet_path (str): Path to save the new Parquet file with embeddings.
        openai_api_key (str): Your OpenAI API key.
    """

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        # Load the Parquet file
        df = pd.read_parquet(input_parquet_path)

        # Initialize OpenAI client
        client = OpenAI(api_key=key)

        # Generate embeddings for each chunk_text
        embeddings = []
        for text in df['chunk_text']:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"  # Using the small embedding model
            )
            embeddings.append(response.data[0].embedding)

        # Add embeddings to the DataFrame
        df['embedding'] = embeddings

        # Save the new DataFrame to a Parquet file
        df.to_parquet(output_parquet_path, index=False)
        print(f"Embeddings generated and saved to {output_parquet_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_parquet_path}")
    except KeyError:
        print("Error: 'chunk_text' column not found in the input Parquet file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def pretty_print_strings(strings: list[str], separator: str = "-"*60) -> str:
    """
    Pretty-print a list of strings with a visible separator between items.

    Each element will be placed on its own line, and a line containing the
    separator (default "-") will be inserted between consecutive items.

    This function prints the result to stdout as a side effect and also returns
    the composed string.
    """
    if not strings:
        output = ""
        print(output)
        return output

    lines = []
    for i, s in enumerate(strings):
        lines.append(s)
        if i != len(strings) - 1:
            lines.append(separator)
    output = "\n".join(lines)
    print(output)

def pretty_print_dicts(dicts: list[dict], separator: str = "-"*60) -> str:
    """
    Pretty-print a list of dictionaries with a visible separator between items.

    Each dictionary will be pretty-printed as JSON with indentation, and a line
    containing the separator (default 60 dashes) will be inserted between
    consecutive items.
    This function prints the result to stdout as a side effect and also returns
    the composed string.
    """
    if not dicts:
        output = ""
        print(output)
        return output

    import json
    blocks = []
    for i, d in enumerate(dicts):
        # Use default=str to gracefully handle non-serializable objects (e.g., Path)
        pretty = json.dumps(d, indent=2, ensure_ascii=False, default=str)
        blocks.append(pretty)
        if i != len(dicts) - 1:
            blocks.append(separator)
    output = "\n".join(blocks)
    print(output)

def main():
    # Example usage: chunk and write parquet files for each Documentation folder
    CHUNK_SIZE_TOKENS = 300
    CHUNK_OVERLAP_TOKENS = 50
    ENCODING_NAME="cl100k_base"
    EMBEDDING_DIM=1536
    current_file_path = Path(__file__).resolve()


    load_documentation_to_parquet(input_dir=current_file_path.parent / "Documentation" / "BusinessService", 
                                  output_file=current_file_path.parent / "BusinessService.parquet", 
                                  chunk_size=CHUNK_SIZE_TOKENS, 
                                  chunk_overlap=CHUNK_OVERLAP_TOKENS, 
                                  encoding_name=ENCODING_NAME)

    embed_and_save_parquet(input_parquet_path=current_file_path.parent / "BusinessService.parquet", 
                            output_parquet_path=current_file_path.parent / "BusinessService_embedded.parquet")

if __name__ == "__main__":
    main()
