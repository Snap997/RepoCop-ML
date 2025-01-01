from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import List
import tempfile
import os
import numpy as np

# Load CodeBERT model and tokenizer
MODEL_NAME = "microsoft/codebert-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def generate_and_save_embeddings_in_chunks(texts, tokenizer, model, temp_file, batch_size=16, max_length=512, chunk_size=5):
    """
    Generate embeddings for a list of texts and save them to a temporary file in chunks.

    Parameters:
        texts (list): List of input texts.
        tokenizer: Hugging Face tokenizer for CodeBERT.
        model: Hugging Face model for CodeBERT.
        temp_file (str): Path to the temporary file to save embeddings.
        batch_size (int): Number of texts to process in each batch.
        max_length (int): Maximum length for tokenization.
        chunk_size (int): Number of batches to accumulate before saving to file.

    Returns:
        None
    """
    total_batches = (len(texts) + batch_size - 1) // batch_size
    accumulated_embeddings = []

    with open(temp_file, "wb") as f:
        for i in range(0, len(texts), batch_size):
            current_batch = (i // batch_size) + 1
            if(current_batch % batch_size == 0):
                print(f"Processing batch {current_batch}/{total_batches}...")

            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            accumulated_embeddings.append(batch_embeddings)

            # Save to file in chunks
            if len(accumulated_embeddings) >= chunk_size:
                np.save(f, np.vstack(accumulated_embeddings))
                accumulated_embeddings = []

        # Save remaining embeddings
        if accumulated_embeddings:
            np.save(f, np.vstack(accumulated_embeddings))

    print(f"Embeddings saved to temporary file: {temp_file}")


def read_and_update_dataframe_with_embeddings(df, temp_file, text_column):
    """
    Load embeddings from a temporary file and update the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to update.
        temp_file (str): Path to the temporary file containing embeddings.
        text_column (str): Name of the text column to prefix the embedding columns.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    embeddings = []

    # Load all embeddings from the temporary file
    with open(temp_file, "rb") as f:
        while True:
            try:
                batch = np.load(f)
                embeddings.append(batch)
            except ValueError:  # End of file
                break

    embeddings = np.vstack(embeddings)

    # Add embeddings to the DataFrame
    embedding_columns = pd.DataFrame(
        embeddings,
        columns=[f"{text_column}_embedding_dim{dim}" for dim in range(embeddings.shape[1])]
    )

    df = pd.concat([df.reset_index(drop=True), embedding_columns.reset_index(drop=True)], axis=1)
    print(f"Added embeddings for '{text_column}' to DataFrame.")
    return df


def process_embeddings(df, text_column, tokenizer, model, batch_size=16, max_length=512, chunk_size=5):
    """
    Complete workflow: Generate embeddings, save to temp file, load back, and update DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column containing text data.
        tokenizer: Hugging Face tokenizer for CodeBERT.
        model: Hugging Face model for CodeBERT.
        batch_size (int): Number of texts to process in each batch.
        max_length (int): Maximum length for tokenization.
        chunk_size (int): Number of batches to accumulate before saving to file.

    Returns:
        pd.DataFrame: Updated DataFrame with embeddings.
    """
    texts = df[text_column].fillna("").astype(str).tolist()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_file = tmp.name

    try:
        # Step 1: Generate and save embeddings in chunks
        generate_and_save_embeddings_in_chunks(texts, tokenizer, model, temp_file, batch_size, max_length, chunk_size)

        # Step 2: Read embeddings and update the DataFrame
        df = read_and_update_dataframe_with_embeddings(df, temp_file, text_column)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Temporary file deleted: {temp_file}")

    return df

class Encoder:

    def encode(self, df, column):
        """
        encode with codebert inplace
        """
        
        df = process_embeddings(df, column, tokenizer, model, chunk_size=50)

        return df