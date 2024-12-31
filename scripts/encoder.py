from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import List

# Load CodeBERT model and tokenizer
MODEL_NAME = "microsoft/codebert-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def add_embeddings_to_dataframe(
    df: pd.DataFrame,
    text_column: str,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 512
):
    """
    Generate embeddings for a text column in a DataFrame and add them as new columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column containing text data.
        tokenizer: Hugging Face tokenizer for CodeBERT.
        model: Hugging Face model for CodeBERT.
        batch_size (int): Number of texts to process in each batch.
        max_length (int): Maximum length for tokenization.

    Returns:
        pd.DataFrame: Updated DataFrame with embedding columns added.
    """
    total_batches = (len(df) + batch_size - 1) // batch_size  # Calculate total number of batches

    # Ensure all texts are valid strings
    texts = df[text_column].fillna("unknown").astype(str).tolist()

    # Process in batches
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        current_batch = (i // batch_size) + 1
        print(f"Processing batch {current_batch}/{total_batches} for '{text_column}'...")

        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embedding

        embeddings_list.append(batch_embeddings)

    # Combine all embeddings
    all_embeddings = torch.cat([torch.tensor(batch) for batch in embeddings_list], dim=0).numpy()

    # Add embeddings to the DataFrame with `_embedding_` in the column name
    for dim in range(all_embeddings.shape[1]):
        df[f"{text_column}_embedding_dim{dim}"] = all_embeddings[:, dim]

    print(f"Embeddings for '{text_column}' added to DataFrame.")
    return df

class Encoder:

    def encode(self, df, column):
        """
        encode with codebert inplace
        """
        
        df = add_embeddings_to_dataframe(df, column, tokenizer, model)

        return df