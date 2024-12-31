from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from typing import List

# Load CodeBERT model and tokenizer
MODEL_NAME = "microsoft/codebert-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def get_codebert_embeddings(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 512
):
    """
    Generate embeddings for a list of texts using CodeBERT.

    Parameters:
        texts (List[str]): List of input texts.
        tokenizer: Hugging Face tokenizer for CodeBERT.
        model: Hugging Face model for CodeBERT.
        batch_size (int): Number of texts to process in each batch.
        max_length (int): Maximum length for tokenization.

    Returns:
        torch.Tensor: Tensor of embeddings (num_texts x embedding_dim).
    """
    embeddings = []

    # Ensure all texts are valid strings
    texts = [str(text) if text is not None else "unknown" for text in texts]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{len(texts) // batch_size + 1}")

        # Tokenize the batch and move inputs to the device
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)

class Encoder:

    def encode(self, df, column):
        """
        encode with codebert inplace
        """
        texts = df[column].tolist()
        embeddings = get_codebert_embeddings(texts, tokenizer, model)

        # Convert embeddings to a DataFrame for integration
        embeddings_df = pd.DataFrame(
            embeddings.numpy(),
            columns=[f"{column}_embedding_{i}" for i in range(embeddings.size(1))]
        )

        # Concatenate the embeddings with the original dataset
        df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)