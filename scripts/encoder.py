from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")


def get_codebert_embeddings(texts, tokenizer, model, batch_size=16, max_length=512):
    """
    Generate embeddings for a list of texts using CodeBERT.

    Parameters:
        texts (list of str): List of input texts.
        tokenizer: CodeBERT tokenizer.
        model: CodeBERT model.
        batch_size (int): Number of texts to process in each batch.
        max_length (int): Maximum length for tokenization.

    Returns:
        torch.Tensor: Tensor of embeddings (shape: num_texts x embedding_dim).
    """
    print(f"Starting embedding process, texts size: {len(texts)}")
    embeddings = []
    texts = [str(text) if text is not None and isinstance(text, str) else "" for text in texts]
    for i in range(0, len(texts), batch_size):
        print(f"Elaborating text n.{i}")
        batch_texts = texts[i:i + batch_size]
        # Tokenize the batch
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():  # Disable gradient computation
            outputs = model(**inputs)
            # Use the [CLS] token's output as the embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(batch_embeddings)
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