import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import json
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel


def load_data(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def generate_embeddings(descriptions, model, tokenizer, output_file):
    """
    Generate embeddings for each description and save them along with their IDs.
    """
    embeddings_with_ids = []

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for id_, description in descriptions.items():
            print(f"Processing ID: {id_}")
            # Tokenize the description
            encoded_input = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Get the model output
            output = model(**encoded_input)

            # Extract the CLS token representation
            embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
            print(embedding.shape)

            # Append the ID and its embedding as a dictionary entry
            embeddings_with_ids.append({"id": id_, "embedding": embedding})

    # Save the embeddings and IDs as a numpy array
    np.save(output_file, embeddings_with_ids, allow_pickle=True)
    print(f"Embeddings and IDs saved to {output_file}")


def main():
    # Path to the input JSON file
    json_file = "./GLM_data/personalized_descriptions.json"

    # Path to save the output embeddings
    output_file = "./GLM_data/descriptions_embeddings_with_ids.npy"
    # Load Roberta model and tokenizer from Hugging Face
    model_name = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    # Load the personalized descriptions
    descriptions = load_data(json_file)
    print(f"Loaded {len(descriptions)} descriptions.")

    # Generate and save embeddings with IDs
    generate_embeddings(descriptions, model, tokenizer, output_file)


if __name__ == "__main__":
    main()
