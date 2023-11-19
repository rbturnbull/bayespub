import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings


def get_bge_embeddings(model_name:str="BAAI/bge-base-en"):
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    return model