
from pathlib import Path

from .huggingface import hugging_face_llm
from .io import PubMedParser
from .prompts import is_bayesian_prompt

def is_bayesian_chain(base_path:Path, hf_auth:str="", **kwargs):
    parser = PubMedParser(base_path)
    prompt = is_bayesian_prompt()
    llm = hugging_face_llm(hf_auth, **kwargs)

    return parser | prompt | llm
