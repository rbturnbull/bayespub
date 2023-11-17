
from pathlib import Path
from langchain.chat_models import ChatOpenAI

from .huggingface import hugging_face_llm
from .io import PubMedParser
from .prompts import is_bayesian_prompt
from .parsers import YesNoOutputParser

def is_bayesian_chain(base_path:Path, hf_auth:str="", use_hf:bool=True, **kwargs):
    parser = PubMedParser(base_path)
    prompt = is_bayesian_prompt()
    llm = hugging_face_llm(hf_auth, **kwargs) if use_hf else ChatOpenAI()
    output_parser = YesNoOutputParser()

    return parser | prompt | llm | output_parser
