
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable.branch import RunnableBranch
from langchain.schema.runnable import RunnableParallel
from langchain.schema.runnable.passthrough import RunnablePassthrough

from .io import OutputResult
from .huggingface import hugging_face_llm
from .io import PubMedParser
from .prompts import is_bayesian_prompt, summarize_prompt, summary_synthesize_prompt
from .parsers import YesNoOutputParser
from .splitters import PubMedSplitter
from .reducers import reduce_any, concatenate_summaries

def is_bayesian_chain(hf_auth:str="", use_hf:bool=True, openai_api_key="", **kwargs):
    prompt = is_bayesian_prompt()
    llm = hugging_face_llm(hf_auth, **kwargs) if use_hf else ChatOpenAI(openai_api_key=openai_api_key)
    output_parser = YesNoOutputParser()

    return prompt | llm | StrOutputParser() | output_parser
    

def is_bayesian_parser_chain(base_path:Path, hf_auth:str="", use_hf:bool=True, **kwargs):
    parser = PubMedParser(base_path)
    chain = is_bayesian_chain(hf_auth, use_hf, **kwargs)

    return parser | chain


def is_bayesian_splitter_chain(base_path:Path, output_path:Path, hf_auth:str="", openai_api_key="", use_hf:bool=True, chunk_size:int=6_000, chunk_overlap:int=200, **kwargs):
    parser = PubMedParser(base_path)
    chain = is_bayesian_chain(hf_auth, use_hf, openai_api_key=openai_api_key, **kwargs)
    splitter = PubMedSplitter(chunk_size, chunk_overlap)

    output_chain = (
        RunnableParallel(
            pmid=RunnablePassthrough(),
            result=parser | splitter | chain.map() | reduce_any
        )
        | OutputResult(output_path)
    )

    return output_chain


def summarize_chain(hf_auth:str="", use_hf:bool=True, openai_api_key="", **kwargs):
    prompt = summarize_prompt()
    llm = hugging_face_llm(hf_auth, **kwargs) if use_hf else ChatOpenAI(openai_api_key=openai_api_key)

    return prompt | llm | StrOutputParser()


def summarize_parser_chain(base_path:Path, hf_auth:str="", use_hf:bool=True, **kwargs):
    parser = PubMedParser(base_path)
    chain = summarize_chain(hf_auth, use_hf, **kwargs)

    return parser | chain


def summarize_chain_map_reduce(hf_auth:str="", use_hf:bool=True, openai_api_key="", chunk_size:int=6_000, chunk_overlap:int=200, **kwargs):
    splitter = PubMedSplitter(chunk_size, chunk_overlap)
    chain = summarize_chain(hf_auth, use_hf, **kwargs)
    prompt = summary_synthesize_prompt()
    breakpoint()

    return splitter | chain.map() | concatenate_summaries | prompt


def summarize_splitter_chain(base_path:Path, hf_auth:str="", use_hf:bool=True, chunk_size:int=6_000, chunk_overlap:int=200, **kwargs):
    parser = PubMedParser(base_path)

    summarize_branch_chain = RunnableBranch(
        (lambda doc: len(doc['abstract']) > chunk_size, summarize_chain_map_reduce(hf_auth, use_hf, **kwargs)),
        summarize_chain(hf_auth, use_hf, **kwargs),
    )
    return parser | summarize_branch_chain | StrOutputParser()




