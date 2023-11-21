from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable.branch import RunnableBranch
from langchain.schema.runnable import RunnableParallel
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.vectorstores import Chroma
from operator import itemgetter
from langchain.schema.runnable import RunnableMap

from .io import OutputResult
from .huggingface import hugging_face_llm
from .io import PubMedParser
from .prompts import (
    bayespub_prompt, 
    is_bayesian_prompt, 
    summarize_prompt, 
    summary_synthesize_prompt, 
    summarize_prompt_full, 
    summary_synthesize_prompt_full,
    bayespub_rag_prompt,
)
from .embeddings import get_bge_embeddings
from .parsers import YesNoOutputParser, RagParser
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


def summarize_chain(hf_auth:str="", use_hf:bool=True, openai_api_key="", full:bool=False, **kwargs):
    prompt = summarize_prompt_full() if full else summarize_prompt()
    llm = hugging_face_llm(hf_auth, **kwargs) if use_hf else ChatOpenAI(openai_api_key=openai_api_key)

    return prompt | llm | StrOutputParser()


def summarize_parser_chain(base_path:Path, hf_auth:str="", use_hf:bool=True, full:bool=False, **kwargs):
    parser = PubMedParser(base_path)
    chain = summarize_chain(hf_auth, use_hf, full=full, **kwargs)

    return parser | chain


def summarize_chain_map_reduce(hf_auth:str="", use_hf:bool=True, openai_api_key="", chunk_size:int=6_000, chunk_overlap:int=200, full:bool=False, **kwargs):
    splitter = PubMedSplitter(chunk_size, chunk_overlap)
    chain = summarize_chain(hf_auth, use_hf, full=full, **kwargs)
    prompt = summary_synthesize_prompt_full() if full else summary_synthesize_prompt()
    llm = chain.middle[0]

    return splitter | chain.map() | concatenate_summaries | prompt | llm | StrOutputParser()


def summarize_splitter_chain(base_path:Path, hf_auth:str="", use_hf:bool=True, chunk_size:int=6_000, chunk_overlap:int=200, full:bool=False, **kwargs):
    parser = PubMedParser(base_path)

    summarize_branch_chain = RunnableBranch(
        (lambda doc: len(doc['abstract']) > chunk_size, summarize_chain_map_reduce(hf_auth, use_hf, full=full, **kwargs)),
        summarize_chain(hf_auth, use_hf, full=full, **kwargs),
    )
    return parser | summarize_branch_chain | StrOutputParser()


def single_pmid_question_chain(base_path:Path, hf_auth:str="", use_hf:bool=True, openai_api_key="", llm=None, **kwargs):
    from langserve.schema import CustomUserType

    class InputType(CustomUserType):
        pmid:int
        system:str
        question:str

    parser = PubMedParser(base_path).with_types(
        input_type=InputType,
    )

    prompt = bayespub_prompt()
    llm = llm or (hugging_face_llm(hf_auth, **kwargs) if use_hf else ChatOpenAI(openai_api_key=openai_api_key))

    return parser | prompt | llm | StrOutputParser()


def rag_chain(
    embeddings_db:Path, 
    embeddings_model_name:str="BAAI/bge-large-en", 
    llm=None, 
    hf_auth:str="", 
    use_hf:bool=True, 
    openai_api_key="",
    context_count:int=5,
):
    embeddings = get_bge_embeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=str(embeddings_db), embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": context_count})
    prompt = bayespub_rag_prompt()
    llm = llm or (hugging_face_llm(hf_auth, **kwargs) if use_hf else ChatOpenAI(openai_api_key=openai_api_key))

    # class InputType(CustomUserType):
    #     system:str
    #     question:str

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Adapted from https://python.langchain.com/docs/use_cases/question_answering/
    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableMap(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    } | RagParser()

    return rag_chain_with_source


