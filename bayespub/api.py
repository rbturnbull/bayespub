from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, Request
from langserve import add_routes
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .chains import single_pmid_question_chain, rag_chain
from .huggingface import hugging_face_llm



def server(
    base_path:Path, 
    full_db_path:Path = None, 
    method_db_path:Path = None, 
    hf_auth:str="", 
    use_hf:bool=True, 
    openai_api_key="", 
    embeddings_model_name:str="BAAI/bge-large-en", 
    **kwargs
):
    app = FastAPI(
        title="BayesPub Server",
        version="1.0",
        description="A simple API server.",
    )
    llm = hugging_face_llm(hf_auth, **kwargs) if use_hf else ChatOpenAI(openai_api_key=openai_api_key)
    
    templates = Jinja2Templates(directory=Path(__file__).parent/"templates")

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        return templates.TemplateResponse("root.html", context={"request": request})

    qa_chain = single_pmid_question_chain(base_path=base_path, llm=llm)
    add_routes(
        app,
        qa_chain,
        path="/qa",
    )

    if full_db_path and full_db_path.exists():
        rag = rag_chain(full_db_path, llm=llm, embeddings_model_name=embeddings_model_name)
        add_routes(
            app,
            rag,
            path="/rag/full",
        )    

    if method_db_path and method_db_path.exists():
        rag = rag_chain(method_db_path, llm=llm, embeddings_model_name=embeddings_model_name)
        add_routes(
            app,
            rag,
            path="/rag/methods",
        )    

    return app
