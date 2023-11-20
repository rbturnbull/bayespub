
from fastapi import FastAPI, Request
from langserve import add_routes
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .chains import single_pmid_question_chain

def server(base_path:Path, hf_auth:str="", use_hf:bool=True, openai_api_key=""):
    app = FastAPI(
        title="BayesPub Server",
        version="1.0",
        description="A simple API server.",
    )
    
    templates = Jinja2Templates(directory=Path(__file__).parent/"templates")

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        return templates.TemplateResponse("root.html", context={"request": request})

    qa_chain = single_pmid_question_chain(base_path=base_path, hf_auth=hf_auth, use_hf=use_hf, openai_api_key=openai_api_key)
    add_routes(
        app,
        qa_chain,
        path="/qa",
    )

    return app
