import typer
from typing_extensions import Annotated
from .huggingface import hugging_face_llm

app = typer.Typer()


@app.command()
def llama2(
    prompt: str, 
    hf_auth:Annotated[str, typer.Argument(envvar=["HF_AUTH"])]="",
):
    llm = hugging_face_llm(hf_auth)
    print(llm(prompt))


if __name__ == "__main__":
    app()
