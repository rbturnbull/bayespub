import typer
from typing_extensions import Annotated
from .llama import llama2_llm

app = typer.Typer()


@app.command()
def llama2(
    prompt: str, 
    hf_auth:Annotated[str, typer.Argument(envvar=["HF_AUTH"])],
):
    llm = llama2_llm(hf_auth)
    print(llm(prompt))


if __name__ == "__main__":
    app()
