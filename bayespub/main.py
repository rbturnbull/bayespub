import typer
from typing_extensions import Annotated
from pathlib import Path
from .huggingface import hugging_face_llm
from bayespub.chains import is_bayesian_splitter_chain
from rich.progress import track

app = typer.Typer()

import warnings
warnings.filterwarnings('ignore')


@app.command()
def is_bayesian(
    base_path: Path, 
    output_path: Path,
    hf_auth:Annotated[str, typer.Argument(envvar=["HF_AUTH"])]="",
):
    chain = is_bayesian_splitter_chain(base_path=base_path, hf_auth=hf_auth, use_hf=True)

    # find all pmids
    files = Path(base_path).glob("*.xml")
    pmids = set()
    for file in files:
        pmid = file.with_suffix("").name
        pmids.add(pmid)

    # find already processed pmids
    processed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                pmid = line.split(",")[0]
                if not pmid.isdigit():
                    continue
                processed.add(pmid)

    todo = pmids - processed

    with open(output_path, "a") as f:
        if len(processed) == 0:
            print("pmid", "is_bayesian", sep=",", file=f, flush=True)
        for pmid in track(todo, description="Processing"):
            try:
                result = chain.invoke(pmid)
                print(pmid, result, sep=",", file=f, flush=True)
            except ValueError as err:
                print(f"Error reading pmid {pmid}:\n{err}")



if __name__ == "__main__":
    app()
