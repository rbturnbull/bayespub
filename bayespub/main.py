import typer
from typing_extensions import Annotated
from pathlib import Path
from bayespub.chains import is_bayesian_splitter_chain, summarize_splitter_chain
from rich.progress import track
from langchain.globals import set_debug, set_verbose


app = typer.Typer()

import warnings
warnings.filterwarnings('ignore')


@app.command()
def is_bayesian(
    base_path: Path, 
    output_path: Path,
    hf_auth:Annotated[str, typer.Argument(envvar=["HF_AUTH"])]="",
):
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

    chain = is_bayesian_splitter_chain(base_path=base_path, output_path=output_path, hf_auth=hf_auth, use_hf=True)

    with open(output_path, "a") as f:
        if len(processed) == 0:
            print("pmid", "is_bayesian", sep=",", file=f, flush=True)
        for pmid in track(todo, description="Processing"):
            try:
                result = chain.invoke(pmid)
                print(pmid, result, sep=",", file=f, flush=True)
            except ValueError as err:
                print(f"Error reading pmid {pmid}:\n{err}")


@app.command()
def summarize(
    base_path: Path, 
    output_path: Path,
    hf_auth:Annotated[str, typer.Argument(envvar=["HF_AUTH"])]="",
    use_hf:bool=True,
    debug:bool=False,
    verbose:bool=False,
):
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

    set_debug(debug)
    set_verbose(verbose)


    chain = summarize_splitter_chain(base_path=base_path, hf_auth=hf_auth, use_hf=use_hf)

    with open(output_path, "a") as f:
        if len(processed) == 0:
            print("pmid", "summary", sep=",", file=f, flush=True)

        for pmid in track(todo, description="Processing"):
            try:
                result = chain.invoke(pmid).strip().replace("\n", " ")
                print(pmid, result, sep=",", file=f, flush=True)
                print(pmid, result, sep=",", flush=True)
            except Exception as err:
                print(f"cannot summarize {pmid}: {err}")




if __name__ == "__main__":
    app()
