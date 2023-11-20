from typing import Optional
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import datetime
from rich.progress import track


def get_text(el) -> str:
    return BeautifulSoup(ET.tostring(el, encoding='unicode'), 'lxml').get_text().strip()


def parse_pubmed_entry(entry) -> dict:
    """
    Reads the XML representation of a citation as text and returns a dictionary with metadata.

    Includes:
        - pmid
        - title
        - abstract
        - keywords
        - date
        - ordinal_date
        - year
        - month
        - day
    """
    pmid = entry.find('.//PMID').text
    title = get_text(entry.find('.//ArticleTitle')) or ""
    try:
        abstract = "\n".join([get_text(el) for el in entry.findall('.//AbstractText')])
    except Exception as err:
        print(f"# unable to read abstract in {pmid}: {err}")
        abstract = "No abstract"

    try:
        keywords = "\n".join([get_text(el) for el in entry.findall('.//Keyword')])
    except Exception:
        keywords = ""

    date = entry.find(".//ArticleDate")
    if date is None:
        date = entry.find(".//DateCompleted")

    if date is None:
        date = entry.find(".//DateRevised")

    if date:
        year = int(date.find("Year").text)
        month = int(date.find("Month").text)
        day = int(date.find("Day").text)
        date_string = f"{year}-{month}-{day}"   
        date = datetime.date(year,month,day)    
        ordinal_date = date.toordinal()
    else:
        year = 0
        month = 0
        day = 0
        ordinal_date = 0     
        date_string = ""

    return dict(
        pmid=pmid,
        title=title,
        abstract=abstract,
        keywords=keywords,
        date=date_string,
        ordinal_date=ordinal_date,     
        year=year,
        month=month,
        day=day,
    )



class PubMedParser(Runnable):
    def __init__(self, base_path:Path):
        self.base_path = Path(base_path)

    def invoke(self, pmid:str, config: Optional[RunnableConfig] = None):
        file = self.base_path/f"{pmid}.xml"
        entry = ET.fromstring(text)
        return parse_pubmed_entry(entry)


class OutputResult(Runnable):
    def __init__(self, output_path:Path, mode:str="a"):
        self.file = open(output_path, mode)

    def invoke(self, result:dict, config: Optional[RunnableConfig] = None):
        pmid = result.pop("pmid")
        print(pmid, ",".join([str(value) for value in result.values()]), file=self.file, flush=True, sep=",")
        return result
    
    def __del__(self):
        if not self.file.closed:
            self.file.close()


def summaries_to_docs(csv:Path, base_path:Path):
    parser = PubMedParser(base_path=base_path)
    documents = []
    with open(csv) as f:
        for index, line in enumerate(f):
            m = re.match(r"^(\d+),(.*)", line)
            if m:
                pmid,summary = m.group(1), m.group(2)
                details = parser.invoke(pmid)

                metadata = dict(
                    pmid=pmid,
                    title=details['title'],
                    date=details['date'],
                    ordinal_date=details['ordinal_date'],
                    year=details['year'],
                )
                document= Document(page_content=summary, metadata=metadata)
                documents.append(document)
                if index % 1000 == 0:
                    print(index)

    return documents


