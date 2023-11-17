from typing import Optional
from pathlib import Path
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


def get_text(el) -> str:
    return BeautifulSoup(ET.tostring(el, encoding='unicode'), 'lxml').get_text().strip()


class PubMedParser(Runnable):
    def __init__(self, base_path:Path):
        self.base_path = Path(base_path)

    def invoke(self, pmid:str, config: Optional[RunnableConfig] = None):
        file = self.base_path/f"{pmid}.xml"
        text = file.read_text()
        article = ET.fromstring(text)
        title = get_text(article.find('.//ArticleTitle')) or ""
        try:
            abstract = "\n".join([get_text(el) for el in article.findall('.//AbstractText')])
        except Exception as err:
            print(f"# unable to read abstract in {pmid}: {err}")
            abstract = "No abstract"

        try:
            keywords = "\n".join([get_text(el) for el in article.findall('.//Keyword')])
        except Exception:
            keywords = ""

        return dict(
            pmid=pmid,
            title=title,
            abstract=abstract,
            keywords=keywords,
        )


