from typing import Optional
from pathlib import Path
import xml.etree.ElementTree as ET
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


class PubMedParser(Runnable):
    def __init__(self, base_path:Path):
        self.base_path = Path(base_path)

    def invoke(self, pmid:str, config: Optional[RunnableConfig] = None):
        file = self.base_path/f"{pmid}.xml"
        text = file.read_text()
        article = ET.fromstring(text)
        title = article.find('.//ArticleTitle').text or ""
        try:
            abstract = "\n".join([el.text for el in article.findall('.//AbstractText')])
        except Exception:
            print(f"# unable to read abstract in {pmid}")
            abstract = "No abstract"

        try:
            keywords = "\n".join([el.text for el in article.findall('.//Keyword')])
        except Exception:
            keywords = ""

        return dict(
            pmid=pmid,
            title=title,
            abstract=abstract,
            keywords=keywords,
        )


