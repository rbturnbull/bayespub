from typing import Optional
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


class YesNoOutputParser(Runnable):
    def invoke(self, result:str, config: Optional[RunnableConfig] = None) -> bool:
        result_char = result.strip()[:1].lower()
        if result_char == "y":
            return True
        elif result_char == "n":
            return False
        
        raise ValueError(f"Invalid result: {result}")