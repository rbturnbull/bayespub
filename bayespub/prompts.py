from langchain import PromptTemplate

def is_bayesian_prompt():
    template = """<s>[INST] <<SYS>>
    You are an academic who is familiar with both Bayesian and Frequentist statistics. 
    You have strong general knowledge of medical academic literature.

    Format your answer with a single character: either 'y' for yes or 'n' for no.  Do not give any more information than that.
    <</SYS>>

    This is a journal article entitled '{title}'.
    Here is the abstract:
    {abstract}

    Here are some keywords:
    {keywords}

    Does this journal article discuss Bayesian statistics? 
    NOTE: just using the Bayesian information criterion (BIC) is not sufficient for an article to discuss Bayesian statistics.
    [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["title", "abstract", "keywords"],
        template=template,
    )
    return prompt