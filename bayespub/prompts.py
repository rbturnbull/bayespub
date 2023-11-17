from langchain.prompts import PromptTemplate


def bayespub_prompt():
    template = """<s>[INST] <<SYS>>
    {system}
    <</SYS>>

    This is a journal article entitled '{title}'.
    Here is the abstract:
    {abstract}

    Here are some keywords:
    {keywords}

    {question}
    [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["title", "abstract", "keywords", "question", "system"],
        template=template,
    )
    return prompt


def is_bayesian_prompt():
    prompt = bayespub_prompt()
    return prompt.partial(
        system = 
            "You are an academic who is familiar with both Bayesian and Frequentist statistics.\n"
            "You have strong general knowledge of medical academic literature.\n"
            "Format your answer with a single character: either 'y' for yes or 'n' for no.  Do not give any more information than that.",
        question = 
            "Does this journal article discuss Bayesian statistics? \n"
            "NOTE: Just using the Bayesian information criterion (BIC) is not sufficient for an article to discuss Bayesian statistics.\n"
            "If the article discusses a posterior probability distribution, then answer 'y'.",
    )
