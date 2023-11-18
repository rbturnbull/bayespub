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


def summarize_prompt():
    prompt = bayespub_prompt()
    return prompt.partial(
        system = 
            "You are an academic who is familiar with both Bayesian and Frequentist statistics. \n"
            "You have strong general knowledge of medical academic literature. \n"
            "Read the title, abstract and keywords of a journal article, paying attention to: \n"
            "   - the main topic of the article,\n"
            "   - the main findings of the article,\n"
            "   - what is the methodological approach of the article,\n"
            "   - what is novel about the methodological approach,\n"
            "   - how Bayesian statistics relates to the methodology.\n",
        question = 
            "Write a very concise summary of the article. Your summary should be no more than 100 words long. \n"
            "Include the main topic, findings, methodological approach, novelty of the methodological approach, "
            "and how Bayesian statistics relates to the methodology.",
    )


def synthesise_summaries():
    prompt = bayespub_prompt()
    return prompt.partial(
        system = 
            "You are an academic who is familiar with both Bayesian and Frequentist statistics. \n"
            "You have strong general knowledge of medical academic literature. \n"
            "Read the list of summaries of a journal article, paying attention to: \n"
            "   - the main topic of the article,\n"
            "   - the main findings of the article,\n"
            "   - what is the methodological approach of the article,\n"
            "   - what is novel about the methodological approach,\n"
            "   - how Bayesian statistics relates to the methodology.\n",
        question = 
            "Write a very concise summary of the article. Your summary should be no more than 100 words long. \n"
            "Include the main topic, findings, methodological approach, novelty of the methodological approach, "
            "and how Bayesian statistics relates to the methodology.",
    )



def summary_synthesize_prompt():
    template = """<s>[INST] <<SYS>>
    You are an academic who is familiar with both Bayesian and Frequentist statistics. 
    You have strong general knowledge of medical academic literature. 
    Read the list of summaries of a journal article, paying attention to: 
       - the main topic of the article,
       - the main findings of the article,
       - what is the methodological approach of the article,
       - what is novel about the methodological approach,
       - how Bayesian statistics relates to the methodology.,
    <</SYS>>

    Write a very concise summary of a journal article by synthesizing the following summaries:
    {summaries}

    Your summary should be no more than 100 words long.
    Include the main topic, findings, methodological approach, novelty of the methodological approach,
    and how Bayesian statistics relates to the methodology.
    [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["summaries"],
        template=template,
    )
    return prompt
