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



def bayespub_prompt2():
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

    Sure! Here's the summary you requested:
    """

    prompt = PromptTemplate(
        input_variables=["title", "abstract", "keywords", "question", "system"],
        template=template,
    )
    return prompt

def summarize_prompt_full():
    prompt = bayespub_prompt2()
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
            "and how Bayesian statistics relates to the methodology.\n"
    )


def summarize_prompt():
    prompt = bayespub_prompt2()
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
            "Write a very concise summary of how Bayesian statistics relates to the methodology of the article.\n"
            "Your summary should be no more than one sentence at most 50 words long. \n"
            # "Respond with just the summary text on a single line and do not include an introductory message.\n",
    )


# def synthesise_summaries():
#     prompt = bayespub_prompt()
#     return prompt.partial(
#         system = 
#             "You are an academic who is familiar with both Bayesian and Frequentist statistics. \n"
#             "You have strong general knowledge of medical academic literature. \n"
#             "Read the list of summaries of a journal article, paying attention to: \n"
#             "   - the main topic of the article,\n"
#             "   - the main findings of the article,\n"
#             "   - what is the methodological approach of the article,\n"
#             "   - what is novel about the methodological approach,\n"
#             "   - how Bayesian statistics relates to the methodology.\n"
#             "Respond with just the summary text on a single line and do not include an introductory message.\n",

#         question = 
#             "Write a very concise summary of the article. Your summary should be no more than 100 words long. \n"
#             "Include the main topic, findings, methodological approach, novelty of the methodological approach, "
#             "and how Bayesian statistics relates to the methodology."
#             "Respond with just the summary text on a single line and do not include an introductory message.\n",
#     )



def summary_synthesize_prompt():
    template = """<s>[INST] <<SYS>>
    You are an academic who is familiar with both Bayesian and Frequentist statistics. 
    You have strong general knowledge of medical academic literature. 
    Read the list of summaries of a journal article, paying attention to: 
       - the main topic of the article,
       - the main findings of the article,
       - what is the methodological approach of the article,
       - what is novel about the methodological approach,
       - how Bayesian statistics relates to the methodology.

    Respond with just the summary text on a single line and do not include an introductory message.
    <</SYS>>

    Write a very concise summary of how Bayesian statistics relates to the methodology of the article by synthesizing the following summaries:
    {summaries}

    Your summary should be no more than one sentence at most 50 words long. \n"
    [/INST]

    Sure! Here's the summary you requested:
    """

    prompt = PromptTemplate(
        input_variables=["summaries"],
        template=template,
    )
    return prompt


def summary_synthesize_prompt_full():
    template = """<s>[INST] <<SYS>>
    You are an academic who is familiar with both Bayesian and Frequentist statistics. 
    You have strong general knowledge of medical academic literature. 
    Read the list of summaries of a journal article, paying attention to: 
       - the main topic of the article,
       - the main findings of the article,
       - what is the methodological approach of the article,
       - what is novel about the methodological approach,
       - how Bayesian statistics relates to the methodology.

    Respond with just the summary text on a single line and do not include an introductory message.
    <</SYS>>

    Write a very concise summary of a journal article by synthesizing the following summaries:
    {summaries}

    Your summary should be no more than 100 words long.
    Include the main topic, findings, methodological approach, novelty of the methodological approach,
    and how Bayesian statistics relates to the methodology. 
    Respond with just the summary text on a single line and do not include an introductory message.
    [/INST]

    Sure! Here's the summary you requested:
    """

    prompt = PromptTemplate(
        input_variables=["summaries"],
        template=template,
    )
    return prompt


def bayespub_rag_prompt():
    template = """<s>[INST] <<SYS>>
    You are an academic who is familiar with both Bayesian and Frequentist statistics.
    You have strong general knowledge of medical academic literature.
    Answer the question '{question}' based on the context.
    Keep the answer less than around 200 words.
    <</SYS>>

    Context:
    {context}

    Question:
    {question}
    [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template,
    )
    return prompt
