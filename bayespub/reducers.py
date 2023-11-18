

def reduce_any(x) -> bool:
    return any(x)


def concatenate_summaries(summaries:list) -> str:
    return dict(summaries="\n\n".join(["Summary:\n"+summary for summary in summaries]))