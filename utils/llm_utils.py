from typing import List, Dict, Any
import time, random
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
import numpy as np

def get_batch_classification_by_llm(
    label,
    data_dict,
    llm,
    prompt, 
    start_index,
    end_index,
):
    # Slice goals (assuming data_dict is a list[dict] with key "Answers")
    goal_list = [x["Answers"] for x in data_dict][start_index:end_index]

    parser = PydanticOutputParser(pydantic_object=label)

    try:
        msgs = prompt.format_messages(goalList=goal_list)
        resp = llm.invoke(msgs)
        text = resp.content
        print(text)

    except AttributeError:
        try:
            text_input = prompt.format_prompt(goalList=goal_list).to_string()
        except AttributeError:
            text_input = str(prompt).format(goalList=goal_list)
        
        resp = llm.invoke(text_input)
        text = resp.content

    return parser.parse(text).goals


def embed_batch(texts, client, model):
    """
        Call OpenAI once for a list[str] -> list[list[float]]
    """
    resp = client.embeddings.create(model=model, input=texts)
    # Order is preserved by the API
    return [d.embedding for d in resp.data]


def embed_with_retries(texts, model, client, max_retries):
    """
        Retry on transient errors / rate limits with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return embed_batch(texts, client, model=model)
        except Exception as e:
            # Backoff with jitter
            sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[embed] error: {e}. retrying in {sleep_s:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_s)
    return [[np.nan]*3072 for _ in texts]