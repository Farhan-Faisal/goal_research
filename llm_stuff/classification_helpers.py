import pandas as pd
import os
import shutil
import config
import json

from tqdm import tqdm
from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import OutputParserException
from langchain.output_parsers import PydanticOutputParser


class Label(BaseModel):
    goals: List[dict[str,str]] = Field(description="List of goals-label pairs")


def write_log(completed_batches, batch_size, classification_log_path):

    data = {"completed_batches": completed_batches, "batch_size": batch_size}
    jstr = json.dumps(data, indent=4)
    with open(classification_log_path, 'w', encoding='utf-8') as outfile:
        json.dump(jstr, outfile, ensure_ascii=False)


def get_required_batches(df):
    required_batches = []
    counter = len(df)
    i = 1
    while counter >= 10:
        required_batches.append(i)
        i = i + 1
        counter = counter - 10

    if counter == 0:
        required_batches.append(i)

    return required_batches


def get_goal_classification_batch(data_dict, 
                                  model_name, temperature, prompt,
                                  start_index, end_index):

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=3000,
        timeout=None,
        max_retries=2,
        api_key="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"
    )

    chain = prompt | model

    goalList = [x['Answers'] for x in data_dict][start_index:end_index]

    parser = PydanticOutputParser(pydantic_object=Label)

    input = prompt.format_prompt(goalList=goalList).to_string()
    
    output = chain.invoke(input)
    try:
        print("Classifying batch:", start_index/10)
        return parser.parse(output.content).goals
    except OutputParserException:
        return []


def classify_goals(batch_list, completed_batches, batch_size, 
                   df_dict, model_name, temperature, prompt,
                   output_directory, file_suffix, classification_log_path):

    Labels = []
    for item in tqdm(batch_list):
        if item in completed_batches:
            continue
        else:
            end_index = item*batch_size
            start_index = end_index - batch_size

            while True:
                try:
                    Labels = get_goal_classification_batch(df_dict, 
                                            model_name, temperature, prompt,
                                            start_index, end_index)
                    
                    for i in range(len(Labels)):
                        try:
                            df_dict[i + start_index]['label'] = Labels[i]['label']
                        except KeyError:
                            try:
                                df_dict[i + start_index]['label'] = list(Labels[i].values())[0]
                                continue
                            except:
                                # Skip to the next while loop iteration on KeyError
                                print(list(Labels[i].values()))
                                raise Exception("KeyError encountered")
                                # df_dict[i + start_index]['label'] = "!!!FIX_ME!!!"
                    break
                except Exception as e:
                    print(f"KeyError occurred: {e}. Retrying with current batch.")
                    continue

            pd.DataFrame(df_dict[start_index:end_index]).to_csv(output_directory + "/" + file_suffix + "_" + str(item) + ".csv")
            completed_batches.append(item)

    write_log(completed_batches, batch_size, classification_log_path)


def delete_batches(directory):
    # Loop through the contents of the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file or link and remove it
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        
        # If it's a directory, use shutil.rmtree() to remove it and its contents
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
