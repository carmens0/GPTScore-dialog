from scripts import gpt3score
import json
import tqdm 
import scipy
import tabulate
import os

file_path=r"\datasets\fed_data 1.json"
api_key=""
results_file = r"\results\results_fed_DIALOGUE.json"

with open(file_path, "r") as file:
        fed_data = json.load(file)



def evaluate( dialog, gpt3model, api_key, method): 
    result = gpt3score(dialog, gpt3model, api_key=api_key, method=method)
    gpt_score=result["raw_score"] 
    return gpt_score

 


if os.path.exists(results_file):
    with open(results_file, "r") as file:
        results = json.load(file)
else:
    results = []

# TURN - LEVEL
for dialog_id, example in enumerate(tqdm.tqdm(fed_data)): 
    if any(result["dialog_id"] == dialog_id for result in results):
        continue    
    conversation = example["context"]
    response = example.get("response")
    system = example['system']
    conversation = conversation.split("\n")
    conversation = [s.replace("User: ", "Human:").replace("System: ", "AI: ").strip() for s in conversation]
    print(f"Processing dialog {dialog_id}: ... ")

    if response is None:
        continue    
    response = response.replace("User: ", "Human:").replace("System: ", "AI:").strip()
    
    full_conversation = " ".join(conversation) + " Response:" + response
    print(full_conversation)
    mean_annotation = sum(example["annotations"]["Overall"]) / len(example["annotations"]["Overall"])    
    predicted_value = evaluate(full_conversation, gpt3model="davinci-002",api_key=api_key, method="turn-level")    
    result = {
        "dialog_id": dialog_id,
        "human_score": mean_annotation,
        "predicted_value": predicted_value
    }
    
    results.append(result)    
    with open(results_file, "w") as file:
        json.dump(results, file, indent=4)



 
 
 
# DIALOG-LEVEL
for dialog_id, example in enumerate(tqdm.tqdm(fed_data)): 
    if any(result["dialog_id"] == dialog_id for result in results):
        continue    
    conversation = example["context"]
    print(f"Processing dialog {dialog_id}: ... ")

    response = example.get("response")
    system = example["system"]
    conversation = conversation.split("\n")
    conversation = [s.replace("User: ", "Human :").replace("System: ", " AI:").strip() for s in conversation]
    full_conversation = " ".join(conversation)
    if response is not None:
        continue
 
    mean_annotation = sum(example["annotations"]["Overall"]) / len(example["annotations"]["Overall"])
    predicted_value = evaluate(full_conversation, gpt3model="davinci-002",api_key=api_key, method="dialogue-level")    
    result = {
        "dialog_id": dialog_id,
        "human_score": mean_annotation,
        "predicted_value": predicted_value
    }
    
    results.append(result)    
    with open(results_file, "w") as file:
        json.dump(results, file, indent=4)

