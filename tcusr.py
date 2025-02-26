import tqdm
import json 
from  scripts import gpt3score
import os
import scipy
file_path=r"C:\Users\carme\Desktop\dialogue-reponse\datasets\tc_usr_data 1.json"
api_key="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"
results_file = r"C:\Users\carme\Desktop\dialogue-reponse\results\results_tcusr.json"

with open(file_path, "r") as file:
    tcusr = json.load(file)



def evaluate( dialog, gpt3model, api_key, method): 
    result = gpt3score(dialog, gpt3model, api_key=api_key, method=method)
    gpt_score=result["raw_score"] 
    return gpt_score





annotations = []
evaluations = []
results = []
 
for i in tqdm.tqdm(range(len(tcusr))):
    print(f"Processing dialog {i}: ... ")
    dialogue_data = tcusr[i]
    context = dialogue_data["context"].replace("\n","")
 
    responses = dialogue_data["responses"]
 
    for response_data in responses:
        response = response_data["response"].split("\n")[0]
        human_score = sum(response_data["Overall"]) / len(response_data["Overall"])
        full_conversation= f"{context} Response: {response}"
        model_evaluation = evaluate(full_conversation, gpt3model="davinci-002",api_key=api_key, method="turn-level" )
        annotations.append(human_score)
        evaluations.append(model_evaluation)
 
        result={
            "dialog_id": i,
            "context": context,
            "response": response,
            "human_score": human_score,
            "predicted_score": model_evaluation
        }
        results.append(result)    
        with open(results_file, "w") as file:
            json.dump(results, file, indent=4)
 
 
