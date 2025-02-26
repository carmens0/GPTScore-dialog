from scripts import gpt3score
import json
import tqdm 
import scipy
import os



annotations = []
evaluations = []
results = []

file_path= r"\datasets\dstc9_data 2.json"
model_name="davinci-002"
api_key=""
results_file = r"\datasets\results.json"



with open(file_path, "r") as file:
    dstc9_data = json.load(file)

def evaluate( dialog, gpt3model, api_key, method): 
    result = gpt3score(dialog, gpt3model, api_key=api_key, method="turn-level")
    gpt_score=result["raw_score"] 
    return gpt_score

 


if os.path.exists(results_file):
    with open(results_file, "r") as file:
        results = json.load(file)
else:
    results = []

for i in tqdm.tqdm(range(len(dstc9_data["contexts"]))):
    if any(result["dialog_id"] == i for result in results):
        continue  

    dialog = " ".join(dstc9_data["contexts"][i])
    dialog = f"{dialog} Response: {dstc9_data['responses'][i]}"
    print(f"Processing dialog {i}: ...")

    model_evaluation = evaluate(dialog, model_name, api_key, method="turn-level")
    annotation = dstc9_data["scores"][i]

    result_entry = {
        "dialog_id": i,
        "human_score": annotation,
        "predicted_score": model_evaluation,
    }
    results.append(result_entry)

    with open(results_file, "w") as file:
        json.dump(results, file, indent=4)

