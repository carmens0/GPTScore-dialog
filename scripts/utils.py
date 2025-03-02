import json

def process_dialogs(file_path):
    """
    Processa i dialoghi dal file JSON e restituisce un elenco di dialoghi completi.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    dialogs = []
    for entry in data:
        dialog_id = entry["dialog_id"]
        dialog_turns = entry["dialog"]

        formatted_dialogue = "\n".join(
            [f"{'Human' if turn['sender'] == 'participant1' else 'AI'}: {turn['text']}" for turn in dialog_turns]
        )

        dialogs.append({"dialog_id": dialog_id, "dialogue": formatted_dialogue})
    print(dialogs)
    return dialogs


def save_scores_to_file(file_path, scores):
    """
    Salva i risultati dei punteggi GPT nel file JSON originale, differenziando per metodo.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    for dialog in data:
        dialog_id = dialog["dialog_id"]
        matching_scores = [score for score in scores if score["dialog_id"] == dialog_id]
        for score_data in matching_scores:
            method = score_data["method"]
            model = score_data["model"]
            metric_name = f"gpt_score_{model}_{method.lower()}"
            dialog[metric_name] = score_data["normalized_score"]
            metric_name_raw = f"raw_gpt_score_{model}_{method.lower()}"
            dialog[metric_name_raw] = score_data["raw_score"]
            metric_name_3=f"eval_score_{model}_{method.lower()}"
            dialog[metric_name_3] = score_data["eval_score"]

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"GPT scores successfully saved in '{file_path}'!")

def save_scores_to_dialog(file_path, dialog_id, scores):

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error:  file '{file_path}' doesn't exist.")
        return
    except json.JSONDecodeError:
        print(f"Error: file '{file_path}' isn't a valid JSON format.")
        return

    dialog = next((d for d in data if d["dialog_id"] == dialog_id), None)
    if dialog is None:
        print(f"Errore: Nessun dialogo trovato con dialog_id='{dialog_id}'.")
        print(f"Debug: dialog_id passato: '{dialog_id}', tipi: {type(dialog_id)}")
        print(f"Debug: Lista dialog_id trovati: {[d['dialog_id'] for d in data]}")
        return


    for score_data in scores:
        if score_data["dialog_id"] != dialog_id:
            continue

        method = score_data["method"]
        model = score_data["model"]
        metric_name = f"gpt_score_{model}_{method.lower()}"
        dialog[metric_name] = score_data["normalized_score"]
        metric_name_raw = f"raw_gpt_score_{model}_{method.lower()}"
        dialog[metric_name_raw] = score_data["raw_score"]
        metric_name_3 = f"eval_score_{model}_{method.lower()}"
        dialog[metric_name_3] = score_data["eval_score"]

    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"GPT scores successfully added to the dialog with id '{dialog_id}' in '{file_path}'!")
    except Exception as e:
        print(f"Error saving file: {e}")
