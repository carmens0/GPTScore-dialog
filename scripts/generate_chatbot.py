import json
import uuid
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(conversation_history, user_input, max_length=100):
    '''Funzione per generare la risposta da un bot
    **args: 
        - conversation_history: cronologia del dialogo; 
        - user_input: input passato dall'utente nel terminal
        - max_length: massima lunghezza della risposta del bot (di default settata a 100) '''
    conversation_history += f" {user_input} {tokenizer.eos_token}"

    inputs = tokenizer(
        conversation_history,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + max_length,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs["attention_mask"], 
            do_sample=True,
            temperature=0.8, # creatività (esplorazione) nel costruire la frase
            top_p=0.9 # probabilità cumulativa
        )

    response = tokenizer.decode(outputs[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)
    return response.strip(), conversation_history + response.strip()

def ask_for_rating():
    while True:
        try:
            rating = int(input("On a scale of 0 to 5, how interesting was this conversation? "))
            if 0 <= rating <= 5:
                return rating
            else:
                print("Please enter a number between 0 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 5.")

def generate_bot(): 
    conversation_history = ""
    dialog_id = f"0x{uuid.uuid4().hex[:8]}"  
    dialog = []
    try:
        message_id = 0
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "esci"]:
                print("ChatBot: It was a pleasure talking with you, see you soon!")
                break
            bot_response, conversation_history = generate_response(conversation_history, user_input)
            dialog.append({"id": message_id, "sender": "participant1", "text": user_input})
            dialog.append({"id": message_id + 1, "sender": "participant2", "text": bot_response})
            message_id += 2
            print(f"ChatBot: {bot_response}")

    except KeyboardInterrupt:
        print("\nChatBot: Seems like you want to exit. It was a pleasure talking with you!")

    finally:
        eval_score = ask_for_rating()
        conversation_data = {
            "dialog_id": dialog_id,
            "dialog": dialog,
            "eval_score": eval_score
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, "../datasets/conversation.json")

        all_conversations = []
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        all_conversations = existing_data
                    elif isinstance(existing_data, dict):
                        all_conversations = [existing_data]
                except json.JSONDecodeError:
                    print("Warning: Failed to read existing conversation.json, starting fresh.")

        all_conversations.append(conversation_data)

        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)  # Crea la directory se non esiste
        with open(json_file_path, "w") as f:
            json.dump(all_conversations, f, indent=4)
        print(f"The conversation has been saved in '{json_file_path}'.")

    return dialog_id
