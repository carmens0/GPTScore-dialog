import uuid
import json
from scripts import  generate_bot, process_dialogs, gpt4score, gpt3score, calculate_and_save_correlations, save_scores_to_dialog
import os
import sys
if __name__ == "__main__":
    print("=" * 50)
    print("🚀  Welcome to the Evaluation of Generative Dialogue Responses  🚀".center(50))
    print("=" * 50)
    print("\n🔷 Please select one of the following actions:\n")
    print("   1️⃣  💬 Try a bot and evaluate")
    print("   2️⃣  🧠 Test models for GPT score")
    print("   3️⃣  📊 Calculate correlations (val_score ↔ gpt_score)")
    print("   4️⃣  ❌ Exit")
    action = int(input("👉 Select an action (1-4): "))
    while True:
        if action == 1:
            print("💬 TRY A BOT AND EVALUATE")
            print("🤖  ChatBot: Hi! I'm your chatbot. Let's start chatting!")
            dialog_id=generate_bot()
            print("📊  Let's use a model for evaluate the conversation!")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "datasets\conversation.json")
            api_key=str(input("The api_key: "))
            model_name=str(input("The model name you would like to use (choose between: gpt-4o,gpt-3.5, davinci-002 and gpt-4o-mini) : " ))
            dialogs = process_dialogs(file_path)
            scores = []
            for dialog in dialogs:
                if dialog["dialog_id"] != dialog_id:
                    continue
            # if dialog is None:
            #     print(f"Error: No dialog found with dialog_id={dialog_id}.")
            #     break
                print(f"Processing Dialog {dialog}...")


                gpt_scpre = gpt3score(dialogue, gpt3model=model_name, api_key=api_key, method=method)
                gpt_scpre.update({"dialog_id": dialog["dialog_id"], "model": model_name})
                scores.append(gpt_scpre)


                print("\nGPTScore Results:")
                for result in scores:
                    metric_name = f"gpt_score_{result['model']}"
                    print(f"Dialog {result['dialog_id']} ({metric_name}): {result['raw_score']}")

            save_scores_to_dialog(file_path, dialog_id, scores)
            break
        if action == 2: 
            print("🧠 TEST MODELS FOR GPT SCORE")
            print("📊  Let's use a model for evaluate GPTscore!")
            print("The implemented model evaluates both TURN and DIALOG  level as follows: \n ") 
            print("a. TURN (Turn Level):Evaluation conducted at the granularity of individual conversational turns, focusing on the quality of a single response in the context of its preceding input.")
            print("b. DIALOG (Dialog Level): Evaluation performed across the entire conversation, assessing the coherence, relevance, and overall quality of the interaction as a whole. ")
            print("Change the settings if you only want one of them. ")
            print("The task evaluated is INT (interest), but it's possibile to change it. ")
            print("There is some information required to use this application.")
            file_path = str(input('Define the file path of the json file (dialogue - response) : '))
            api_key=str(input("The api_key: "))
            model_name=str(input("The model name you would like to use (choose between: gpt-4o,gpt-3.5 and gpt-4o-mini) : " ))
            method=str(input("The kind of method (choose between dialogue-level or turn-level): "))
           
            if model_name=="davinci-002": 
                dialogs=process_dialogs(file_path)
                scores = []
                with open(file_path, "r") as file:
                    data = json.load(file)
                for i, dialog in enumerate(dialogs):
                    dialogue = dialog["dialogue"]
                    print(f"Processing Dialog {dialog['dialog_id']}, {i}...")
                    gpt_scpre = gpt3score(dialogue, gpt3model=model_name, api_key=api_key, method=method)
                    gpt_scpre.update({"dialog_id": dialog["dialog_id"], "model": model_name})
                    
                    if data[i]["dialog_id"] == gpt_scpre["dialog_id"]:
                        metric_name_raw = f"gpt_score_{model_name}"
                        data[i][metric_name_raw] = gpt_scpre["raw_score"]

                    with open(file_path, "w") as file:
                        json.dump(data, file, indent=4)
                    scores.append(gpt_scpre)
                print("\nGPTScore Results:")
                for result in scores:
                    metric_name = f"gpt_score_{result['model']}"
                    print(f"Dialog {result['dialog_id']} ({metric_name}): {result['raw_score']}")

            else:  
                dialogs=process_dialogs(file_path)
                scores = []
                with open(file_path, "r") as file:
                    data = json.load(file)
                for i, dialog in enumerate(dialogs):
                    dialogue = dialog["dialogue"]
                    print(f"Processing Dialog {dialog['dialog_id']}, {i}...")
                    ist_score_data = gpt4score(dialogue, gpt3model=model_name, api_key=api_key, method=method)
                    ist_score_data.update({"dialog_id": dialog["dialog_id"], "model": model_name})
                    
                    if data[i]["dialog_id"] == ist_score_data["dialog_id"]:
                        metric_name_raw = f"gpt_score_{model_name}"
                        data[i][metric_name_raw] = ist_score_data["raw_score"]

                    with open(file_path, "w") as file:
                        json.dump(data, file, indent=4)
                    scores.append(ist_score_data)
                print("\nGPTScore Results:")
                for result in scores:
                    metric_name = f"gpt_score_{result['model']}"
                    print(f"Dialog {result['dialog_id']} ({metric_name}): {result['raw_score']}")
                #save_scores_to_file(file_path, scores)
            corr=str(input('Would you like to compute correlations between gptscore and human score? (yes or no) : '))
            if corr=="no":
                print("Closing the application") 
                sys.exit()
            elif corr=="yes" or corr=="si": 
                print("Welcome in correlation task")
                if not os.path.exists(file_path):
                    print("Error: path doesn't exists")
                output_dir = input("Outhput directory for table of results (default: 'correlation_results'): ").strip()
                if not output_dir:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    output_dir = os.path.join(current_dir, "../correlations_results")
                calculate_and_save_correlations(file_path, model=model_name, output_dir=output_dir)
    
            else: 
                corr=str(input('Invalide choice. Please choose between : yes or no : '))    
        if action==3: 
            print("📊 CALCULATE STATISTICAL CORRELATIONS")
            print("📈  Analyze correlations between human scores and GPT scores.")
            file_path=str(input("file path of .json data with gpt_scores: "))
            model_name=str(input("Model : "))
            if not os.path.exists(file_path):
                print("Error: path doesn't exists")
            output_dir = input("Outhput directory for table of results (default: 'correlation_results'): ").strip()
            if not output_dir:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(current_dir, "../correlations_results")
            calculate_and_save_correlations(file_path, model=model_name, output_dir=output_dir)

                
        if action==4: 
            print("👋  Thank you for using the system. Goodbye!")
            sys.exit()
        else:
            print("❌ Invalid choice. Please select a valid action (1-4).\n")

            action = int(input('Please, insert a supported action: '))



        
