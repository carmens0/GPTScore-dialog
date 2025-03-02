
from .gpt_inference import GPT4Model
import json
import re

def normalize_score(value):
    """
    Normalizza un valore qualsiasi tra 0 e 5.
    Usa una compressione lineare fissa.
    """
    compressed_value = 1 / (1 + abs(value))  
    normalized_score = compressed_value * 5  
    return normalized_score



def gpt4score(dialogue, aspect="interesting", gpt3model=None, api_key=None, method="VAL"):
    """
    Calcola GPTScore per un dialogo completo, includendo il metodo (VAL o IST).
    **args: 
    - dialogue: dialogo formattato in modo tale da essere inserito correttamente nel prompt
    - aspect : di default l'aspetto da valutare è intersting
    - gpt3model: nome del modello di opena da utilizzare
    - method : di default è VAL, si può scegliere tra VAL, IST e BOTH 

    """
    gpt3model_name = {
        "gpt-4o": "gpt-4o",
        "gpt-3.5": "gpt-3.5-turbo",
        "gpt-4o-mini": "gpt-4o-mini",
    }.get(gpt3model, "gpt-4o-mini")  

    if method=="dialogue-level": 
        prompt = (
                f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the overall quality of the dialogue satisfactory? (a) Yes. (b) No.\nConversation: {dialogue}\nAnswer: Yes."
            )
    elif method=="turn-level": 
        prompt=(f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the overall quality of the AI's most recent response satisfactory? (a) Yes. (b) No.\nConversation: {dialogue}\nAnswer: Yes.")
   
    else:
        raise ValueError(f"Invalid method: {method}. Use 'dialogue-level' or 'turn-level'.")

    metaicl_model = GPT4Model(gpt3model_name, api_key)
    avg_loss = metaicl_model.do_inference(prompt)
    score = -avg_loss
    return { "raw_score": score}




