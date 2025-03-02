from .gpt3_inference import GPT3Model
import json
from .utils import process_dialogs

# Il codice definisce una funzione gpt3score che calcola un punteggio di "perdita"
# (loss) per un completamento generato da un modello GPT-3.
# La perdita è calcolata usando il metodo do_inference della classe GPT3Model
def gpt3score(dialogue, gpt3model=None,api_key=None, method=None):
    '''La funzione gpt3score prende quattro argomenti:
    - input: Il testo di input che viene passato al modello GPT-3.
    - output: Il testo previsto o generato che sarà confrontato con l'output del modello.
    - gpt3model: Un argomento opzionale che specifica quale modello GPT-3 utilizzare (può essere uno dei vari modelli come "ada", "babbage", "curie", ecc.).
    - api_key: La chiave API di OpenAI necessaria per autenticarsi e utilizzare l'API GPT-3.'''
    gpt3model_name = {
        "davinci-002": "davinci-002"
    }.get(gpt3model, "davinci-002")  
    if method=="dialogue-level": 
        prompt = (
                f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the overall quality of the dialogue satisfactory? (a) Yes. (b) No.\nConversation: {dialogue}\nAnswer: Yes."
            )
    if method=="turn-level": 
        prompt=(f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the overall quality of the AI's most recent response satisfactory? (a) Yes. (b) No.\nConversation: {dialogue}\nAnswer: Yes.")
   
    metaicl_model = GPT3Model(gpt3model_name, api_key) 
    avg_loss = metaicl_model.do_inference(prompt) 
    if avg_loss is None:
        score = None  
    else:
        score = -avg_loss  
    
    return { "raw_score": score}


