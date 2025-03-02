import time
import sys
from transformers import GPT2Tokenizer
import openai


'''La classe GPT3Model che interagisce con l'API di OpenAI per eseguire inferenze utilizzando il modello GPT-3. 
La classe è progettata per calcolare una "perdita" (loss) media durante l'inferenza, utilizzando anche il
tokenizer di GPT-2 per tokenizzare l'input'''


class GPT3Model(object):

    def __init__(self, model_name, api_key, logger=None):
        '''- model_name: Il nome del modello GPT-3 da utilizzare (ad esempio, "text-davinci-003").
            - api_key: La chiave API di OpenAI, utilizzata per autorizzare l'accesso ai modelli GPT-3.
            - logger: Un'opzione per passare un logger per la registrazione delle attività, se necessario (non viene usato nel codice che hai fornito).
        Dentro il costruttore:
            La chiave API viene impostata tramite openai.api_key = api_key.
            Viene caricato il tokenizer di GPT-2 usando GPT2Tokenizer.from_pretrained("gpt2-xl") 
'''
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger

    def do_inference(self, prompt, max_length=2048):
        '''esegue il calcolo della "perdita" (loss) media durante il completamento del modello GPT-3.
    Parametri:
    input: Il testo di input che sarà completato dal modello.
    output: Il testo che si prevede che venga generato come risposta dal modello.
    max_length: La lunghezza massima del completamento generato (di default è 2048 token).'''
        response = self.gpt3(prompt, max_len=max_length, num_log_probs=5, echo=True)
        out = response.choices[0] 
        i = 0
        i = out.logprobs.text_offset.index(len(prompt) - 1)
        if i == 0:
            i = i + 1

        print('eval text', out.logprobs.tokens[i: -1])
        loss = -sum(out.logprobs.token_logprobs[i:-1]) # ignore the last '.'
        if len(out.logprobs.text_offset) - i - 1 > 0:
            avg_loss = loss / (len(out.logprobs.text_offset) - i - 1)
        else:
            avg_loss = None
        return avg_loss


    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        '''
        Parametri:
        - prompt: Il testo che viene passato al modello GPT-3 per completarlo.
        - max_len: La lunghezza massima del completamento.
        - temp: La temperatura, che influisce sulla casualità del completamento (un valore tra 0 e 1).
        - num_log_probs: Il numero di log-probabilità da restituire (utile per il calcolo della perdita).
        - echo: Se impostato su True, il modello restituirà anche il prompt originale.
        - n: Il numero di risposte da generare.
        '''
        
        response = None
        received = False
        while not received:
            try:
                response = openai.completions.create(model=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                received = True
            except:
                #Se c'è un errore nell'interazione con l'API (ad esempio, se il prompt
                # è troppo lungo o se c'è un altro errore nell'API), il codice stampa l'errore
                # e aspetta un secondo prima di riprovare.
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response

