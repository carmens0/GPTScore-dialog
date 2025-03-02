import openai
import sys
from transformers import GPT2Tokenizer
import re


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-o4-mini",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,  
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = openai.chat.completions.create(**params)
    return completion


class GPT4Model:
    
    def __init__(self, model_name, api_key):
        '''
        **args: 
        - model_name: Il nome del modello GPT-3 da utilizzare (ad esempio, "text-davinci-003").
        - api_key: La chiave API di OpenAI, utilizzata per autorizzare l'accesso ai modelli GPT-3.'''
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception as e:
            print(f"Api Key Error: : {e}")
            sys.exit(1)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

        
    def do_inference(self, prompt, max_length=2048):
        """
        Esegue il calcolo della perdita (loss) per un dato prompt.
        """
        response = self.gpt4(prompt, max_len=max_length, num_log_probs=5, echo=True)
        out = response.choices[0]
        token_logprobs = out.logprobs.content  
        filtered_logprobs = [token.logprob for token in token_logprobs if token.token != '.']
        total_loss = -sum(filtered_logprobs)  
        avg_loss = total_loss / len(filtered_logprobs) if filtered_logprobs else 0

        response_content = response.choices[0].message.content.strip()
        print(response_content)
        #punt = extract_score_from_response(response_content)
        return avg_loss

    def gpt4(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        """
        Richiama il modello GPT-3 per ottenere log-probabilities.
        """
        response = None
        received = False
        while not received:
            try:
                response = get_completion(
                    [{"role": "user", "content": prompt}], 
                    model=self.model_name,
                    logprobs=True,
                    top_logprobs=2,
                    max_tokens=max_len,
                    temperature=temp,
                    #echo=echo,
                    stop='\n',
                    #n=n,
                )
                received = True

            except Exception as e:
                print(f"Error API: {e}")
                sys.exit(1)
        return response

def extract_score_from_response(response_content):
        """
        Estrae un punteggio numerico da una risposta testuale usando regex.
        """
        match = re.search(r"\b\d+\b", response_content)  
        if match:
            return float(match.group(0))  
        else:
            print(f"Error: impossibile to exctract a score from  '{response_content}'")
            return None