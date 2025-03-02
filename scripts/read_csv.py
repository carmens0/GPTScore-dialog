import pandas as pd

def leggi_csv(file_path, separatore=",", encoding="utf-8"):
    """
    Legge un file CSV e lo restituisce come un DataFrame Pandas.

   
    """
    try:
        df = pd.read_csv(file_path, sep=separatore, encoding=encoding)
        print(f"File {file_path} letto correttamente. Dimensioni: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Errore: il file {file_path} non è stato trovato.")
    except pd.errors.ParserError:
        print(f"Errore: impossibile analizzare il file {file_path}. Verifica il formato e il separatore.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

df = leggi_csv(r"C:\Users\carme\Desktop\dialogue-reponse\correlation\GPTScore_gpt-4o-mini.csv")
print(df)

