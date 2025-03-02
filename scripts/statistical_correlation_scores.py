import os
import json
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import cohen_kappa_score
from tabulate import tabulate


def display_results_as_table(results):
    """
    Mostra i risultati in una tabella formattata.

    Parametri:
        results (dict): Dizionario con i risultati delle metriche.
    """
    table = [
        ["Spearman's Correlation", f"{results['spearman_corr']:.3f}", f"{results['spearman_pval']:.3f}"],
        ["Pearson's Correlation", f"{results['pearson_corr']:.3f}", f"{results['pearson_pval']:.3f}"],
        ["Kendall-Tau Correlation", f"{results['kendall_corr']:.3f}", f"{results['kendall_pval']:.3f}"]
    ]

    headers = ["Metric", "Value", "P-Value"]
    print(tabulate(table, headers=headers, tablefmt="grid"))


def calculate_and_save_correlations(json_file, model="gpt-4o-mini",  output_dir=None):
    """
    Calcola correlazioni (Cohen's Kappa, Spearman, Pearson, Kendall-Tau) tra eval_score e GPT scores.
    Salva i risultati in file CSV e visualizza la tabella dei risultati.

    Parametri:
        json_file (str): Percorso al file JSON.
        method (str): Metodo da considerare ("VAL", "IST", "BOTH").
        output_dir (str): Directory dove salvare i risultati.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        valid_data = [conv for conv in data if "eval_score" in conv and conv["eval_score"] is not None]

        if not valid_data:
            raise ValueError("Nessun dato valido con eval_score trovato.")

        if output_dir is None:  
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "../correlations_results")            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  
        else:
            if not os.path.exists(output_dir):  
                os.makedirs(output_dir)
        
        print(f"Risultati verranno salvati nella directory: {output_dir}")



        def compute_correlations(human_scores, gpt_scores, title):
            """
            Calcola le correlazioni tra due insiemi di punteggi, visualizza e salva i risultati.

            Parametri:
                human_scores (list): Punteggi umani.
                gpt_scores (list): Punteggi GPT.
                title (str): Titolo per il file CSV.
            """
            spearman_corr, spearman_pval = spearmanr(human_scores, gpt_scores)
            pearson_corr, pearson_pval = pearsonr(human_scores, gpt_scores)
            kendall_corr, kendall_pval = kendalltau(human_scores, gpt_scores)

            results = {
                "spearman_corr": spearman_corr,
                "spearman_pval": spearman_pval,
                "pearson_corr": pearson_corr,
                "pearson_pval": pearson_pval,
                "kendall_corr": kendall_corr,
                "kendall_pval": kendall_pval
            }

            print(f"\nResults for {title}:")
            display_results_as_table(results)

            results_csv = {
                "Metric": ["Cohen's Kappa", "Spearman Correlation", "Pearson Correlation", "Kendall-Tau"],
                "Value": [
                    results["cohen_kappa"],
                    results["spearman_corr"],
                    results["pearson_corr"],
                    results["kendall_corr"]
                ],
                "P-Value": ["-", results["spearman_pval"], results["pearson_pval"], results["kendall_pval"]]
            }

            df = pd.DataFrame(results_csv)
            csv_filename = os.path.join(output_dir, f"{title}.csv")
            df.to_csv(csv_filename, index=False)
            print(f"Tabella '{title}' salvata in {csv_filename}")

        human_scores = [conv["eval_score"] for conv in valid_data]

        
        raw_ist_scores = [conv.get(f"gpt_score_{model}") for conv in valid_data]
        
        print("Calculating correlations ...")
        compute_correlations(human_scores, raw_ist_scores, f"GPTScore_{model}")

        print("Correlation calculation completed.")

    except Exception as e:
        print(f"Errore: {e}")


