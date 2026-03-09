import pandas as pd
import sys
from pathlib import Path


def generate_ranking(csv_path, output_csv=None, to_latex=True):
    # Lê o CSV
    df = pd.read_csv(csv_path)

    if "balanced_accuracy_mean" not in df.columns:
        raise ValueError("Coluna 'balanced_accuracy_mean' não encontrada no CSV.")

    # Ordena em ordem decrescente
    df_sorted = df.sort_values(by="balanced_accuracy_mean", ascending=False)

    # Seleciona top 10
    top10 = df_sorted.head(10).copy()

    # Adiciona coluna de ranking
    top10.insert(0, "rank", range(1, len(top10) + 1))

    if to_latex:
        # Gera tabela LaTeX rotacionada
        latex_table = top10.to_latex(
            index=False,
            float_format="%.4f",
            escape=True
        )

        print(r"\begin{sidewaystable}[ht]")
        print(r"\centering")
        print(r"\caption{Top 10 resultados ordenados por balanced\_accuracy\_mean}")
        print(r"\label{tab:top10_balanced_accuracy}")
        print(latex_table)
        print(r"\end{sidewaystable}")

        print("\n% IMPORTANTE: incluir no preâmbulo:")
        print("% \\usepackage{rotating}")

    else:
        if output_csv is None:
            output_csv = Path(csv_path).stem + "_top10_ranked.csv"

        top10.to_csv(output_csv, index=False)
        print(f"Arquivo salvo em: {output_csv}")


if __name__ == "__main__":
    dataset = "radiomics_lgg"
    csv_path = f"outputs/{dataset}/{dataset}_benchmark_results.csv"

    # Se passar --csv, gera CSV ao invés de LaTeX
    to_latex = True

    generate_ranking(csv_path, to_latex=to_latex)