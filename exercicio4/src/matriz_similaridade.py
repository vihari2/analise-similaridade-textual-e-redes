"""Cálculo em lote e exportação da matriz de similaridade textual (Etapa 2)."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from similaridade import demonstrar_levenshtein


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_ENTRADA = RAIZ_PROJETO / "resultados" / "tweets_tratados.csv"
ARQUIVO_MATRIZ = RAIZ_PROJETO / "resultados" / "matriz_similaridade.csv"
ARQUIVO_TEMAS = RAIZ_PROJETO / "resultados" / "metadados_similaridade.csv"
ARQUIVO_HISTOGRAMA = RAIZ_PROJETO / "resultados" / "histograma_similaridade.png"
SAIDAS = (ARQUIVO_MATRIZ, ARQUIVO_TEMAS, ARQUIVO_HISTOGRAMA)


def carregar_textos() -> tuple[list[str], pd.Series]:
    """Lê somente as colunas necessárias e mantém o tema de cada tweet."""
    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {ARQUIVO_ENTRADA}")
    dados = pd.read_csv(ARQUIVO_ENTRADA, usecols=["texto_limpo", "tema"])
    if dados[["texto_limpo", "tema"]].isna().any().any():
        raise ValueError("As colunas 'texto_limpo' e 'tema' não podem conter valores vazios.")
    textos = dados["texto_limpo"].astype(str).tolist()
    temas = dados["tema"].astype(str).rename("tema")
    if not textos:
        raise ValueError("O CSV não contém tweets para comparar.")
    return textos, temas


def calcular_matriz(textos: list[str]) -> np.ndarray:
    """Calcula a matriz float32, reduzindo o uso de memória."""
    # cdist produz diretamente float32; evitar float64 reduz pela metade a RAM.
    matriz = process.cdist(
        textos,
        textos,
        scorer=Levenshtein.normalized_similarity,
        dtype=np.float32,
        workers=-1,
    )
    # Copia cada triângulo para o outro, assegurando simetria sem outra matriz.
    for indice in range(matriz.shape[0]):
        matriz[indice + 1 :, indice] = matriz[indice, indice + 1 :]
    np.fill_diagonal(matriz, 1.0)
    return matriz


def salvar_resultados(matriz: np.ndarray, temas: pd.Series) -> None:
    """Salva matriz com IDs estáveis, metadados de tema e histograma."""
    quantidade = matriz.shape[0]
    ids = np.arange(quantidade)
    colunas = [f"tweet_{i}" for i in ids]
    quadro = pd.DataFrame(matriz, columns=colunas)
    quadro.insert(0, "tweet_id", ids)
    quadro.to_csv(ARQUIVO_MATRIZ, index=False, float_format="%.6f")

    metadados = pd.DataFrame({"tweet_id": ids, "tema": temas.to_numpy()})
    metadados.to_csv(ARQUIVO_TEMAS, index=False, encoding="utf-8")

    # Usa cache temporário, pois alguns ambientes não permitem escrita na pasta pessoal.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "exercicio4-matplotlib"))
    # Importação local para permitir identificar a falta da dependência claramente.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.hist(matriz.ravel(), bins=50, range=(0, 1), color="steelblue", edgecolor="white")
    plt.title("Distribuição da similaridade de Levenshtein")
    plt.xlabel("Similaridade normalizada")
    plt.ylabel("Quantidade de pares")
    plt.tight_layout()
    plt.savefig(ARQUIVO_HISTOGRAMA, dpi=150)
    plt.close()


def executar(forcar: bool = False) -> None:
    """Reutiliza resultados completos existentes, salvo quando forçado."""
    demonstrar_levenshtein()
    if all(arquivo.exists() for arquivo in SAIDAS) and not forcar:
        print("\nAs saídas já existem; recálculo ignorado. Use --force para recalcular.")
        for arquivo in SAIDAS:
            print(f"Arquivo existente: {arquivo}")
        return

    textos, temas = carregar_textos()
    matriz = calcular_matriz(textos)
    simetrica = bool(np.array_equal(matriz, matriz.T))
    diagonal_unitaria = bool(np.all(np.diag(matriz) == 1.0))
    if not simetrica or not diagonal_unitaria:
        raise RuntimeError("A matriz não passou nas verificações de simetria e diagonal.")

    salvar_resultados(matriz, temas)
    print(f"\nTweets comparados: {matriz.shape[0]}")
    print(f"Temas: {temas.value_counts().to_dict()}")
    print(f"Matriz simétrica: {simetrica}; diagonal igual a 1: {diagonal_unitaria}")
    print(f"Similaridade mínima: {matriz.min():.6f}")
    print(f"Similaridade máxima: {matriz.max():.6f}")
    print(f"Similaridade média: {matriz.mean():.6f}")
    for arquivo in SAIDAS:
        print(f"Arquivo gerado: {arquivo}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="recalcula mesmo se os arquivos de saída existirem"
    )
    args = parser.parse_args()
    try:
        executar(forcar=args.force)
        return 0
    except Exception as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
