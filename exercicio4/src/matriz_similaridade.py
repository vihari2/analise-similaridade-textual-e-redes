
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
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {ARQUIVO_ENTRADA}"
        )

    dados = pd.read_csv(
        ARQUIVO_ENTRADA,
        usecols=["texto_limpo", "tema"],
    )

    if dados[["texto_limpo", "tema"]].isna().any().any():
        raise ValueError(
            "As colunas 'texto_limpo' e 'tema' não podem conter valores vazios."
        )

    textos = dados["texto_limpo"].astype(str).tolist()
    temas = dados["tema"].astype(str).rename("tema")

    if not textos:
        raise ValueError("O CSV não contém tweets para comparar.")

    return textos, temas


def calcular_matriz(textos: list[str]) -> np.ndarray:
    """Calcula a matriz float32, reduzindo o uso de memória."""

    matriz = process.cdist(
        textos,
        textos,
        scorer=Levenshtein.normalized_similarity,
        dtype=np.float32,
        workers=-1,
    )

    # Copia cada triângulo para o outro, assegurando simetria.
    for indice in range(matriz.shape[0]):
        matriz[indice + 1:, indice] = matriz[indice, indice + 1:]

    # Garante que a diagonal principal seja igual a 1.
    np.fill_diagonal(matriz, 1.0)

    return matriz


def obter_similaridades_pares(matriz: np.ndarray) -> np.ndarray:
    """
    Obtém as similaridades entre pares distintos,
    excluindo a diagonal e evitando comparações duplicadas.
    """

    indices = np.triu_indices_from(matriz, k=1)

    return matriz[indices]


def salvar_resultados(
    matriz: np.ndarray,
    temas: pd.Series,
    similaridades_pares: np.ndarray,
) -> None:
    """Salva matriz, metadados e histograma."""

    quantidade = matriz.shape[0]

    ids = np.arange(quantidade)
    colunas = [f"tweet_{i}" for i in ids]

    # Converte a matriz em DataFrame.
    quadro = pd.DataFrame(matriz, columns=colunas)
    quadro.insert(0, "tweet_id", ids)

    # Salva a matriz em CSV.
    quadro.to_csv(
        ARQUIVO_MATRIZ,
        index=False,
        float_format="%.6f",
    )

    # Salva os metadados dos temas.
    metadados = pd.DataFrame({
        "tweet_id": ids,
        "tema": temas.to_numpy(),
    })

    metadados.to_csv(
        ARQUIVO_TEMAS,
        index=False,
        encoding="utf-8",
    )

    # Configuração do Matplotlib.
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "exercicio4-matplotlib"),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Histograma considerando apenas pares distintos.
    plt.figure(figsize=(9, 5))

    plt.hist(
        similaridades_pares,
        bins=50,
        range=(0, 1),
        color="steelblue",
        edgecolor="white",
    )

    plt.title("Distribuição da similaridade de Levenshtein")
    plt.xlabel("Similaridade normalizada")
    plt.ylabel("Quantidade de pares distintos")

    plt.tight_layout()

    plt.savefig(ARQUIVO_HISTOGRAMA, dpi=150)
    plt.close()


def executar(forcar: bool = False) -> None:
    """Reutiliza resultados existentes, salvo quando forçado."""

    # Demonstração do exemplo solicitado no exercício.
    demonstrar_levenshtein()

    # Verifica se os arquivos já existem.
    if all(arquivo.exists() for arquivo in SAIDAS) and not forcar:
        print(
            "\nAs saídas já existem; recálculo ignorado. "
            "Use --force para recalcular."
        )

        for arquivo in SAIDAS:
            print(f"Arquivo existente: {arquivo}")

        return

    # Carrega os textos e os temas.
    textos, temas = carregar_textos()

    # Calcula a matriz de similaridade.
    matriz = calcular_matriz(textos)

    # Verifica as propriedades da matriz.
    simetrica = bool(np.array_equal(matriz, matriz.T))

    diagonal_unitaria = bool(
        np.all(np.diag(matriz) == 1.0)
    )

    if not simetrica or not diagonal_unitaria:
        raise RuntimeError(
            "A matriz não passou nas verificações de simetria e diagonal."
        )

    # Obtém somente os pares distintos de tweets.
    similaridades_pares = obter_similaridades_pares(matriz)

    # Salva os resultados.
    salvar_resultados(
        matriz,
        temas,
        similaridades_pares,
    )

    # Apresenta os resultados.
    print(f"\nTweets comparados: {matriz.shape[0]}")

    print(
        f"Quantidade de pares distintos: "
        f"{len(similaridades_pares)}"
    )

    print(f"Temas: {temas.value_counts().to_dict()}")

    print(
        f"Matriz simétrica: {simetrica}; "
        f"diagonal igual a 1: {diagonal_unitaria}"
    )

    print(
        f"Similaridade mínima entre pares distintos: "
        f"{similaridades_pares.min():.6f}"
    )

    print(
        f"Similaridade máxima entre pares distintos: "
        f"{similaridades_pares.max():.6f}"
    )

    print(
        f"Similaridade média entre pares distintos: "
        f"{similaridades_pares.mean():.6f}"
    )

    for arquivo in SAIDAS:
        print(f"Arquivo gerado: {arquivo}")


def main() -> int:
    """Executa o programa e trata possíveis erros."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--force",
        action="store_true",
        help="recalcula mesmo se os arquivos de saída existirem",
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
