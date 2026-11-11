"""Etapa 1: preparação dos tweets do TweetEval sobre clima."""

import html
import re
import sys
from pathlib import Path

import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
NOME_DATASET = "tweet_eval"
CONFIGURACOES = {
    "clima": "stance_climate",
    "ateismo": "stance_atheism"
}
ROTULOS = {0: "sem_posicionamento", 1: "contra", 2: "favor"}
DIVISOES = ("train", "validation", "test")



def limpar_texto(texto: str) -> str:
    
    texto = html.unescape(str(texto)).lower()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"@[\w]+", " ", texto, flags=re.UNICODE)
    return re.sub(r"\s+", " ", texto).strip()


def carregar_dataset() -> pd.DataFrame:
    partes = []

    for tema, configuracao in CONFIGURACOES.items():
        for divisao in DIVISOES:

            caminho = f"{configuracao}/{divisao}-00000-of-00001.parquet"
            uri = f"hf://datasets/cardiffnlp/tweet_eval/{caminho}"

            parte = pd.read_parquet(uri)

            parte["divisao"] = divisao
            parte["tema"] = tema

            partes.append(parte)

    return pd.concat(partes, ignore_index=True)

def preparar_dados(dados: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Preserva os dados originais e cria colunas para análise."""
    colunas_originais = list(dados.columns)
    if "text" not in dados or "label" not in dados:
        raise ValueError(
            "O dataset precisa conter as colunas 'text' e 'label'. "
            f"Colunas encontradas: {colunas_originais}"
        )

    dados["texto_limpo"] = dados["text"].map(limpar_texto)
    rotulos = pd.to_numeric(dados["label"], errors="coerce")
    desconhecidos = sorted(set(rotulos.dropna().astype(int)) - set(ROTULOS))
    if rotulos.isna().any() or desconhecidos:
        raise ValueError(f"Foram encontrados rótulos inválidos: {desconhecidos}")
    dados["categoria"] = rotulos.astype(int).map(ROTULOS)
    
    dados["grupo"] = dados["categoria"]
    return dados, colunas_originais



def apresentar_resumo(
    dados: pd.DataFrame,
    colunas_originais: list[str]
) -> None:
    

    print("Colunas existentes no dataset:", ", ".join(colunas_originais))

    print(f"\nQuantidade total de tweets: {len(dados)}")

    print("\nQuantidade de tweets por tema:")
    print(dados["tema"].value_counts().to_string())

    print("\nDistribuição geral dos rótulos:")
    print(
        dados["categoria"]
        .value_counts()
        .reindex(["sem_posicionamento", "contra", "favor"], fill_value=0)
        .to_string()
    )

    print("\nDistribuição dos rótulos por tema:")
    print(
        dados.groupby(["tema", "categoria"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["sem_posicionamento", "contra", "favor"], fill_value=0)
        .to_string()
    )

    tamanhos = dados["text"].fillna("").astype(str).str.len()

    print("\nTamanho médio das mensagens por tema:")
    print(
        tamanhos.groupby(dados["tema"])
        .mean()
        .round(2)
        .to_string()
    )

    print("\nTamanho médio por grupo:")
    print(
        tamanhos.groupby([dados["tema"], dados["grupo"]])
        .mean()
        .round(2)
        .to_string()
    )

def main() -> int:
    try:
        dados = carregar_dataset()
        dados, colunas_originais = preparar_dados(dados)
        apresentar_resumo(dados, colunas_originais)
        pasta_resultados = RAIZ_PROJETO / "resultados"
        pasta_resultados.mkdir(parents=True, exist_ok=True)
        arquivo_csv = pasta_resultados / "tweets_tratados.csv"
        dados.to_csv(arquivo_csv, index=False, encoding="utf-8")
        print(f"\nCSV exportado para: {arquivo_csv}")
        return 0
    except Exception as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
