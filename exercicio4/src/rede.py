"""Etapa 3: construção e análise da rede de similaridade textual."""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "exercicio4-matplotlib")
)

import igraph as ig
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
PASTA_RESULTADOS = RAIZ / "resultados"
ARQ_MATRIZ = PASTA_RESULTADOS / "matriz_similaridade.csv"
ARQ_METADADOS = PASTA_RESULTADOS / "metadados_similaridade.csv"
ARQ_DADOS = PASTA_RESULTADOS / "tweets_tratados.csv"
LIMIARES = (0.4, 0.5, 0.6, 0.7, 0.8)
TAMANHO_BLOCO = 64


def carregar_vertices() -> pd.DataFrame:
   
    metadados = pd.read_csv(ARQ_METADADOS)
    origem = pd.read_csv(ARQ_DADOS, usecols=["text", "tema", "categoria"])
    if len(metadados) != len(origem):
        raise ValueError("A matriz/metadados e o CSV tratado têm quantidades diferentes de tweets.")
    if metadados["tweet_id"].duplicated().any():
        raise ValueError("Há tweet_id duplicado nos metadados.")
    if not np.array_equal(metadados["tema"].to_numpy(), origem["tema"].to_numpy()):
        raise ValueError("A ordem dos temas nos metadados não corresponde ao CSV tratado.")
    vertices = metadados.copy()
    vertices["categoria"] = origem["categoria"].to_numpy()
    vertices["text"] = origem["text"].fillna("").astype(str).to_numpy()
    return vertices


def dimensao_matriz() -> int:
    
    with ARQ_MATRIZ.open("r", newline="", encoding="utf-8") as arquivo:
        cabecalho = next(csv.reader(arquivo))
    if not cabecalho or cabecalho[0] != "tweet_id":
        raise ValueError("Formato inesperado na matriz: primeira coluna deve ser tweet_id.")
    return len(cabecalho) - 1


def coletar_arestas(limiar: float, quantidade: int) -> list[tuple[int, int]]:
    
    arestas: list[tuple[int, int]] = []
    
    for inicio, bloco in zip(
        range(0, quantidade, TAMANHO_BLOCO),
        pd.read_csv(ARQ_MATRIZ, usecols=range(1, quantidade + 1),
                    dtype=np.float32, chunksize=TAMANHO_BLOCO),
    ):
        valores = bloco.to_numpy(copy=False)
        for deslocamento, linha in enumerate(valores):
            origem = inicio + deslocamento
            primeiro_destino = origem + 1
            if primeiro_destino >= quantidade:
                continue
            destinos = np.flatnonzero(linha[primeiro_destino:] >= limiar)
            arestas.extend((origem, primeiro_destino + int(destino)) for destino in destinos)
    return arestas


def construir_grafo(vertices: pd.DataFrame, arestas: list[tuple[int, int]]) -> ig.Graph:
   
    grafo = ig.Graph(n=len(vertices), edges=arestas, directed=False)
    for coluna in ("tweet_id", "tema", "categoria", "text"):
        grafo.vs[coluna] = vertices[coluna].tolist()
    return grafo


def medir_limiar(grafo: ig.Graph, limiar: float) -> dict[str, int | float]:
    componentes = grafo.connected_components()
    tamanhos = componentes.sizes()
    return {
        "limiar": limiar,
        "arestas": grafo.ecount(),
        "vertices_isolados": sum(grau == 0 for grau in grafo.degree()),
        "componentes_conectados": len(tamanhos),
        "tamanho_maior_componente": max(tamanhos, default=0),
    }


def escolher_limiar(resultados: pd.DataFrame) -> float:
   
    n = int(resultados["vertices"].iloc[0])
    elegiveis = resultados[
        (resultados["tamanho_maior_componente"] >= 0.80 * n)
        & (resultados["vertices_isolados"] <= 0.10 * n)
    ]
    if not elegiveis.empty:
        return float(elegiveis["limiar"].max())
    maior_lcc = resultados["tamanho_maior_componente"].max()
    candidatos = resultados[resultados["tamanho_maior_componente"] == maior_lcc]
    menor_numero_arestas = candidatos["arestas"].min()
    candidatos = candidatos[candidatos["arestas"] == menor_numero_arestas]
    return float(candidatos["limiar"].max())


def desenhar_rede(grafo: ig.Graph, destino: Path, titulo: str) -> None:
    
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "exercicio4-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cores = {"clima": "#3182bd", "ateismo": "#e6550d"}
   
    coordenadas = np.asarray(grafo.layout("fr").coords)
    pares = np.asarray(grafo.get_edgelist(), dtype=np.int32)
    print("Quantidade de arestas:", len(pares))
    print("Formato dos pares:", pares.shape)
    print("Coordenadas válidas:", np.isfinite(coordenadas).all())
    print("Primeiras 5 arestas:", pares[:5])
    figura, eixo = plt.subplots(figsize=(16, 12))
    if pares.size:
        segmentos = coordenadas[pares]
        eixo.add_collection(
            LineCollection(
                segmentos,
                colors="#3f3f3f",
                linewidths=1.0,
                alpha=0.85,
                zorder=1,
            )
        )
    for tema, cor in cores.items():
        indices = [i for i, valor in enumerate(grafo.vs["tema"]) if valor == tema]
        if indices:
            pontos = coordenadas[indices]
            eixo.scatter(pontos[:, 0], pontos[:, 1], s=18, c=cor, label=tema, zorder=2)
    eixo.autoscale()
    eixo.set_aspect("equal", adjustable="datalim")
    eixo.set_title(titulo, fontsize=16, pad=18)
    eixo.axis("off")
    eixo.legend(title="Tema", loc="upper right", frameon=True)
    figura.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


def executar() -> None:
    for caminho in (ARQ_MATRIZ, ARQ_METADADOS, ARQ_DADOS):
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo necessário não encontrado: {caminho}")
    vertices = carregar_vertices()
    n_matriz = dimensao_matriz()
    if len(vertices) != n_matriz:
        raise ValueError(f"Matriz de dimensão {n_matriz}, mas existem {len(vertices)} tweets.")

    resumo = []
    for limiar in LIMIARES:
        arestas = coletar_arestas(limiar, n_matriz)
        grafo = construir_grafo(vertices, arestas)
        linha = medir_limiar(grafo, limiar)
        linha["vertices"] = grafo.vcount()
        resumo.append(linha)
        print(
            f"limiar={limiar:.1f}: arestas={linha['arestas']}, "
            f"isolados={linha['vertices_isolados']}, "
            f"componentes={linha['componentes_conectados']}, "
            f"LCC={linha['tamanho_maior_componente']}"
        )
        del arestas, grafo

    tabela = pd.DataFrame(resumo)[
        ["limiar", "vertices", "arestas", "vertices_isolados",
         "componentes_conectados", "tamanho_maior_componente"]
    ]
    arq_resumo = PASTA_RESULTADOS / "resultados_limiares.csv"
    tabela.to_csv(arq_resumo, index=False)

    limiar_final = escolher_limiar(tabela)
    arestas_finais = coletar_arestas(limiar_final, n_matriz)
    rede = construir_grafo(vertices, arestas_finais)
    metricas = {
        "limiar": limiar_final,
        "ordem": rede.vcount(),
        "arestas": rede.ecount(),
        "densidade": rede.density(loops=False),
     
        "diametro": rede.diameter(directed=False, unconn=True),
        "transitividade": rede.transitivity_undirected(mode="zero"),
    }
    tabela_metricas = pd.DataFrame([metricas])
    tabela_metricas.to_csv(PASTA_RESULTADOS / "metricas_rede_final.csv", index=False)

    arquivo_graphml = PASTA_RESULTADOS / "rede_final.graphml"
    rede.write_graphml(str(arquivo_graphml))
    arquivo_imagem = PASTA_RESULTADOS / "rede_final.png"
    desenhar_rede(rede, arquivo_imagem, "Rede de similaridade — todos os tweets")

    
    maior_componente = rede.connected_components().giant()
    print(
        "Maior componente conexa:"
        f" {maior_componente.vcount()} vértices e {maior_componente.ecount()} arestas"
    )
    arquivo_maior_componente = PASTA_RESULTADOS / "rede_maior_componente.png"
    desenhar_rede(
        maior_componente,
        arquivo_maior_componente,
        "Rede de similaridade — maior componente conexa",
    )

    print(f"\nLimiar escolhido: {limiar_final:.1f}")
    print("Critério: maior limiar com LCC >= 80% dos vértices e isolados <= 10%;")
    print("se nenhum atende, maior LCC e menor quantidade de arestas em empate.")
    print(f"Métricas finais: {metricas}")
    print("Diâmetro: maior caminho mínimo finito dentro de qualquer componente.")
    print(f"CSV dos limiares: {arq_resumo}")
    print(f"CSV das métricas finais: {PASTA_RESULTADOS / 'metricas_rede_final.csv'}")
    print(f"GraphML: {arquivo_graphml}")
    print(f"Visualização: {arquivo_imagem}")
    print(f"Visualização da maior componente: {arquivo_maior_componente}")


def main() -> int:
    try:
        executar()
        return 0
    except Exception as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
