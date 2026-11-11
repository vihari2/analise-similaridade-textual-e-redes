# Exercício 4 — Similaridade de Textos e Redes de Mensagens

Projeto desenvolvido para a disciplina de **Biologia Computacional**, com o objetivo de analisar a similaridade textual entre tweets e construir redes de mensagens a partir dessas similaridades.

## 1. Objetivo

O projeto utiliza a distância de Levenshtein para calcular a similaridade entre mensagens e construir redes nas quais:

- Cada vértice representa um tweet.
- Cada aresta representa uma conexão entre dois tweets cuja similaridade é igual ou superior ao limiar escolhido.
- As cores dos vértices representam os temas analisados.

Foram utilizados dois conjuntos de dados do TweetEval:

- `stance_climate`: tweets relacionados às mudanças climáticas.
- `stance_atheism`: tweets relacionados ao ateísmo.

## 2. Tecnologias utilizadas

- Python
- Pandas
- NumPy
- RapidFuzz
- igraph
- Matplotlib
- Hugging Face Datasets

## 3. Estrutura do projeto

```text
exercicio4/
├── src/
│   ├── main.py
│   ├── similaridade.py
│   ├── matriz_similaridade.py
│   └── rede.py
├── resultados/
│   ├── tweets_tratados.csv
│   ├── matriz_similaridade.csv
│   ├── metadados_similaridade.csv
│   ├── histograma_similaridade.png
│   ├── resultados_limiares.csv
│   ├── metricas_rede_final.csv
│   ├── rede_final.graphml
│   ├── rede_final.png
│   └── rede_maior_componente.png
├── requirements.txt
└── README.md
```

## 4. Instalação

Clone ou baixe o projeto e acesse sua pasta.

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## 5. Execução

Execute os scripts na seguinte ordem, a partir da pasta `exercicio4`:

**Etapa 1 — Importação e tratamento dos tweets**

```bash
python src/main.py
```

Realiza a importação dos conjuntos de dados, limpeza textual, classificação dos rótulos e geração do arquivo `tweets_tratados.csv`.

**Etapa 2 — Similaridade textual**

```bash
python src/similaridade.py
python src/matriz_similaridade.py
```

Calcula a distância de Levenshtein, testa o exemplo solicitado no exercício, constrói a matriz de similaridade entre os tweets e gera o histograma.

Para forçar o recálculo da matriz:

```bash
python src/matriz_similaridade.py --force
```

**Etapa 3 — Construção e análise da rede**

```bash
python src/rede.py
```

Constrói os grafos para diferentes limiares de similaridade, compara suas estruturas, seleciona o limiar final e calcula as métricas da rede.

## 6. Resultados

Foram analisados 1.297 tweets:

| Tema | Quantidade |
|---|---:|
| Ateísmo | 733 |
| Clima | 564 |
| **Total** | **1.297** |

Foram testados cinco limiares de similaridade: 0.4, 0.5, 0.6, 0.7 e 0.8.

O limiar escolhido foi 0.4, pois apresentou a maior componente conexa entre os limiares testados.

### Métricas da rede final

| Métrica | Resultado |
|---|---:|
| Ordem (vértices) | 1.297 |
| Arestas | 767 |
| Densidade | 0,000913 |
| Diâmetro | 15 |
| Transitividade | 0,554988 |

A rede apresentou 830 vértices isolados e 884 componentes conexas. Sua maior componente conexa contém 323 vértices.

## 7. Observação metodológica

As arestas representam similaridade textual entre mensagens, não necessariamente concordância de opinião.

Os rótulos `favor`, `contra` e `sem posicionamento` são relativos ao tema de cada conjunto de dados. Portanto, a comparação entre os dois temas não permite concluir que os mesmos usuários possuem determinadas opiniões sobre ambos os assuntos.

## 8. Referências

- CARDIFF NLP. TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. Disponível em: https://huggingface.co/datasets/cardiffnlp/tweet_eval.
- RAPIDFUZZ. Documentation. Disponível em: https://rapidfuzz.github.io/RapidFuzz/.
- CSARDI, Gabor; NEPUSZ, Tamas. The igraph software package for complex network research. InterJournal, Complex Systems, v. 1695, n. 5, p. 1–9, 2006. Disponível em: https://igraph.org.

