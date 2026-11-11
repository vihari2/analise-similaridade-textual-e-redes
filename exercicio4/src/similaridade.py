"""Demonstração do cálculo de Levenshtein pedido no enunciado."""

from rapidfuzz.distance import Levenshtein


def demonstrar_levenshtein() -> tuple[int, float]:
    
    texto1 = "Elisabete"
    texto2 = "Elizabete"
    distancia = Levenshtein.distance(texto1, texto2)
    similaridade = Levenshtein.normalized_similarity(texto1, texto2)

    
    distancia_esperada = 1
    similaridade_esperada = 8 / 9
    confere = (
        distancia == distancia_esperada
        and abs(similaridade - similaridade_esperada) < 1e-12
    )

    print("Comparação de exemplo — Levenshtein")
    print(f"Texto 1: {texto1}")
    print(f"Texto 2: {texto2}")
    print(f"Distância: {distancia} (esperado: {distancia_esperada})")
    print(
        f"Similaridade normalizada: {similaridade:.6f} "
        f"(esperado: {similaridade_esperada:.6f})"
    )
    print(f"Resultado confere com o exemplo: {'sim' if confere else 'não'}")
    return distancia, similaridade
