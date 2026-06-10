"""
Controlador de Logica Fuzzy para decisoes de combate (aula 25 da disciplina).

Entrada:  energia do drone (0-100) e distancia do inimigo (passos).
Saida:    agressividade em [0, 1] (inferencia Sugeno de ordem zero:
          media ponderada dos consequentes pelas ativacoes das regras).

A FSM usa a agressividade para escolher entre ATTACK / HUNT / FLEE em vez
de limiares rigidos: um drone com 55 de energia e inimigo longe se comporta
diferente de um com 55 de energia e inimigo colado.
"""


def tri(x, a, b, c):
    """Funcao de pertinencia triangular."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trap_left(x, a, b):
    """Pertinencia 1 ate 'a', decaindo linearmente ate 0 em 'b'."""
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def trap_right(x, a, b):
    """Pertinencia 0 ate 'a', crescendo linearmente ate 1 em 'b'."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def combat_aggressiveness(energy, enemy_dist):
    """Agressividade [0,1] dada a energia e a distancia do inimigo."""
    # conjuntos fuzzy da energia
    e_low = trap_left(energy, 20, 45)
    e_med = tri(energy, 30, 55, 80)
    e_high = trap_right(energy, 60, 85)
    # conjuntos fuzzy da distancia (sensor 'enemy' alcanca ate 10 passos)
    d_close = trap_left(enemy_dist, 2, 6)
    d_far = trap_right(enemy_dist, 4, 9)

    # regras: (ativacao, consequente)
    rules = [
        (min(e_high, d_close), 1.0),   # forte e inimigo perto -> ataque total
        (min(e_high, d_far), 0.8),     # forte e longe -> ataca
        (min(e_med, d_close), 0.65),   # mediano e perto -> ataca com cautela
        (min(e_med, d_far), 0.5),      # mediano e longe -> neutro
        (min(e_low, d_close), 0.1),    # fraco e perto -> fugir!
        (min(e_low, d_far), 0.3),      # fraco e longe -> evita confronto
    ]
    num = sum(w * v for w, v in rules)
    den = sum(w for w, _ in rules)
    return num / den if den > 0 else 0.5
