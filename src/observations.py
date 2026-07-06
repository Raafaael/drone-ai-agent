"""Traduz a lista crua de tokens que o servidor manda em algo legível.

O servidor varia a grafia das observações (`blueLight`, `eneny#3` em vez de
`enemy#3`, etc.) e cada `ai_agent` original reimplementava essa limpeza à
sua maneira. Aqui isso acontece uma vez só, e o resto do código lê a
`Observation` resultante por nome (`observation.damage`, `observation.light`)
em vez de vasculhar strings.
"""

from dataclasses import dataclass, field


LIGHT_TOKENS = {
    "bluelight": "treasure",
    "redlight": "powerup",
    "weaklight": "unknown",
    "greenlight": "poison",
}


def _clean(token):
    return token.strip().replace(" ", "").lower()


def normalize_token(token):
    """Converte um token cru para a forma canônica, ou None se vazio.

    Cobre as variações conhecidas do protocolo: `h`/`d` como atalho para
    hit/damage, `eneny#N` como grafia alternativa de `enemy#N`, e distância
    de inimigo malformada caindo num valor padrão em vez de quebrar."""
    low = _clean(str(token))
    if not low:
        return None
    if low == "h":
        return "hit"
    if low == "d":
        return "damage"
    if low.startswith("eneny"):
        low = "enemy" + low[5:]
    if low.startswith("enemy"):
        if "#" in low:
            head, value = low.split("#", 1)
            if head == "enemy":
                try:
                    return f"enemy#{max(0, int(value))}"
                except ValueError:
                    return "enemy#5"
        return "enemy#5"
    return low


def normalize_tokens(tokens):
    normalized = []
    for token in tokens or []:
        if token is None:
            continue
        parts = str(token).split(",")
        for part in parts:
            item = normalize_token(part)
            if item is not None:
                normalized.append(item)
    return normalized


@dataclass(frozen=True)
class Observation:
    """Visão estruturada e imutável de uma leitura de sensores."""

    tokens: tuple = field(default_factory=tuple)

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tuple(normalize_tokens(tokens)))

    @property
    def as_list(self):
        return list(self.tokens)

    @property
    def blocked(self):
        return "blocked" in self.tokens

    @property
    def steps(self):
        return "steps" in self.tokens

    @property
    def breeze(self):
        return "breeze" in self.tokens

    @property
    def flash(self):
        return "flash" in self.tokens

    @property
    def damage(self):
        return "damage" in self.tokens

    @property
    def hit(self):
        return "hit" in self.tokens

    @property
    def enemy_distance(self):
        """Distância do inimigo avistado, ou None se nenhum estiver visível."""
        for token in self.tokens:
            if token.startswith("enemy#"):
                try:
                    return int(token.split("#", 1)[1])
                except ValueError:
                    return 5
        return None

    @property
    def enemy(self):
        return self.enemy_distance is not None

    @property
    def light(self):
        """"treasure"/"powerup"/"unknown"/"poison", ou None sem luz nenhuma."""
        for token, kind in LIGHT_TOKENS.items():
            if token in self.tokens:
                return kind
        return None

    @property
    def has_item(self):
        return self.light in ("treasure", "powerup", "unknown")


def parse_observation_payload(payload):
    """Constrói uma Observation a partir do payload cru de uma mensagem
    'o' do servidor (string separada por vírgulas)."""
    if payload is None or payload == "":
        return Observation()
    return Observation.from_tokens(str(payload).split(","))
