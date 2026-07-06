"""Normalization and structured access for GameServer observations."""

from dataclasses import dataclass, field


LIGHT_TOKENS = {
    "bluelight": "treasure",
    "redlight": "powerup",
    "weaklight": "unknown",
    "greenlight": "poison",
}

KNOWN_FLAGS = {
    "blocked",
    "steps",
    "breeze",
    "flash",
    "damage",
    "hit",
    "bluelight",
    "redlight",
    "greenlight",
    "weaklight",
}


def _clean(token):
    return token.strip().replace(" ", "").lower()


def normalize_token(token):
    """Return a canonical lowercase token or None for empty tokens."""
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
    """Structured view of the server's observation tokens."""

    tokens: tuple = field(default_factory=tuple)

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tuple(normalize_tokens(tokens)))

    @property
    def as_list(self):
        return list(self.tokens)

    def has(self, token):
        return normalize_token(token) in self.tokens

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
        for token, kind in LIGHT_TOKENS.items():
            if token in self.tokens:
                return kind
        return None

    @property
    def has_item(self):
        return self.light in ("treasure", "powerup", "unknown")

    @property
    def async_events(self):
        return [t for t in self.tokens if t in ("hit", "damage")]

    @property
    def environment_tokens(self):
        return [t for t in self.tokens if t not in ("hit", "damage")]

    @property
    def steps_only(self):
        return self.steps and not self.damage and not self.hit and not self.enemy


def parse_observation_payload(payload):
    if payload is None or payload == "":
        return Observation()
    return Observation.from_tokens(str(payload).split(","))
