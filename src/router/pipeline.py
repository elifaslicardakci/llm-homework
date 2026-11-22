"""Message → routing decision: small-talk rules first, then the intent model."""
import time
from dataclasses import dataclass, field

from router import eliza
from router.models.base import IntentModel
from router.preprocessing.normalize import extract_model_codes, normalize
from router.teams import HUMAN_AGENT, OOS, SMALL_TALK, domain_of, team_of


@dataclass
class Decision:
    team: str
    intent: str
    confidence: float | None
    reply: str | None
    model_codes: list[str]
    normalized: str
    latency_ms: float
    source: str  # "rules", "model" or "model (low confidence)"
    notes: list[str] = field(default_factory=list)


def route(message: str, model: IntentModel) -> Decision:
    start = time.perf_counter()
    codes = extract_model_codes(message)
    normalized = normalize(message)
    rule_reply = eliza.reply(message)
    if rule_reply is not None:
        return Decision("Auto reply", "small_talk (rule)", None, rule_reply, codes, normalized,
                        (time.perf_counter() - start) * 1000, "rules")

    probs = model.predict_proba([message])[0]
    best = int(probs.argmax())
    intent, confidence = model.labels[best], float(probs[best])
    source = "model"
    if confidence < model.threshold and intent != OOS:
        intent, source = OOS, "model (low confidence)"
    reply = None
    if intent == OOS:
        team = HUMAN_AGENT
    else:
        team = team_of(intent)
        if domain_of(intent) == SMALL_TALK:
            reply = eliza.fallback(message)
    return Decision(team, intent, confidence, reply, codes, normalized,
                    (time.perf_counter() - start) * 1000, source)
