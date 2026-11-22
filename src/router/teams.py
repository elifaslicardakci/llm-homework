"""Intent → domain → team routing table."""
import json
from pathlib import Path

RESOURCES = Path(__file__).parent / "resources"

OOS = "oos"
HUMAN_AGENT = "Human agent"
SMALL_TALK = "small_talk"

APPLIANCE_INTENTS = {
    "appliance_fault": "Technical service",
    "warranty_claim": "Warranty and returns",
    "spare_parts_order": "Spare parts",
    "installation_request": "Installation",
}

DOMAIN_TEAMS = {
    "banking": "Billing and payments",
    "credit_cards": "Billing and payments",
    "kitchen_and_dining": "Product usage",
    "home": "Product usage",
    "auto_and_commute": "General support",
    "travel": "General support",
    "utility": "General support",
    "work": "General support",
    "meta": "Account settings",
    SMALL_TALK: "Auto reply",
}


def _intent_domains() -> dict[str, str]:
    domains = json.loads((RESOURCES / "clinc_domains.json").read_text())
    mapping = {intent: domain for domain, intents in domains.items() for intent in intents}
    mapping.update({intent: "appliance" for intent in APPLIANCE_INTENTS})
    mapping[OOS] = OOS
    return mapping


INTENT_DOMAIN = _intent_domains()


def domain_of(intent: str) -> str:
    return INTENT_DOMAIN[intent]


def team_of(intent: str) -> str:
    if intent == OOS:
        return HUMAN_AGENT
    if intent in APPLIANCE_INTENTS:
        return APPLIANCE_INTENTS[intent]
    return DOMAIN_TEAMS[INTENT_DOMAIN[intent]]
