from router.models.tfidf import TfidfModel
from router.pipeline import route
from router.teams import HUMAN_AGENT, team_of

TEXTS = ["my dishwasher will not drain", "dishwasher is leaking water", "bulaşık makinem çalışmıyor",
         "what is my bank balance", "check my account balance", "hesap bakiyem ne kadar",
         "tell me a joke", "say something funny", "bir fıkra anlat"] * 5
INTENTS = (["appliance_fault"] * 3 + ["balance"] * 3 + ["tell_joke"] * 3) * 5
MODEL = TfidfModel.train(TEXTS, INTENTS, C=10)


def test_greeting_uses_rules():
    d = route("Merhaba!", MODEL)
    assert d.source == "rules" and d.team == "Auto reply" and d.reply


def test_routes_to_team_and_extracts_code():
    MODEL.threshold = 0.0
    d = route("dishwasher not draining, model dw 6534", MODEL)
    assert d.intent == "appliance_fault"
    assert d.team == team_of("appliance_fault") == "Technical service"
    assert d.model_codes == ["DW-6534"]


def test_low_confidence_goes_to_human():
    MODEL.threshold = 0.99
    d = route("dishwasher not draining", MODEL)
    assert d.intent == "oos" and d.team == HUMAN_AGENT and d.source == "model (low confidence)"
    MODEL.threshold = 0.0


def test_small_talk_intent_gets_reply():
    MODEL.threshold = 0.0
    d = route("tell me something funny", MODEL)
    assert d.intent == "tell_joke" and d.team == "Auto reply" and d.reply
