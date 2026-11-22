"""ELIZA-style rule-based replies for greetings and small talk (Turkish and English).

Each rule is a regex over the lowercased message and a reply template; captured
groups are inserted after pronoun reflection, as in Weizenbaum's ELIZA.
"""
import re

from router.preprocessing.normalize import turkish_lower

REFLECTIONS = {
    "i": "you", "me": "you", "my": "your", "am": "are", "mine": "yours",
    "you": "I", "your": "my", "yours": "mine",
}

# (pattern, reply) — the first matching rule wins. Patterns must cover the whole message.
RULES: list[tuple[re.Pattern, str]] = [(re.compile(p), r) for p, r in [
    (r"(merhaba|selam|selamlar|iyi günler|günaydın|iyi akşamlar)\W*", "Merhaba! Size nasıl yardımcı olabilirim? Arızayı ya da isteğinizi yazabilirsiniz."),
    (r"(hello|hi|hey|good (morning|afternoon|evening))( there)?\W*", "Hello! How can I help you today? Describe the problem or your request."),
    (r"(teşekkürler|teşekkür ederim|sağ ol|sağol|eyvallah)\W*", "Rica ederim! Başka bir konuda yardımcı olabilir miyim?"),
    (r"(thanks|thank you|thx|cheers)( a lot| so much)?\W*", "You're welcome! Is there anything else I can help with?"),
    (r"(görüşürüz|hoşça kal|iyi günler dilerim|bay bay)\W*", "Görüşmek üzere, iyi günler!"),
    (r"(bye|goodbye|see you|have a nice day)\W*", "Goodbye, have a nice day!"),
    (r"(nasılsın|naber|nasılsınız)\W*", "Teşekkürler, iyiyim. Sizin için ne yapabilirim?"),
    (r"how are you( doing)?\W*", "I'm fine, thanks for asking. What can I do for you?"),
    (r"(sen )?(bot musun|robot musun|kimsin)\W*", "Ben otomatik destek asistanıyım. Mesajınızı doğru ekibe yönlendiriyorum."),
    (r"(are you a (bot|robot|human)|who are you)\W*", "I'm an automated support assistant. I route your message to the right team."),
    (r"(.*) hissediyorum\W*", "Neden {0} hissediyorsunuz? Cihazınızla ilgili bir sorun varsa anlatabilirsiniz."),
    (r"i feel (.*?)\W*", "Why do you feel {0}? If it is about your appliance, tell me what happened."),
    (r"i am (.*?)\W*", "How long have you been {0}? Tell me how I can help."),
]]

DEFAULT_TR = "Sohbet için teşekkürler! Cihazınızla ilgili bir sorun varsa yazabilirsiniz."
DEFAULT_EN = "Nice chatting! If you have a question about your appliance, just ask."


def reflect(fragment: str) -> str:
    return " ".join(REFLECTIONS.get(w, w) for w in fragment.split())


def reply(message: str) -> str | None:
    """Return a rule-based reply if the whole message is small talk, else None."""
    # Turkish lowercasing maps English "I" to "ı", so try the plain lowercase form too.
    candidates = [turkish_lower(message).strip(), message.lower().strip()]
    for pattern, template in RULES:
        for text in candidates:
            m = pattern.fullmatch(text)
            if m:
                groups = [reflect(g) for g in m.groups() if g] if "{0}" in template else []
                return template.format(*groups)
    return None


def fallback(message: str) -> str:
    """Generic small-talk reply when the classifier says small talk but no rule matched."""
    return DEFAULT_TR if re.search(r"[çğıöşü]", turkish_lower(message)) else DEFAULT_EN
