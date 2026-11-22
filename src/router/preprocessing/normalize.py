"""Regex normalization, model-code extraction and product-name correction."""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import Levenshtein

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"(?:\+90|0)?\s*5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
# Product model codes: 2–3 letters, optional dash or space, 3–5 digits (DW-6534, wm 8120, AC1290)
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{2,3})[-\s]?(\d{3,5})\b")
REPEAT_RE = re.compile(r"(\w)\1{2,}")
TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")

TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
ASCII_FOLD = str.maketrans("çğıöşü", "cgiosu")

# Canonical product words (Turkish and English); misspellings are corrected to these.
PRODUCT_TERMS = [
    "bulaşık", "çamaşır", "makine", "makinesi", "makinem", "makinesini", "buzdolabı",
    "buzdolabım", "fırın", "fırını", "ocak", "kurutma", "klima", "dondurucu", "aspiratör",
    "süpürge", "kombi", "televizyon", "mikrodalga",
    "dishwasher", "washing", "machine", "washer", "dryer", "fridge", "refrigerator",
    "freezer", "oven", "cooktop", "microwave", "conditioner", "vacuum", "boiler",
]
_FOLDED_TERMS = {t.translate(ASCII_FOLD): t for t in PRODUCT_TERMS}

# Tokens seen in the training messages; written by scripts/prepare_data.py.
KNOWN_WORDS_FILE = Path(__file__).resolve().parents[1] / "resources" / "known_words.txt"


@lru_cache(maxsize=1)
def known_words() -> frozenset[str]:
    if not KNOWN_WORDS_FILE.exists():
        return frozenset()
    return frozenset(KNOWN_WORDS_FILE.read_text(encoding="utf-8").split())


def turkish_lower(text: str) -> str:
    return text.translate(TR_LOWER).lower()


def ascii_fold(text: str) -> str:
    return text.translate(ASCII_FOLD)


def extract_model_codes(text: str) -> list[str]:
    """Return product model codes in canonical form, e.g. 'dw 6534' → 'DW-6534'."""
    codes = []
    for letters, digits in MODEL_CODE_RE.findall(text):
        code = f"{letters.upper()}-{digits}"
        if code not in codes:
            codes.append(code)
    return codes


def correct_token(token: str, vocab: frozenset[str] | set[str] | None = None) -> str:
    """Map a misspelled product word to its canonical form; leave known words alone.

    Known words (training vocabulary by default) are never changed. Otherwise: exact
    match after ASCII folding (bulasik → bulaşık), then Levenshtein distance ≤ 1
    (≤ 2 for 10+ letters) to a product term with the same first letter.
    """
    vocab = known_words() if vocab is None else vocab
    if token in PRODUCT_TERMS or token in vocab or len(token) < 5 or not token.isalpha():
        return token
    folded = ascii_fold(token)
    if folded in _FOLDED_TERMS:
        return _FOLDED_TERMS[folded]
    max_dist = 2 if len(token) >= 10 else 1
    best, best_dist = token, max_dist + 1
    for f_term, term in _FOLDED_TERMS.items():
        if f_term[0] != folded[0] or abs(len(f_term) - len(folded)) > max_dist:
            continue
        dist = Levenshtein.distance(folded, f_term, score_cutoff=max_dist)
        if dist < best_dist:
            best, best_dist = term, dist
    return best


def normalize(text: str, *, vocab: frozenset[str] | set[str] | None = None,
              correct: bool = True, mask_codes: bool = True) -> str:
    """Lowercase (Turkish-aware), replace URLs / e-mails / phones / model codes with
    placeholder tokens, collapse repeated letters, drop punctuation, fix product names."""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" url ", text)
    text = EMAIL_RE.sub(" email ", text)
    text = PHONE_RE.sub(" phone ", text)
    if mask_codes:
        text = MODEL_CODE_RE.sub(" modelcode ", text)
    text = turkish_lower(text)
    text = REPEAT_RE.sub(r"\1\1", text)
    tokens = TOKEN_RE.findall(text)
    if correct:
        tokens = [correct_token(t, vocab) for t in tokens]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return text.split()
