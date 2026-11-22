from router.preprocessing.normalize import (
    correct_token, extract_model_codes, normalize, turkish_lower,
)


def test_turkish_lower():
    assert turkish_lower("IŞIK İzmir") == "ışık izmir"


def test_extract_model_codes():
    assert extract_model_codes("bulaşık makinem DW-6534 ve wm 8120") == ["DW-6534", "WM-8120"]
    assert extract_model_codes("AC1290 çalışmıyor") == ["AC-1290"]
    assert extract_model_codes("3 gündür bekliyorum") == []


def test_normalize_placeholders():
    out = normalize("Bakın: https://x.com/a, mail a@b.com, tel 0532 123 45 67, model DW-6534!!")
    assert out == "bakın url mail email tel phone model modelcode"


def test_normalize_repeats_and_punctuation():
    assert normalize("çoook acil!!!", correct=False) == "çook acil"


def test_correct_ascii_folded():
    assert correct_token("bulasik", set()) == "bulaşık"
    assert correct_token("camasir", set()) == "çamaşır"


def test_correct_misspelling():
    assert correct_token("buzdolabi", set()) == "buzdolabı"
    assert correct_token("dishwaser", set()) == "dishwasher"
    assert correct_token("makinasi", set()) == "makinesi"


def test_correct_leaves_known_words():
    assert correct_token("open", set()) == "open"            # too short
    assert correct_token("watching", set()) == "watching"    # distance 2 to washing
    assert correct_token("bridge", {"bridge"}) == "bridge"   # in vocabulary
    assert correct_token("2023", set()) == "2023"


def test_default_vocabulary_protects_training_words():
    assert correct_token("weather") == "weather"
    assert correct_token("fishing") == "fishing"


def test_normalize_corrects_products():
    assert normalize("Bulasik makinasi su bosaltmiyor") == "bulaşık makinesi su bosaltmiyor"
