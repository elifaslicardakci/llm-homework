from router import eliza


def test_greetings():
    assert eliza.reply("Merhaba!").startswith("Merhaba")
    assert eliza.reply("hi there").startswith("Hello")


def test_thanks_and_bye():
    assert eliza.reply("Teşekkür ederim") is not None
    assert eliza.reply("thank you so much!") is not None
    assert eliza.reply("bye") is not None


def test_reflection():
    assert eliza.reply("I feel tired of my washer") == (
        "Why do you feel tired of your washer? If it is about your appliance, tell me what happened."
    )
    assert eliza.reply("çok mutsuz hissediyorum").startswith("Neden çok mutsuz hissediyorsunuz")


def test_real_requests_are_not_small_talk():
    assert eliza.reply("merhaba bulaşık makinem su boşaltmıyor") is None
    assert eliza.reply("what is my account balance") is None


def test_fallback_language():
    assert eliza.fallback("ne yapıyorsun bugün") == eliza.DEFAULT_TR
    assert eliza.fallback("tell me a joke") == eliza.DEFAULT_EN
