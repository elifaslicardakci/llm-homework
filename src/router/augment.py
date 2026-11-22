"""Template augmentation for the appliance intents (training split only).

The 100 hand-written appliance messages give 15 training examples per intent,
against 100 for every CLINC150 intent. Product × phrase templates add more
training variety; validation and test keep only the hand-written messages.
"""
import random

PRODUCTS = {
    "tr": ["bulaşık makinesi", "çamaşır makinesi", "buzdolabı", "fırın", "kurutma makinesi",
           "klima", "derin dondurucu", "ocak", "aspiratör", "mikrodalga"],
    "en": ["dishwasher", "washing machine", "fridge", "oven", "dryer", "air conditioner",
           "freezer", "cooktop", "range hood", "microwave"],
}
CODE_PREFIX = ["DW", "WM", "RF", "OV", "DR", "AC", "DF", "CT", "HD", "MW"]

TEMPLATES = {
    "appliance_fault": {
        "tr": ["{p} çalışmıyor", "{p} arıza yaptı", "{p} bozuldu ne yapmalıyım",
               "{p} garip bir ses çıkarıyor", "{p} su kaçırıyor", "{p} hata kodu veriyor {c}",
               "{c} model {p} açılmıyor", "{p} birden durdu", "{p} düzgün çalışmıyor teknisyen lazım"],
        "en": ["my {p} is not working", "the {p} broke down", "{p} is making a strange noise",
               "{p} shows an error code {c}", "{p} stopped working suddenly", "the {p} {c} will not start",
               "my {p} is leaking", "{p} keeps turning off", "i need a technician for my {p}"],
    },
    "warranty_claim": {
        "tr": ["{p} garantisi var mı", "aldığım {p} iade etmek istiyorum", "{p} için garanti başvurusu yapmak istiyorum",
               "{p} değişimi istiyorum", "{p} {c} garanti kapsamında mı", "{p} garantisi ne zaman bitiyor",
               "arızalı {p} için para iadesi istiyorum", "{p} garanti belgesi kayboldu"],
        "en": ["is my {p} under warranty", "i want to return my {p}", "warranty claim for my {p}",
               "can i get a replacement {p}", "is the {p} {c} covered by warranty", "when does the {p} warranty expire",
               "i want a refund for the faulty {p}", "lost the warranty card for my {p}"],
    },
    "spare_parts_order": {
        "tr": ["{p} için yedek parça", "{p} parçası sipariş etmek istiyorum", "{p} {c} yedek parça fiyatı",
               "{p} için yeni conta lazım", "{p} aksesuarı satıyor musunuz", "{p} filtresini nereden alabilirim",
               "{p} rafı kırıldı yenisini almak istiyorum", "{p} kapak menteşesi sipariş"],
        "en": ["spare parts for my {p}", "i want to order a part for the {p}", "price of a spare part for {p} {c}",
               "i need a new seal for the {p}", "do you sell accessories for the {p}", "where can i buy a {p} filter",
               "the {p} shelf broke i want a new one", "order a replacement hinge for the {p}"],
    },
    "installation_request": {
        "tr": ["{p} kurulumu için randevu", "{p} montajı ne zaman yapılır", "{p} kurulum ücreti ne kadar",
               "yeni {p} kurulumu istiyorum", "{p} {c} için teknisyen kurulum yapar mı", "{p} kurulum randevumu değiştirmek istiyorum",
               "{p} montajı ücretsiz mi", "taşındım {p} tekrar kurulmalı"],
        "en": ["book an installation for my new {p}", "when will you install my {p}", "how much is {p} installation",
               "i need my {p} installed", "can a technician install the {p} {c}", "reschedule the installation of my {p}",
               "is {p} installation free", "i moved and need the {p} installed again"],
    },
}


def augment(n_per_intent: int = 50, seed: int = 42) -> list[dict]:
    """Return n_per_intent synthetic messages per appliance intent, half Turkish, half English."""
    rng = random.Random(seed)
    rows = []
    for intent, by_lang in TEMPLATES.items():
        for lang in ("tr", "en"):
            combos = [(t, i) for t in by_lang[lang] for i in range(len(PRODUCTS[lang]))]
            rng.shuffle(combos)
            for template, i in combos[: n_per_intent // 2]:
                code = f"{CODE_PREFIX[i]}-{rng.randint(1000, 9999)}"
                text = template.format(p=PRODUCTS[lang][i], c=code)
                rows.append({"text": text, "intent": intent, "lang": lang, "source": "appliance_aug"})
    return rows
