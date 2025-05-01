import re

def extract_preferences(text):
    # Ekstraksi preferensi user dari teks
    usia = re.search(r"usia.*?(\d+)", text.lower())
    harga = re.search(r"(\d{2,3})\s*juta", text.lower())
    mesin = re.search(r"(\d{3,4})\s*cc", text.lower())

    transmisi = "Otomatis" if "otomatis" in text.lower() else "Manual" if "manual" in text.lower() else "Otomatis"
    bbm = "Bensin" if "bensin" in text.lower() else "Diesel" if "diesel" in text.lower() else "Bensin"

    return {
        "usia": int(usia.group(1)) if usia else 5,
        "harga": int(harga.group(1)) * 1_000_000 if harga else 150_000_000,
        "mesin": int(mesin.group(1)) if mesin else 1300,
        "transmisi": transmisi,
        "bbm": bbm
    }
