# app/format.py
from typing import Any, Dict

def _pick(d: Dict[str, Any], cands, default=None):
    for k in cands:
        if k in d and str(d[k]).strip() != "":
            return d[k]
    return default

def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def rupiah(x):
    n = _to_float(x)
    if n is None:
        s = str(x) if x is not None else "-"
        return s if s else "-"
    return f"Rp {n:,.0f}".replace(",", ".")

def normalize_row(raw: Any) -> Dict[str, Any]:
    d = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)

    nama  = _pick(d, ["nama_mobil","Nama Mobil","nama","Nama","judul","title","tipe","tipe_mobil","model"])
    merek = _pick(d, ["merek","Merk","brand","Brand"])
    tahun = _pick(d, ["tahun","Tahun","year","Year"])
    transm= _pick(d, ["transmisi","Transmisi","transmission","transmision","Transmision"])
    bb    = _pick(d, ["bahan_bakar","Bahan Bakar","bbm","BBM","fuel"])
    cc    = _pick(d, ["kapasitas_mesin","Kapasitas Mesin","engine_cc","cc","mesin"])

    harga_str  = _pick(d, ["harga","Harga"])
    harga_num  = _pick(d, ["harga_angka","Harga Angka","harga_num","harga_number"])
    usia       = _pick(d, ["usia","Usia","umur","Umur","age"])

    # harga string → jika kosong, format dari angka
    harga_fmt = harga_str if (harga_str and str(harga_str).strip()) else rupiah(harga_num)
    usia_int  = _to_int(usia)

    return {
        "nama_mobil": nama,
        "merek": merek,
        "tahun": _to_int(tahun) or tahun,
        "transmisi": transm,
        "bahan_bakar": bb,
        "kapasitas_mesin": cc,
        "harga": harga_fmt,
        "harga_angka": _to_float(harga_num),
        "usia": usia_int if usia_int is not None else usia,
    }
