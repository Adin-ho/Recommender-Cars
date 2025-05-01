import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("app/data_mobil.csv")

def preprocess():
    fitur = df[["Usia Kendaraan (tahun)", "Harga", "Kapasitas Mesin (cc)", "Transmisi", "Bahan Bakar"]].copy()
    
    # One-hot encode fitur kategori
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(fitur[["Transmisi", "Bahan Bakar"]]).toarray()
    
    numerik = fitur[["Usia Kendaraan (tahun)", "Harga", "Kapasitas Mesin (cc)"]].values
    all_features = pd.DataFrame(
        data = pd.concat([
            pd.DataFrame(numerik, columns=["usia", "harga", "mesin"]),
            pd.DataFrame(encoded)
        ], axis=1),
        index=df.index
    )
    return all_features, encoder

def recommend(prefs):
    df = pd.read_csv("app/data_mobil.csv")

    # Hitung usia mobil
    df["Usia"] = 2025 - df["Tahun"]

    # Hitung skor kemiripan
    df["score"] = (
        -abs(df["Usia"] - prefs["usia"]) * 2
        -abs(df["Harga"] - prefs["harga"]) / 1_000_000
        -abs(df["Kapasitas Mesin"] - prefs["mesin"])
    )

    # FILTER TRANSMISI DAN BBM SESUAI
    df = df[df["Transmisi"].str.lower() == prefs["transmisi"].lower()]
    df = df[df["Bahan Bakar"].str.lower() == prefs["bbm"].lower()]

    # Urutkan berdasarkan skor
    return df.sort_values("score", ascending=False).head(10)

def recommend(preferences: dict, top_n=3):
    X, encoder = preprocess()
    
    # Buat vektor preferensi user
    user_input = pd.DataFrame([[
        preferences.get("usia", 5),
        preferences.get("harga", 150_000_000),
        preferences.get("mesin", 1300),
        preferences.get("transmisi", "Otomatis"),
        preferences.get("bbm", "Bensin")
    ]], columns=["Usia Kendaraan (tahun)", "Harga", "Kapasitas Mesin (cc)", "Transmisi", "Bahan Bakar"])

    encoded_user = encoder.transform(user_input[["Transmisi", "Bahan Bakar"]]).toarray()
    numerik_user = user_input[["Usia Kendaraan (tahun)", "Harga", "Kapasitas Mesin (cc)"]].values
    user_vector = pd.concat([
        pd.DataFrame(numerik_user, columns=["usia", "harga", "mesin"]),
        pd.DataFrame(encoded_user)
    ], axis=1)

    # Hitung kemiripan cosine
    similarities = cosine_similarity(X, user_vector)[::, 0]
    df["similarity"] = similarities
    return df.sort_values(by="similarity", ascending=False).head(top_n)
