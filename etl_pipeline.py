"""ETL-pipeline för friskvårdsdata.

Steg:
1) Läs rådata från CSV.
2) Rengör och standardisera kolumner.
3) Spara till SQLite (gemensam tabell) med dataset-flagga.
"""

import sqlite3
import numpy as np
import pandas as pd

# Mapping för standardisering
medlemstyp_map = {
    "Grund": "Bas",
    "Basic": "Bas",
    "Studerande": "Student",
    "Gold": "Premium",
    "Plus": "Premium",
}

anlaggning_map = {
    "Linköping C": "Linköping",
    "Lund C": "Lund",
    "Sthlm City": "Stockholm City",
    "City": "Stockholm City",
    "Sthlm Södermalm": "Stockholm Södermalm",
    "Södermalm": "Stockholm Södermalm",
    "Uppsala C": "Uppsala",
    "Göteborg C": "Göteborg Centrum",
    "Gbg Centrum": "Göteborg Centrum",
    "Gbg Hisingen": "Göteborg Hisingen",
    "Hisingen": "Göteborg Hisingen",
    "Sthlm Kungsholmen": "Stockholm Kungsholmen",
    "Örebro C": "Örebro",
    "Malmö Vh": "Malmö Västra Hamnen",
    "Västra Hamnen": "Malmö Västra Hamnen",
    "Malmö C": "Malmö Centrum",
    "Malmö City": "Malmö Centrum",
    "Västerås C": "Västerås",
    "Kungsholmen": "Stockholm Kungsholmen",
}

status_map = {
    "Deltog": "Genomförd",
    "Klar": "Genomförd",
    "Struken": "Avbokad",
    "Cancelled": "Avbokad",
    "Ej Närvarande": "No-Show",
    "Missad": "No-Show",
    "No Show": "No-Show",
}

passnamn_map = {
    "H.I.I.T": "Hiit",
    "Högintensiv": "Hiit",
    "Intervall": "Hiit",
    "Core": "Styrketräning",
    "Styrka": "Styrketräning",
    "Styrkepass": "Styrketräning",
    "Gympass": "Styrketräning",
    "Strength": "Styrketräning",
    "Cykel": "Spinning",
    "Spin": "Spinning",
    "Indoor Cycling": "Spinning",
    "Zumba": "Dans",
    "Dance": "Dans",
    "Vinyasa": "Yoga",
    "Hatha Yoga": "Yoga",
    "Boxing": "Boxning",
    "Fightpass": "Boxning",
}

# Svenska månadsnamn till engelska för att pandas ska kunna tolka dem
sv_months = {
    "januari": "january",
    "februari": "february",
    "mars": "march",
    "april": "april",
    "maj": "may",
    "juni": "june",
    "juli": "july",
    "augusti": "august",
    "september": "september",
    "oktober": "october",
    "november": "november",
    "december": "december",
}

# =========================
# Numeriska konverteringar
# =========================
def clean_månadskostnad(df: pd.DataFrame) -> pd.DataFrame:
    """Skapar flagga för negativa belopp och absolutbelopp."""
    df["är_negativt_belopp"] = df["månadskostnad"] < 0
    df["månadskostnad_abs"] = df["månadskostnad"].abs()
    return df


def clean_födelseår(df: pd.DataFrame) -> pd.DataFrame:
    """Konverterar födelseår till numeriskt med stöd för saknade värden."""
    df["födelseår"] = pd.to_numeric(df["födelseår"], errors="coerce").astype("Int64")
    return df

# Textkolumner
# ========= MEDLEMSTYP =========
def clean_medlemstyp(df: pd.DataFrame) -> pd.DataFrame:
    """Standardiserar medlemstyp till title case och mappar varianter."""
    series = df["medlemstyp"].astype("string").str.title().replace(medlemstyp_map)
    df["medlemstyp"] = series
    return df

# ========= ANLÄGGNING =========
def clean_anläggning(df: pd.DataFrame) -> pd.DataFrame:
    """Standardiserar anläggningsnamn och mappar förkortningar."""
    series = df["anläggning"].astype("string").str.title().replace(anlaggning_map)
    df["anläggning"] = series
    return df

# ========= STATUS =========
def clean_status(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliserar statusvärden (t.ex. Avbokad, Genomförd, No-Show)."""
    series = df["status"].astype("string").str.title().replace(status_map)
    df["status"] = series
    return df

# ========= PASSNAMN =========
def clean_passnamn(df: pd.DataFrame) -> pd.DataFrame:
    """Standardiserar passnamn (t.ex. HIIT, Yoga, Spinning)."""
    series = df["passnamn"].astype("string").str.title().replace(passnamn_map)
    df["passnamn"] = series
    return df

# Null-hantering
def clean_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fyller saknade värden för textfält med rimliga default-värden."""
    text_columns = ["anläggning", "instruktör"]
    for col in text_columns:
        df[col] = df[col].fillna("Okänd")

    df["feedback_text"] = df["feedback_text"].fillna("")
    return df

# Datum och tid
def clean_date(df: pd.DataFrame) -> pd.DataFrame:
    """Rensar och konverterar datumkolumner till datetime.

    Hanterar:
    - Tomma strängar
    - Svenska månadsnamn
    - Blandade format
    """
    date_columns = [
        "medlem_startdatum",
        "medlem_slutdatum",
        "bokningsdatum",
        "passdatum",
        "feedbackdatum",
    ]
    # Loopa över datumkolumner och ta bort mellanslag
    for col in date_columns:
        series = df[col].astype("string").str.strip()
        # Tomma värden blir saknade <NA>, för att inte misstolkas som datum
        series = series.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        # Ta bort kommatecken och gör all text till små bokstäver
        series = series.str.replace(",", "", regex=False).str.lower()
        # Ersätt svenska månadsnamn till engelska
        for sv, en in sv_months.items():
            series = series.str.replace(sv, en, regex=False)
        # Tolka blandade format och språk, konvertera till datetime
        df[col] = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    return df

# Konvertera till dt.time
def clean_passtid(df: pd.DataFrame) -> pd.DataFrame:
    """Konverterar passtid (HH:MM) till python time-objekt."""
    df["passtid"] = pd.to_datetime(df["passtid"], format="%H:%M", errors="coerce").dt.time
    return df

# Kategori konventeringar
def convert_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """Konverterar utvalda kolumner till kategorier för lägre minnesanvändning."""
    categorical_columns = ["medlemstyp", "anläggning", "status", "passnamn", "instruktör"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# Master-funktion
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Kör hela rengöringspipen i rätt ordning."""
    df_clean = df.copy()
    # Ta bort dubletter först
    df_clean = df_clean.drop_duplicates()
    # Numeriska konverteringar
    df_clean = clean_månadskostnad(df_clean)
    df_clean = clean_födelseår(df_clean)
    # Textkolumner
    df_clean = clean_medlemstyp(df_clean)
    df_clean = clean_anläggning(df_clean)
    df_clean = clean_status(df_clean)
    df_clean = clean_passnamn(df_clean)
    # Null-hantering
    df_clean = clean_null_values(df_clean)
    # Datum och tid
    df_clean = clean_date(df_clean)
    df_clean = clean_passtid(df_clean)
    # Kategori konventeringar
    df_clean = convert_to_category(df_clean)

    return df_clean

# === Läs rådata och kör pipeline ===
df_main_raw = pd.read_csv("friskvard_data.csv")
df_val_raw = pd.read_csv("friskvard_validation.csv")

df_clean = transform_data(df_main_raw)
df_val = transform_data(df_val_raw)

# Sätt dataset-flagga
df_clean["dataset"] = "main"
df_val["dataset"] = "validation"

# Spara i SQLite-databas
def load_to_sqlite(
    df: pd.DataFrame,
    db_path: str = "friskvard_data_cleaned.db",
    table: str = "friskvard_data",
    if_exists: str = "replace",
) -> None:
    """Sparar DataFrame till SQLite."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists=if_exists, index=False)
    conn.close()


def load_dataset_to_db(
    df: pd.DataFrame,
    db_path: str = "friskvard_data_cleaned.db",
    table: str = "friskvard_data",
    method: str = "append",
) -> pd.DataFrame:
    """Sparar dataset till SQLite-databasen."""
    df = df.copy()
    load_to_sqlite(df, db_path=db_path, table=table, if_exists=method)
    return df

# Spara till SQLite (gemensam tabell)
load_dataset_to_db(df=df_clean, method="replace")
load_dataset_to_db(df=df_val, method="append")