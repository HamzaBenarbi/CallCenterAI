# src/data/prepare.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Chemins
RAW_PATH = "data/raw/all_tickets_processed_improved_v3.csv"
OUT_DIR = "data/raw"

# Créer le dossier si besoin
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Charger
print("Chargement du dataset...")
df = pd.read_csv(RAW_PATH)

# 2. Renommer les colonnes pour le projet
df = df.rename(columns={"Document": "text", "Topic_group": "label"})
df = df[["text", "label"]]

# 3. Nettoyage léger
df = df.dropna().reset_index(drop=True)
df = df[df["text"].str.strip() != ""]  # Supprime textes vides

print(f"Nombre de tickets après nettoyage : {len(df)}")
print("Répartition des classes :")
print(df["label"].value_counts())

# 4. Split stratifié 80/10/10
train_val, test = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df["label"]
)
train, val = train_test_split(
    train_val, test_size=0.1111, random_state=42, stratify=train_val["label"]
)  # 0.1111 × 0.9 ≈ 0.10

# 5. Sauvegarde
train.to_csv(f"{OUT_DIR}/train.csv", index=False)
val.to_csv(f"{OUT_DIR}/val.csv", index=False)
test.to_csv(f"{OUT_DIR}/test.csv", index=False)

print("Fichiers créés :")
print("→ data/raw/train.csv  :", len(train))
print("→ data/raw/val.csv    :", len(val))
print("→ data/raw/test.csv   :", len(test))