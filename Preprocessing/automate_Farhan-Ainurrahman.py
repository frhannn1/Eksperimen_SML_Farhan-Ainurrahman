import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==============================
# CONFIG (SESuaikan dengan datasetmu)
# ==============================
RAW_DATA_PATH = "../bank_transactions_data_edited.csv"
OUTPUT_FOLDER = "../Preprocessing"

NUM_COLS = ['TransactionAmount','TransactionDuration', 'LoginAttempts','AccountBalance']          
CAT_COLS = ['TransactionType', 'Channel', 'Location_Grouped']

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(RAW_DATA_PATH)
df = df.drop(columns=['TransactionDate','PreviousTransactionDate','CustomerAge','CustomerOccupation','TransactionID', 'AccountID','DeviceID','MerchantID','IP Address'])
print("[INFO] Data Loaded:", df.shape)

# ==============================
# REMOVE DUPLICATES
# ==============================
df = df.drop_duplicates()
print("[INFO] Duplicate removed:", df.shape)

# ==============================
# HANDLE MISSING VALUES
# ==============================
df = df.dropna()

print("[INFO] Missing values handled")

# ==============================
# ENCODE CATEGORICAL FEATURES
# ==============================
label = LabelEncoder()
for col in CAT_COLS:
    df[col] = label.fit_transform(df[col])
print("[INFO] Encoding categorical OK")

# ==============================
# FEATURE SCALING
# ==============================
if NUM_COLS:
    scaler = StandardScaler()
    df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
    print("[INFO] Feature scaling OK")


# ==============================
# OUTPUT SAVE
# ==============================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, "datasetbank_automate.csv")
df.to_csv(output_path, index=False)

print("\n[SUCCESS] Preprocessing selesai!")
print("[SAVED] File di:", output_path)
