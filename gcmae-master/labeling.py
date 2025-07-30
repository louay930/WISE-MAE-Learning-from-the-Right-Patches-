import pandas as pd
from pathlib import Path

# === Input paths ===
finetune_csv = Path("/user/louay.hamdi/u13592/project_split/TCGA-NSCLC-finetune.csv")
luad_csv = Path("/user/louay.hamdi/u13592/project_split/TCGA-LUAD.csv")
lusc_csv = Path("/user/louay.hamdi/u13592/project_split/TCGA-LUSC.csv")
output_csv = Path("/user/louay.hamdi/u13592/project_split/TCGA-NSCLC-finetune-labeled.csv")

# === Load CSVs ===
df_finetune = pd.read_csv(finetune_csv)
df_luad = pd.read_csv(luad_csv)
df_lusc = pd.read_csv(lusc_csv)

# === Extract base slide IDs ===
df_finetune["slide_id_clean"] = df_finetune["slide_id"].apply(lambda x: x.split(".")[0])
luad_ids = set(df_luad["slide_id"].apply(lambda x: x.split(".")[0]))
lusc_ids = set(df_lusc["slide_id"].apply(lambda x: x.split(".")[0]))

# === Assign labels ===
def get_label(slide_id):
    if slide_id in luad_ids:
        return "LUAD"
    elif slide_id in lusc_ids:
        return "LUSC"
    else:
        return "UNKNOWN"

df_finetune["label"] = df_finetune["slide_id_clean"].apply(get_label)

# === Drop unused columns and keep only slide_id and label ===
df_out = df_finetune[["slide_id", "label"]]
df_out = df_out[df_out["label"] != "UNKNOWN"]  # drop unmatched slides if any

# === Save to new CSV ===
df_out.to_csv(output_csv, index=False)
print(f"✅ Saved labeled finetune set to: {output_csv}")
print(f"🧪 Label distribution:\n{df_out['label'].value_counts()}")
