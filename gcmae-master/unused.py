import pandas as pd
from pathlib import Path

# === INPUT PATHS (adjust if needed) ===
csv_path = Path("/user/louay.hamdi/u13592/project_split/TCGA-NSCLC.csv")
image_dir = Path("/user/louay.hamdi/u13592/.project/dir.project/NSCLC_pretraining_dataset/images")
output_csv = Path("/user/louay.hamdi/u13592/project_split/TCGA-NSCLC-finetune.csv")

# === 1. Read all NSCLC slides from CSV ===
df_all = pd.read_csv(csv_path)
df_all["base_slide_id"] = df_all["slide_id"].apply(lambda x: x.split(".svs")[0])
all_slide_ids = set(df_all["base_slide_id"])

print(f"📄 Total slides in CSV: {len(all_slide_ids)}")
print(f"🔍 Sample CSV slide IDs: {list(all_slide_ids)[:5]}")

# === 2. Extract used slide IDs from image filenames ===
image_files = list(image_dir.glob("*.png"))
used_slide_ids = set([p.name.split("_patch_")[0] for p in image_files])

print(f"🧠 Total used slide IDs found in images: {len(used_slide_ids)}")
print(f"🔍 Sample used slide IDs: {list(used_slide_ids)[:5]}")

# === 3. Determine unused slides ===
unused_slide_ids = all_slide_ids - used_slide_ids
df_unused = df_all[df_all["base_slide_id"].isin(unused_slide_ids)]

# === 4. Save results ===
df_unused.drop(columns=["base_slide_id"]).to_csv(output_csv, index=False)

print(f"✅ Found {len(unused_slide_ids)} unused slides.")
print(f"📁 Saved finetuning CSV to: {output_csv}")
