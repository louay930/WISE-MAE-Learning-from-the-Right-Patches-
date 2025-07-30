import os
from pathlib import Path
from PIL import Image, PngImagePlugin
from multiprocessing import Pool
from tqdm import tqdm

# Root directory containing all your images (e.g., train/ and val/)
ROOT_DIR = "/user/louay.hamdi/u13592/.project/dir.project/NSCLC_pretraining_split"

def clean_png(img_path):
    try:
        img = Image.open(img_path)
        data = list(img.getdata())
        img_clean = Image.new(img.mode, img.size)
        img_clean.putdata(data)
        
        # Remove all chunks (e.g., iCCP) by re-saving without metadata
        img_clean.save(img_path, format="PNG", optimize=True)
        return True
    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")
        return False

def main():
    print(f"📂 Scanning PNG files in {ROOT_DIR} ...")
    all_pngs = list(Path(ROOT_DIR).rglob("*.png"))
    print(f"🔍 Found {len(all_pngs)} .png files")

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(clean_png, all_pngs), total=len(all_pngs)))

    success_count = sum(results)
    print(f"\n✅ Cleaned {success_count} / {len(all_pngs)} images successfully")

if __name__ == "__main__":
    main()
