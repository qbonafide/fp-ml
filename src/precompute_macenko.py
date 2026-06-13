import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Import the normalizer from the existing dataset script
from dataset import MacenkoNormalize

def process_csv(csv_path, output_csv_path, normalizer):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    new_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_path = row["image_path"]
        
        # Replace the base folder with the new macenko folder
        new_path_str = orig_path.replace("data/raw/", "data/macenko/").replace("data\\raw\\", "data/macenko/")
        
        # Fallback if 'data/raw/' wasn't in the exact string for some reason
        if "data/macenko/" not in new_path_str:
            new_path_str = os.path.join("data", "macenko", os.path.basename(orig_path))
            
        new_paths.append(new_path_str)
        
        if os.path.exists(new_path_str):
            continue
            
        try:
            # 1. Open the raw original image
            img = Image.open(orig_path).convert("RGB")
            
            # 2. Apply Macenko normalization (Runs CPU Heavy Code Here)
            norm_img = normalizer(img)
            
            # 3. Save it perfectly processed
            os.makedirs(os.path.dirname(new_path_str), exist_ok=True)
            norm_img.save(new_path_str)
            
        except Exception as e:
            print(f"Failed to process {orig_path}: {e}")
            
    # Update DataFrame
    df["image_path"] = new_paths
    
    # Save the new DataFrames
    df.to_csv(output_csv_path, index=False)
    print(f"Finished! Saved dataset map to {output_csv_path}\n")

def main():
    csv_files = [
        ("data/train_all.csv", "data/train_macenko.csv"),
        ("data/val_all.csv", "data/val_macenko.csv"),
        ("data/test_all.csv", "data/test_macenko.csv")
    ]
    
    print("Initializing Macenko Normalizer...")
    normalizer = MacenkoNormalize()
    
    if not normalizer.is_fit:
        print("Warning: Normalizer failed to fit. Check if torchstain is parsing correctly.")
        
    for in_csv, out_csv in csv_files:
        if os.path.exists(in_csv):
            process_csv(in_csv, out_csv, normalizer)
        else:
            print(f"Skipping {in_csv}: File not found.")

if __name__ == "__main__":
    main()
