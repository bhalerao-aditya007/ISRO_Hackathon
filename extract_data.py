import zipfile
import glob
import os

ohrc_dir = r"D:\PRISM_DATA\02_OHRC"
zip_files = glob.glob(os.path.join(ohrc_dir, "*.zip"))

for z in zip_files:
    print(f"Extracting {z}...")
    with zipfile.ZipFile(z, 'r') as zip_ref:
        zip_ref.extractall(ohrc_dir)
print("Done extracting OHRC data.")
