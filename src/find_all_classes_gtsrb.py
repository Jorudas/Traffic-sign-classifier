import pandas as pd
import shutil
import os

# ✅ Teisingi keliai (duomenys vienu lygiu aukščiau)
csv_path = "../data/GTSRB_Final_Test_GT/GT-final_test.csv"
images_dir = "../data/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/"
output_dir = "examples_gtsrb"

# ✅ Sukuriame aplanką, jei jo nėra
os.makedirs(output_dir, exist_ok=True)

# ✅ Nuskaitome anotacijas
df = pd.read_csv(csv_path, sep=';')

# ✅ Randame visas klases (0–42)
unique_classes = sorted(df['ClassId'].unique())
print(f"🔍 Rastos {len(unique_classes)} unikalios klasės.")

# ✅ Kopijuojame po vieną pavyzdį iš kiekvienos klasės
missing = []
copied = 0

for class_id in unique_classes:
    row = df[df['ClassId'] == class_id].iloc[0]
    filename = row['Filename']
    src = os.path.join(images_dir, filename)
    dst = os.path.join(output_dir, f"class_{class_id}.ppm")

    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        missing.append(filename)

print(f"\n✅ Po vieną pavyzdį iš kiekvienos klasės nukopijuota: {copied} failų.")
print(f"📁 Aplankas: {output_dir}/")

if missing:
    print("\n⚠️ Šių failų nerasta ir jie praleisti:")
    for name in missing:
        print(" -", name)