import os

folders = {
    "lost": r"C:\Users\Matthew .K. Maunga\OneDrive\Desktop\MSU project\MSU_Lost_Found_ML_System\data\raw\images\lost",
    "found": r"C:\Users\Matthew .K. Maunga\OneDrive\Desktop\MSU project\MSU_Lost_Found_ML_System\data\raw\images\found"
}

for status, path in folders.items():
    print(f"\n--- {status.upper()} FOLDER ---")
    categories = {}
    for fname in os.listdir(path):
        if fname.endswith(('.jpg', '.jpeg', '.png')):
            parts = fname.split('_')
            if len(parts) >= 2:
                # Extract category (everything between status and number)
                category = '_'.join(parts[1:-1])
                categories[category] = categories.get(category, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} images")