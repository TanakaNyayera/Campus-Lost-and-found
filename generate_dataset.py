import os
import csv
import random
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = r"C:\Users\Matthew .K. Maunga\OneDrive\Desktop\MSU project\MSU_Lost_Found_ML_System"
LOST_PATH  = os.path.join(BASE, "data", "raw", "images", "lost")
FOUND_PATH = os.path.join(BASE, "data", "raw", "images", "found")
OUTPUT_CSV = os.path.join(BASE, "data", "raw", "descriptions.csv")

# ── Color detection ───────────────────────────────────────────────────────────
COLOR_MAP = {
    "red":    ([150,  50,  50], [255, 100, 100]),
    "orange": ([180, 100,  50], [255, 165,  80]),
    "yellow": ([180, 180,  50], [255, 255, 100]),
    "green":  ([ 30, 100,  30], [100, 200, 100]),
    "blue":   ([ 30,  50, 100], [100, 150, 255]),
    "purple": ([100,  30, 100], [180,  80, 180]),
    "pink":   ([200, 100, 150], [255, 180, 210]),
    "brown":  ([ 80,  40,  20], [160, 100,  60]),
    "grey":   ([100, 100, 100], [180, 180, 180]),
    "white":  ([200, 200, 200], [255, 255, 255]),
    "black":  ([  0,   0,   0], [ 60,  60,  60]),
    "navy":   ([  0,   0,  80], [ 30,  30, 130]),
}

def get_dominant_color(image_path, k=3):
    """Detect the dominant color name from an image using KMeans."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "grey"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        pixels = img.reshape(-1, 3).astype(np.float32)

        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        kmeans.fit(pixels)

        counts = Counter(kmeans.labels_)
        dominant_rgb = kmeans.cluster_centers_[counts.most_common(1)[0][0]]

        best_color    = "grey"
        best_distance = float("inf")
        for color_name, (low, high) in COLOR_MAP.items():
            center   = [(low[i] + high[i]) / 2 for i in range(3)]
            distance = sum((dominant_rgb[i] - center[i]) ** 2 for i in range(3)) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_color    = color_name
        return best_color
    except Exception:
        return "grey"

# ── Description templates ─────────────────────────────────────────────────────
TEMPLATES = {
    "backpack": [
        "A {colour} backpack with {straps} straps and a {size} main compartment, suitable for carrying books.",
        "A {size} {colour} backpack with multiple pockets and a {brand} logo on the front.",
        "A worn {colour} backpack with a {zipper} zipper pocket and {straps} shoulder straps.",
        "A {brand} backpack in {colour} with a padded laptop sleeve inside.",
        "A {size} {colour} school backpack with side mesh pockets for water bottles.",
    ],
    "comb": [
        "A {colour} plastic comb with fine teeth and a long handle.",
        "A small {colour} comb with a handle, appears to be a {brand} brand.",
        "A {size} {colour} comb used for hair grooming, slightly bent on one side.",
        "A {colour} wide-tooth comb with a protective carry case.",
        "A pocket-sized {colour} comb with densely packed teeth.",
    ],
    "cup": [
        "A {colour} ceramic cup with a {brand} logo printed on the side.",
        "A {size} {colour} plastic cup with a fitted lid and straw slot.",
        "A {colour} stainless steel travel cup with a screw-on lid and handle.",
        "A {colour} mug with a hairline crack on the handle and {brand} branding.",
        "A {size} {colour} reusable cup found near the university cafeteria.",
    ],
    "earphones": [
        "A pair of {colour} wired earphones with a {brand} logo on the earbuds.",
        "A pair of {colour} in-ear earphones with a 3.5mm jack connector.",
        "A set of {colour} earphones with foam ear tips and slightly tangled wire.",
        "A {brand} pair of {colour} earphones with an inline microphone and volume control.",
        "A pair of {colour} Bluetooth earphones with a magnetic charging case.",
    ],
    "laptop": [
        "A {colour} {brand} laptop with a {size} inch widescreen display.",
        "A {brand} laptop in {colour} with several stickers on the lid.",
        "A {size} inch {colour} laptop with a small crack on the bottom corner.",
        "A {brand} {colour} laptop with a charger and mouse pad attached.",
        "A thin {colour} laptop with a {brand} logo and USB-C charging port.",
    ],
    "phone": [
        "A {colour} {brand} smartphone with a cracked front screen.",
        "A {brand} phone enclosed in a {colour} silicone protective case.",
        "A {size} inch {colour} phone with a fingerprint scanner on the rear.",
        "A {brand} {colour} phone with a damaged charging port and scratched back.",
        "A {colour} smartphone with a {brand} logo and a tempered glass screen protector.",
    ],
    "rugby_ball": [
        "A {colour} rugby ball with black lace stitching and visible scuff marks.",
        "A {brand} {colour} rugby ball with the MSU team logo printed on the side.",
        "A {size} {colour} match rugby ball with a {brand} brand label.",
        "A {colour} training rugby ball with worn grip texture on the panels.",
        "A {colour} rugby ball with a slightly deflated feel and mud stains.",
    ],
    "rugby_boots": [
        "A pair of {colour} rugby boots with metal screw-in studs.",
        "A pair of {brand} {colour} rugby boots size {shoe_size} with ankle padding.",
        "A {colour} pair of rugby boots with one missing stud and mud on the soles.",
        "A pair of {colour} {brand} rugby boots with reinforced toe cap.",
        "A size {shoe_size} pair of {colour} rugby boots with high ankle support.",
    ],
    "sneakers": [
        "A pair of {colour} {brand} sneakers size {shoe_size} with white rubber soles.",
        "A worn pair of {colour} sneakers with a {brand} logo stitched on the side.",
        "A {brand} pair of {colour} running sneakers with mesh upper and cushioned insole.",
        "A pair of {colour} canvas sneakers with lace-up front and flat sole.",
        "A size {shoe_size} pair of {colour} {brand} sneakers with a velcro strap and reflective strip.",
    ],
    "soccer_ball": [
        "A {colour} and white soccer ball with black hexagonal panel patches.",
        "A {brand} {colour} soccer ball slightly deflated with scuff marks.",
        "A {size} {colour} soccer ball with a {brand} label on one panel.",
        "A {colour} match soccer ball with grass stains and worn surface.",
        "A {colour} training soccer ball with a small puncture on the surface.",
    ],
    "suitcase": [
        "A {colour} hard-shell suitcase with four spinner wheels and a combination lock.",
        "A {size} {colour} suitcase with a broken retractable handle.",
        "A {colour} fabric suitcase with a {brand} logo tag and two front zip pockets.",
        "A {size} {colour} rolling suitcase with a TSA-approved combination lock.",
        "A {colour} travel suitcase with a luggage tag and name label attached.",
    ],
    "wallet": [
        "A {colour} leather bifold wallet with multiple card slots and a note compartment.",
        "A {brand} {colour} wallet with a coin zip pocket and embossed logo.",
        "A slim {colour} wallet with a {brand} logo and visible wear on the edges.",
        "A {colour} wallet found containing a student ID card and some coins.",
        "A {size} {colour} zip-around wallet with wrist strap attachment.",
    ],
    "water_bottle": [
        "A {colour} plastic water bottle with a flip-top sports lid.",
        "A {brand} {colour} stainless steel insulated water bottle 500ml capacity.",
        "A {size} {colour} water bottle with a carry handle and built-in straw.",
        "A {colour} gym water bottle with a {brand} logo and minor dents on the side.",
        "A {colour} transparent water bottle with a screw-on cap and measurement markings.",
    ],
    "wrist_watch": [
        "A {colour} wrist watch with a leather strap and classic round face.",
        "A {brand} {colour} digital watch with a rubber sports strap.",
        "A {colour} analog watch with a stainless steel bracelet and {brand} logo on the face.",
        "A {colour} watch with a large face and a cracked crystal glass.",
        "A {brand} {colour} sports watch with a chronograph and date display.",
    ],
}

# ── Fill-in values ────────────────────────────────────────────────────────────
SIZES      = ["small", "medium", "large"]
BRANDS     = ["Nike", "Adidas", "Samsung", "Apple", "HP", "Lenovo", "Puma", "Reebok", "Huawei", "Generic"]
ZIPPERS    = ["front", "side", "top"]
STRAPS     = ["padded", "adjustable", "thin", "wide"]
SHOE_SIZES = ["6", "7", "8", "9", "10", "11", "12"]

def fill(template, colour):
    return template.format(
        colour    = colour,
        size      = random.choice(SIZES),
        brand     = random.choice(BRANDS),
        zipper    = random.choice(ZIPPERS),
        straps    = random.choice(STRAPS),
        shoe_size = random.choice(SHOE_SIZES),
    )

# ── Build records ─────────────────────────────────────────────────────────────
records = []
item_id = 1
skipped = 0

print("Scanning images and detecting dominant colors...\n")

for status, folder_path in [("lost", LOST_PATH), ("found", FOUND_PATH)]:
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for fname in image_files:
        parts = fname.replace('.jpg','').replace('.jpeg','').replace('.png','').split('_')
        if len(parts) < 3:
            print(f"  Skipping (bad filename format): {fname}")
            skipped += 1
            continue

        category = '_'.join(parts[1:-1])
        if category not in TEMPLATES:
            print(f"  Skipping unknown category: {category} ({fname})")
            skipped += 1
            continue

        full_image_path = os.path.join(folder_path, fname)
        colour          = get_dominant_color(full_image_path)
        template        = random.choice(TEMPLATES[category])
        description     = fill(template, colour)
        image_path_rel  = f"data/raw/images/{status}/{fname}"

        print(f"  [{status}] {fname} → detected color: {colour}")

        records.append({
            "item_id":       item_id,
            "item_name":     category.replace('_', ' ').title(),
            "description":   description,
            "category":      category,
            "status":        status,
            "date_reported": f"2025-{random.randint(1,9):02d}-{random.randint(1,28):02d}",
            "image_path":    image_path_rel,
            "label":         0
        })
        item_id += 1

# ── Write CSV ─────────────────────────────────────────────────────────────────
fieldnames = ["item_id","item_name","description","category","status",
              "date_reported","image_path","label"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"✅ Dataset created successfully!")
print(f"   Total records : {len(records)}")
print(f"   Skipped       : {skipped}")
print(f"   Saved to      : {OUTPUT_CSV}")
print(f"\n--- Records by status ---")
for s, c in Counter(r["status"] for r in records).items():
    print(f"  {s}: {c}")
print(f"\n--- Records by category ---")
for cat, count in sorted(Counter(r["category"] for r in records).items()):
    print(f"  {cat}: {count}")
print(f"\n--- Sample descriptions ---")
for r in random.sample(records, min(5, len(records))):
    print(f"  [{r['status']}] {r['item_name']}: {r['description']}")
