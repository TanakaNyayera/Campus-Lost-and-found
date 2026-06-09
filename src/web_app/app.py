import os
import pickle
import secrets
import csv
import numpy as np
import cv2
import yaml
import pymysql
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify)
from flask_mail import Mail, Message
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# ── Load configuration ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')

if not os.path.exists(CONFIG_PATH):
    # For deployments (GitHub/Render/etc.), config/config.yaml may be provided via secrets/volume.
    # Fall back to config.example.yaml if the real config isn't present.
    example_path = os.path.join(BASE_DIR, 'config', 'config.example.yaml')
    if os.path.exists(example_path):
        CONFIG_PATH_TO_LOAD = example_path
    else:
        raise FileNotFoundError(
            "Missing config/config.yaml and config/config.example.yaml. "
            "Create config/config.yaml (or add config.example.yaml to the repo) "
            "and fill in your local settings. "
            f"Checked example path: {example_path}"
        )
else:
    CONFIG_PATH_TO_LOAD = CONFIG_PATH

with open(CONFIG_PATH_TO_LOAD, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)



def _set_config_from_env(section, key, env_name, cast=str):
    value = os.getenv(env_name)
    if value is not None:
        config[section][key] = cast(value)


_set_config_from_env('database', 'host', 'MSU_DB_HOST')
_set_config_from_env('database', 'port', 'MSU_DB_PORT', int)
_set_config_from_env('database', 'name', 'MSU_DB_NAME')
_set_config_from_env('database', 'user', 'MSU_DB_USER')
_set_config_from_env('database', 'password', 'MSU_DB_PASSWORD')
_set_config_from_env('flask', 'secret_key', 'MSU_FLASK_SECRET_KEY')
_set_config_from_env('email', 'server', 'MSU_MAIL_SERVER')
_set_config_from_env('email', 'port', 'MSU_MAIL_PORT', int)
_set_config_from_env('email', 'username', 'MSU_MAIL_USERNAME')
_set_config_from_env('email', 'password', 'MSU_MAIL_PASSWORD')
_set_config_from_env('email', 'sender', 'MSU_MAIL_SENDER')

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

app.secret_key = config['flask']['secret_key']
app.config['MAX_CONTENT_LENGTH'] = config['flask']['max_content_length']

UPLOAD_FOLDER      = os.path.join(BASE_DIR, config['flask']['upload_folder'])
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
RAW_DESCRIPTIONS_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'descriptions.csv')
RAW_DESCRIPTIONS_FIELDS = [
    'item_id',
    'item_name',
    'description',
    'category',
    'status',
    'date_reported',
    'image_path',
    'label',
]

ITEM_CATEGORIES = {
    "Bags": ["Backpack", "Suitcase"],
    "Electronics": ["Phone", "Laptop", "Earphones"],
    "Accessories": ["Wallet", "Wrist Watch"],
    "Sports": ["Rugby Ball", "Rugby Boots", "Soccer Ball"],
    "Personal Items": ["Comb", "Cup", "Water Bottle"],
    "Footwear": ["Sneakers"],
    "Other": ["Other"],
}

LEGACY_CATEGORY_BY_ITEM_TYPE = {
    item_type: category
    for category, item_types in ITEM_CATEGORIES.items()
    for item_type in item_types
}

LEGACY_ITEM_SLUGS_BY_CATEGORY = {
    category: [
        item_type.lower().replace(" ", "_")
        for item_type in item_types
    ]
    for category, item_types in ITEM_CATEGORIES.items()
}

# ── Dynamic item types (CSV “learning” expansion) ──────────────────────────
# Extends dropdown suggestions and match candidate filtering from stored CSV
# reports. The chosen UI category is still the grouping key; the typed “Other”
# value is stored as item_name and learned under that same category.
DYNAMIC_ITEM_CATEGORIES: dict[str, list[str]] = {}
DYNAMIC_ITEM_SLUGS_BY_CATEGORY: dict[str, list[str]] = {}


def _slugify_item_name(name: str) -> str:
    return (name or '').strip().lower().replace(' ', '_')


def _load_learned_item_types_from_csv(max_unique_per_category: int = 2000):
    """Load learned item types from RAW_DESCRIPTIONS_CSV.

    Returns:
      (learned_item_names_by_category, learned_item_slugs_by_category)

    learned_item_names_by_category keeps original free-text values (for UI).
    learned_item_slugs_by_category keeps slugs (for match candidate filtering).
    """
    learned_item_names_by_category: dict[str, set[str]] = {}
    learned_item_slugs_by_category: dict[str, set[str]] = {}

    try:
        if not os.path.exists(RAW_DESCRIPTIONS_CSV) or os.path.getsize(RAW_DESCRIPTIONS_CSV) == 0:
            return learned_item_names_by_category, learned_item_slugs_by_category

        with open(RAW_DESCRIPTIONS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return learned_item_names_by_category, learned_item_slugs_by_category

            for row in reader:
                category = (row.get('category') or '').strip()
                item_name = (row.get('item_name') or '').strip()
                if not category or not item_name:
                    continue

                # Only learn within known UI categories.
                if category not in ITEM_CATEGORIES:
                    continue

                learned_item_names_by_category.setdefault(category, set()).add(item_name)
                learned_item_slugs_by_category.setdefault(category, set()).add(_slugify_item_name(item_name))

        # Cap per category to avoid unbounded dropdown growth.
        for category in list(learned_item_names_by_category.keys()):
            names = learned_item_names_by_category.get(category, set())
            if len(names) > max_unique_per_category:
                keep_names = set(sorted(names)[:max_unique_per_category])
                learned_item_names_by_category[category] = keep_names
                learned_item_slugs_by_category[category] = {_slugify_item_name(n) for n in keep_names}

    except Exception as e:
        print(f"[Dynamic item learning] Failed to load CSV: {e}")

    return learned_item_names_by_category, learned_item_slugs_by_category


def _build_dynamic_item_mappings():
    global DYNAMIC_ITEM_CATEGORIES, DYNAMIC_ITEM_SLUGS_BY_CATEGORY

    learned_names_by_category, learned_slugs_by_category = _load_learned_item_types_from_csv()

    # Start from hardcoded categories/items.
    dynamic_categories: dict[str, list[str]] = {cat: list(types) for cat, types in ITEM_CATEGORIES.items()}
    dynamic_slugs: dict[str, list[str]] = {cat: list(slugs) for cat, slugs in LEGACY_ITEM_SLUGS_BY_CATEGORY.items()}

    for category, learned_names in learned_names_by_category.items():
        if category not in ITEM_CATEGORIES:
            continue

        existing_names = set(dynamic_categories.get(category, []))
        merged_names = list(dynamic_categories.get(category, []))
        for n in sorted(learned_names):
            if n in existing_names:
                continue
            merged_names.append(n)
        dynamic_categories[category] = merged_names

        existing_slugs = set(dynamic_slugs.get(category, []))
        merged_slugs = list(dynamic_slugs.get(category, []))
        for s in sorted(learned_slugs_by_category.get(category, set())):
            if s in existing_slugs:
                continue
            merged_slugs.append(s)
        dynamic_slugs[category] = merged_slugs

    DYNAMIC_ITEM_CATEGORIES = dynamic_categories
    DYNAMIC_ITEM_SLUGS_BY_CATEGORY = dynamic_slugs


_build_dynamic_item_mappings()

# ── Mail setup ────────────────────────────────────────────────────────────────
app.config['MAIL_SERVER']         = config['email']['server']
app.config['MAIL_PORT']           = config['email']['port']
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = config['email']['username']
app.config['MAIL_PASSWORD']       = config['email']['password']
app.config['MAIL_DEFAULT_SENDER'] = config['email']['sender']
mail = Mail(app)

# ── Database connection ───────────────────────────────────────────────────────
def get_db():
    return pymysql.connect(
        host       = config['database']['host'],
        port       = config['database']['port'],
        user       = config['database']['user'],
        password   = config['database']['password'],
        database   = config['database']['name'],
        cursorclass= pymysql.cursors.DictCursor
    )


def ensure_settings_tables(db):
    with db.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key VARCHAR(100) NOT NULL PRIMARY KEY,
                setting_value TEXT,
                updated_by INT NULL,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INT NOT NULL PRIMARY KEY,
                email_notifications TINYINT(1) NOT NULL DEFAULT 1,
                match_alerts TINYINT(1) NOT NULL DEFAULT 1,
                report_status_updates TINYINT(1) NOT NULL DEFAULT 1,
                sound_notifications TINYINT(1) NOT NULL DEFAULT 0,
                two_factor_enabled TINYINT(1) NOT NULL DEFAULT 0,
                dark_mode TINYINT(1) NOT NULL DEFAULT 0,
                theme_color VARCHAR(20) NOT NULL DEFAULT '#003366',
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_admin_state (
                user_id INT NOT NULL PRIMARY KEY,
                suspended TINYINT(1) NOT NULL DEFAULT 0,
                updated_by INT NULL,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                user_id INT NULL,
                email VARCHAR(255) NOT NULL,
                role VARCHAR(50) NULL,
                action VARCHAR(100) NOT NULL,
                ip_address VARCHAR(64) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_activity_logs_created_at (created_at),
                INDEX idx_activity_logs_user_id (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    db.commit()


def ensure_match_status_column(db):
    with db.cursor() as cursor:
        cursor.execute("SHOW COLUMNS FROM matches LIKE 'status'")
        if not cursor.fetchone():
            cursor.execute("""
                ALTER TABLE matches
                ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending'
            """)
            db.commit()


def get_app_settings(db):
    defaults = {
        'support_phone': '+263 000 000 000',
        'default_language': 'English',
        'time_zone': 'Africa/Johannesburg',
        'admin_alerts': '1',
        'archive_resolved_reports': '0',
        'auto_delete_old_reports': '0',
        'max_image_upload_size_mb': '16',
        'allowed_file_types': 'jpg,jpeg,png',
        'enable_anonymous_reporting': '0',
        'session_timeout_minutes': '30',
        'login_attempt_limit': '5',
    }
    with db.cursor() as cursor:
        cursor.execute("SELECT setting_key, setting_value FROM app_settings")
        stored = {row['setting_key']: row['setting_value'] for row in cursor.fetchall()}
    defaults.update(stored)
    return defaults


def get_user_preferences(db, user_id):
    with db.cursor() as cursor:
        cursor.execute("""
            INSERT IGNORE INTO user_preferences (user_id) VALUES (%s)
        """, (user_id,))
        db.commit()
        cursor.execute("SELECT * FROM user_preferences WHERE user_id=%s", (user_id,))
        return cursor.fetchone()


def log_activity(db, user_id, email, role, action):
    with db.cursor() as cursor:
        cursor.execute("""
            INSERT INTO activity_logs (user_id, email, role, action, ip_address)
            VALUES (%s,%s,%s,%s,%s)
        """, (user_id, email, role, action, request.remote_addr))
    db.commit()


def is_user_suspended(db, user_id):
    with db.cursor() as cursor:
        cursor.execute("SELECT suspended FROM user_admin_state WHERE user_id=%s", (user_id,))
        row = cursor.fetchone()
    return bool(row and row.get('suspended'))


def _build_external_url(path):
    base = config['flask'].get('base_url', 'http://localhost:5000').rstrip('/')
    try:
        return request.url_root.rstrip('/') + '/' + path.lstrip('/')
    except RuntimeError:
        return f"{base}/{path.lstrip('/')}"

# ── Load ML models ────────────────────────────────────────────────────────────
print("Loading ML models...")
MODEL_PATH = os.path.join(BASE_DIR, config['model']['hybrid_model_path'])
nlp_model  = SentenceTransformer(config['model']['nlp_model'])
cv_model   = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
NLP_WEIGHT = config['model']['nlp_weight']
CV_WEIGHT  = config['model']['cv_weight']
THRESHOLD  = config['model']['threshold']

with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)

print("Models loaded!")

# ── Helper functions ──────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def append_report_to_csv(item_name, description, category, status, image_path=None):
    os.makedirs(os.path.dirname(RAW_DESCRIPTIONS_CSV), exist_ok=True)

    next_item_id = 1
    if os.path.exists(RAW_DESCRIPTIONS_CSV) and os.path.getsize(RAW_DESCRIPTIONS_CSV) > 0:
        with open(RAW_DESCRIPTIONS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_ids = [
                int(row['item_id'])
                for row in reader
                if row.get('item_id', '').isdigit()
            ]
            if existing_ids:
                next_item_id = max(existing_ids) + 1

    file_has_header = os.path.exists(RAW_DESCRIPTIONS_CSV) and os.path.getsize(RAW_DESCRIPTIONS_CSV) > 0
    with open(RAW_DESCRIPTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_DESCRIPTIONS_FIELDS)
        if not file_has_header:
            writer.writeheader()
        writer.writerow({
            'item_id': next_item_id,
            'item_name': item_name,
            'description': description,
            'category': category,
            'status': status,
            'date_reported': date.today().isoformat(),
            'image_path': image_path or '',
            'label': 0,
        })

def get_nlp_embedding(text):
    return nlp_model.encode([text])[0]

def get_cv_embedding(img_path):
    try:
        img       = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return cv_model.predict(img_array, verbose=0)[0]
    except Exception as e:
        print(f"Error extracting CV features: {e}")
        return None

def get_admin_emails():
    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT email, full_name FROM users WHERE role='admin'")
            return cursor.fetchall()
    finally:
        db.close()

def get_dominant_color_name(img_path):
    """Extract dominant color name from image."""
    COLOR_MAP = {
        "red":    ([150,50,50],   [255,100,100]),
        "orange": ([180,100,50],  [255,165,80]),
        "yellow": ([180,180,50],  [255,255,100]),
        "green":  ([30,100,30],   [100,200,100]),
        "blue":   ([30,50,100],   [100,150,255]),
        "purple": ([100,30,100],  [180,80,180]),
        "pink":   ([200,100,150], [255,180,210]),
        "brown":  ([80,40,20],    [160,100,60]),
        "grey":   ([100,100,100], [180,180,180]),
        "white":  ([200,200,200], [255,255,255]),
        "black":  ([0,0,0],       [60,60,60]),
        "navy":   ([0,0,80],      [30,30,130]),
    }
    try:
        from sklearn.cluster import KMeans
        from collections import Counter
        img = cv2.imread(img_path)
        if img is None:
            return "unknown"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        pixels = img.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
        kmeans.fit(pixels)
        counts   = Counter(kmeans.labels_)
        dominant = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
        best_color    = "unknown"
        best_distance = float("inf")
        for color_name, (low, high) in COLOR_MAP.items():
            center   = [(low[i]+high[i])/2 for i in range(3)]
            distance = sum((dominant[i]-center[i])**2 for i in range(3))**0.5
            if distance < best_distance:
                best_distance = distance
                best_color    = color_name
        return best_color
    except Exception:
        return "unknown"

def apply_color_penalty(hybrid_score, lost_color, found_color):
    """Apply penalty if colors don't match."""
    SIMILAR_COLORS = {
        "black": ["navy", "grey"],
        "navy":  ["black", "blue"],
        "grey":  ["black", "white"],
        "white": ["grey"],
        "blue":  ["navy", "purple"],
    }
    if lost_color == "unknown" or found_color == "unknown":
        return hybrid_score
    if lost_color == found_color:
        return hybrid_score
    if found_color in SIMILAR_COLORS.get(lost_color, []):
        penalty = 0.05
    else:
        penalty = 0.15
    return max(0, hybrid_score - penalty)

def find_matches(new_description, new_img_path, search_in='found', category=None):
    """Find matches with color penalty applied."""
    db = get_db()
    matches = []
    try:
        with db.cursor() as cursor:
            table = 'found_items' if search_in == 'found' else 'lost_items'
            if category:
                categories = [category] + (
                    DYNAMIC_ITEM_SLUGS_BY_CATEGORY.get(category, [])
                    if DYNAMIC_ITEM_SLUGS_BY_CATEGORY
                    else LEGACY_ITEM_SLUGS_BY_CATEGORY.get(category, [])
                )
                placeholders = ",".join(["%s"] * len(categories))
                cursor.execute(
                    f"SELECT * FROM {table} WHERE status='unmatched' AND category IN ({placeholders})",
                    categories
                )
            else:
                cursor.execute(f"SELECT * FROM {table} WHERE status='unmatched'")
            items = cursor.fetchall()

        if not items:
            return []

        new_nlp   = get_nlp_embedding(new_description)
        new_cv    = get_cv_embedding(new_img_path) if new_img_path else None
        new_color = get_dominant_color_name(new_img_path) if new_img_path else "unknown"

        for item in items:
            # NLP similarity
            item_nlp  = get_nlp_embedding(item['description'])
            nlp_score = float(cosine_similarity([new_nlp], [item_nlp])[0][0])

            # CV similarity
            cv_score = 0.0
            item_color = "unknown"
            if new_cv is not None and item['image_path']:
                full_path  = os.path.join(BASE_DIR, item['image_path'])
                item_cv    = get_cv_embedding(full_path)
                item_color = get_dominant_color_name(full_path)
                if item_cv is not None:
                    cv_score = float(cosine_similarity([new_cv], [item_cv])[0][0])

            # Hybrid/NLP-only score
            has_image = new_cv is not None
            if not has_image:
                # NLP-only mode: use a tuned threshold since the old THRESHOLD was for hybrid scoring.
                hybrid_score = nlp_score
                threshold_eff = config['model'].get('nlp_threshold', THRESHOLD * 0.8)
            else:
                hybrid_score = (NLP_WEIGHT * nlp_score) + (CV_WEIGHT * cv_score)
                threshold_eff = THRESHOLD

            # Apply color penalty (will be a no-op when colors are unknown)
            hybrid_score = apply_color_penalty(hybrid_score, new_color, item_color)

            if hybrid_score >= threshold_eff:
                matches.append({
                    'item'        : item,
                    'nlp_score'   : round(nlp_score, 4),
                    'cv_score'    : round(cv_score, 4),
                    'hybrid_score': round(hybrid_score, 4),
                    'lost_color'  : new_color,
                    'found_color' : item_color
                })

        matches.sort(key=lambda x: x['hybrid_score'], reverse=True)
    finally:
        db.close()

    return matches


def resolve_item_name(category: str, item_name: str, item_name_other: str | None):
    """Resolve an item type while keeping the selected category.

    Supports two inputs:
      1) Selecting one of the suggestions in the item type dropdown.
      2) Typing a new free-text value in item_name_other.

    Behavior:
      - If the dropdown is set to '__other__', the typed value becomes the stored item_name.
      - If the dropdown is set to a suggestion, we accept it.
      - If the dropdown value isn't recognized, we accept the typed value (item_name_other)
        as the manual free-text item type.
    """
    item_name = (item_name or '').strip()
    item_name_other = (item_name_other or '').strip() if item_name_other is not None else ''

    if not category:
        return None

    # Explicit manual typing via the '__other__' option.
    if item_name == '__other__':
        return item_name_other or None

    # Accept known suggestion (hardcoded) values.
    if item_name in ITEM_CATEGORIES.get(category, []):
        return item_name

    # Accept typed value even if item_name doesn't match a suggestion.
    if item_name_other:
        return item_name_other

    return None



def validate_category_item_type(category, item_name):
    # Kept for backward compatibility in any legacy paths.
    return item_name in ITEM_CATEGORIES.get(category, [])



def is_valid_student_id(student_id: str) -> bool:
    """Validate MSU reg number format like R245109P (letter + 6 digits + letter)."""
    if not student_id:
        return False
    student_id = student_id.strip().upper()
    import re
    return re.fullmatch(r"R\d{6}[A-Z]", student_id) is not None


def normalize_phone_digits(phone: str) -> str:
    if phone is None:
        return ""
    return "".join(ch for ch in phone if ch.isdigit())


def is_valid_phone(phone: str) -> bool:
    digits = normalize_phone_digits(phone)
    return len(digits) == 10


def parse_and_validate_report_date(date_str: str) -> date | None:
    """Return parsed date if it is today or earlier; otherwise None."""
    if not date_str:
        return None
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None

    # Reject future dates
    if parsed > date.today():
        return None
    return parsed


def is_valid_email(email: str) -> bool:
    """Validate email syntax to reject obvious wrong formats."""
    import re

    if not email:
        return False

    email = email.strip()
    if len(email) > 254:
        return False

    # local@domain.tld (basic, but rejects consecutive dots and requires a real TLD)
    # - allows common characters in local part
    # - domain labels must be alnum/hyphen separated by dots
    pattern = re.compile(
        r"^(?P<local>[A-Za-z0-9_%+\-]+(?:\.[A-Za-z0-9_%+\-]+)*)"
        r"@"
        r"(?P<domain>[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*)$"
    )

    m = pattern.match(email)
    if not m:
        return False

    # Reject consecutive dots (defensive)
    if ".." in email:
        return False

    local = m.group("local")
    domain = m.group("domain")

    # No leading/trailing dots (defensive)
    if local.startswith(".") or local.endswith("."):
        return False
    if domain.startswith(".") or domain.endswith("."):
        return False

    # Require TLD of 2+ letters
    tld = domain.rsplit(".", 1)[-1]
    if not re.fullmatch(r"[A-Za-z]{2,}", tld):
        return False

    return True

# ── Email functions ───────────────────────────────────────────────────────────
def notify_admin_lost_item(student_name, student_email, student_phone,
                            student_id, item_name, category,
                            description, colour, location, date_lost, img_path):
    admins = get_admin_emails()
    if not admins:
        return

    img_html = ""
    if img_path:
        img_url = _build_external_url(f"uploaded/{img_path.replace(os.sep, '/')}" )
        img_html = f'<br><img src="{img_url}" alt="Lost item" style="max-width:300px; border-radius:10px; margin-top:10px;">'

    logo_url = _build_external_url('static/Logo.jpeg')
    for admin in admins:
        try:
            msg = Message(
                subject=f"MSU Lost & Found — New Lost Item Report: {item_name}",
                recipients=[admin['email']]
            )
            msg.html = f"""
            <div style="font-family:Arial,sans-serif; max-width:600px; margin:0 auto;">
                <div style="background:#003366; color:white; padding:20px; text-align:center;">
                    <img src="{logo_url}" alt="MSU Logo" style="height:44px;width:auto;display:block;margin:0 auto 6px;">
                    <div style="font-family:Arial,sans-serif;font-weight:700;color:white;">Lost & Found</div>
                    <p>A student has reported a lost item</p>
                </div>
                <div style="padding:30px; background:#f9f9f9;">
                    <h2 style="color:#003366;">Student Details</h2>
                    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
                        <tr><td style="padding:8px; font-weight:bold; width:40%;">Full Name</td>
                            <td style="padding:8px;">{student_name}</td></tr>
                        <tr style="background:#eee;"><td style="padding:8px; font-weight:bold;">Email</td>
                            <td style="padding:8px;">{student_email}</td></tr>
                        <tr><td style="padding:8px; font-weight:bold;">Phone</td>
                            <td style="padding:8px;">{student_phone or 'Not provided'}</td></tr>
                        <tr style="background:#eee;"><td style="padding:8px; font-weight:bold;">Student ID</td>
                            <td style="padding:8px;">{student_id or 'Not provided'}</td></tr>
                    </table>
                    <h2 style="color:#003366;">Lost Item Details</h2>
                    <table style="width:100%; border-collapse:collapse;">
                        <tr><td style="padding:8px; font-weight:bold; width:40%;">Item Name</td>
                            <td style="padding:8px;">{item_name}</td></tr>
                        <tr style="background:#eee;"><td style="padding:8px; font-weight:bold;">Category</td>
                            <td style="padding:8px;">{category}</td></tr>
                        <tr><td style="padding:8px; font-weight:bold;">Colour</td>
                            <td style="padding:8px;">{colour or 'Not specified'}</td></tr>
                        <tr style="background:#eee;"><td style="padding:8px; font-weight:bold;">Description</td>
                            <td style="padding:8px;">{description}</td></tr>
                        <tr><td style="padding:8px; font-weight:bold;">Location Lost</td>
                            <td style="padding:8px;">{location or 'Not specified'}</td></tr>
                        <tr style="background:#eee;"><td style="padding:8px; font-weight:bold;">Date Lost</td>
                            <td style="padding:8px;">{date_lost or 'Not specified'}</td></tr>
                    </table>
                    <p style="margin-top:12px;"><strong>Item Photo:</strong>{img_html}</p>
                    <div style="margin-top:24px; text-align:center;">
                        <a href="{_build_external_url('admin')}"
                           style="background:#003366; color:white; padding:12px 30px;
                                  text-decoration:none; border-radius:5px;">
                            Go to Admin Dashboard
                        </a>
                    </div>
                </div>
                <div style="background:#003366; color:white; padding:10px; text-align:center;">
                    <div style="font-size:0.8rem;opacity:0.95;">
                        Lost & Found — Midlands State University
                    </div>
                </div>
            </div>
            """
            mail.send(msg)
            print(f"Admin notified: {admin['email']}")
        except Exception as e:
            print(f"Admin email error: {e}")


def send_match_notification(user_email, user_name, lost_item, found_item, hybrid_score):
    try:
        logo_url = _build_external_url('static/Logo.jpeg')
        msg = Message(
            subject="MSU Lost & Found — Match Found for Your Lost Item!",
            recipients=[user_email]
        )
        msg.html = f"""
        <div style="font-family:Arial,sans-serif; max-width:600px; margin:0 auto;">
            <div style="background:#003366; color:white; padding:20px; text-align:center;">
                <img src="{logo_url}" alt="MSU Logo" style="height:44px;width:auto;display:block;margin:0 auto 6px;">
                <div style="font-family:Arial,sans-serif;font-weight:700;color:white;">Lost & Found</div>
                <p>Great news! A match has been found!</p>
            <div style="padding:30px; background:#f9f9f9;">
                <p>Dear <strong>{user_name}</strong>,</p>
                <p>Our system has found a potential match for your lost item.</p>
                <table style="width:100%; border-collapse:collapse; margin:20px 0;">
                    <tr style="background:#003366; color:white;">
                        <th style="padding:10px;">Your Lost Item</th>
                        <th style="padding:10px;">Matched Found Item</th>
                    </tr>
                    <tr>
                        <td style="padding:10px; border:1px solid #ddd;">
                            <strong>{lost_item['item_name']}</strong><br>
                            <small>{lost_item['description']}</small>
                        </td>
                        <td style="padding:10px; border:1px solid #ddd;">
                            <strong>{found_item['item_name']}</strong><br>
                            <small>{found_item['description']}</small>
                        </td>
                    </tr>
                </table>
                <p><strong>Match Confidence:</strong> {hybrid_score*100:.1f}%</p>
                <p style="color:#888; font-size:0.9rem;">
                    Please log in and submit a claim with proof of ownership.
                    Visit the Lost & Found office after your claim is approved.
                </p>
                <div style="text-align:center; margin-top:20px;">
                    <a href="{_build_external_url('matches')}"
                       style="background:#003366; color:white; padding:12px 30px;
                              text-decoration:none; border-radius:5px;">
                        View Match & Claim Item
                    </a>
                </div>
            </div>
            <div style="background:#003366; color:white; padding:10px; text-align:center;">
                <div style="font-size:0.8rem;opacity:0.95;">
                    Lost & Found — Midlands State University
                </div>
            </div>
        </div>
        """
        mail.send(msg)
        print(f"Match notification sent to {user_email}")
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


def send_no_match_notification(user_email, user_name, item_name):
    try:
        msg = Message(
            subject="MSU Lost & Found — Item Reported, No Match Yet",
            recipients=[user_email]
        )
        logo_url = _build_external_url('static/Logo.jpeg')
        msg.html = f"""
        <div style="font-family:Arial,sans-serif; max-width:600px; margin:0 auto;">
            <div style="background:#003366; color:white; padding:20px; text-align:center;">
                <img src="{logo_url}" alt="MSU Logo" style="height:44px;width:auto;display:block;margin:0 auto 6px;">
                <div style="font-family:Arial,sans-serif;font-weight:700;color:white;">Lost & Found</div>
                <p>Your lost item has been reported</p>
            <div style="padding:30px; background:#f9f9f9;">
                <p>Dear <strong>{user_name}</strong>,</p>
                <p>Your lost <strong>{item_name}</strong> has been successfully reported.</p>
                <div style="background:#fff3cd; border-left:4px solid #C9973A;
                            padding:16px; border-radius:8px; margin:20px 0;">
                    <strong>⏳ No match found yet</strong><br><br>
                    Our system searched all currently reported found items 
                    but found no match at this time.<br><br>
                    <strong>What happens next:</strong><br>
                    • The Lost & Found office has been notified of your report<br>
                    • You will receive an email immediately when a match is found<br>
                    • If you do not hear from us, it means no matching item has been turned in<br>
                    • You may also visit the Lost & Found office in person
                </div>
            </div>
            <div style="background:#003366; color:white; padding:10px; text-align:center;">
                <div style="font-size:0.8rem;opacity:0.95;">
                    Lost & Found — Midlands State University
                </div>
            </div>
        </div>
        """
        mail.send(msg)
        print(f"No-match notification sent to {user_email}")
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


# ── ROUTES ────────────────────────────────────────────────────────────────────
def create_handling_task(db, claim_id: int):
    with db.cursor() as cursor:
        cursor.execute("""
            SELECT c.id AS claim_id, c.match_id, c.user_id,
                   m.lost_item_id, m.found_item_id,
                   f.location_found
            FROM claims c
            JOIN matches m ON c.match_id = m.id
            JOIN found_items f ON m.found_item_id = f.id
            WHERE c.id=%s
        """, (claim_id,))
        claim = cursor.fetchone()
        if not claim:
            return None

        destination = claim.get('location_found') or 'MSU Lost & Found Office'
        cursor.execute("""
            INSERT INTO handling_tasks
            (claim_id, match_id, lost_item_id, found_item_id, user_id, destination, status, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,'pending',NOW(),NOW())
            ON DUPLICATE KEY UPDATE updated_at=VALUES(updated_at)
        """, (
            claim['claim_id'], claim['match_id'], claim['lost_item_id'],
            claim['found_item_id'], claim['user_id'], destination
        ))
        db.commit()
        return claim


def send_delivery_confirmation_email(user_email, user_name, item_name, task_id, token):
    try:
        logo_url = _build_external_url('static/Logo.jpeg')
        confirm_url = url_for('confirm_delivery', task_id=task_id, token=token, _external=True)
        dispute_url = url_for('dispute_delivery', task_id=task_id, token=token, _external=True)
        msg = Message(
            subject="MSU Lost & Found - Confirm Item Delivery",
            recipients=[user_email]
        )
        msg.html = f"""
        <div style="font-family:Arial,sans-serif; max-width:600px; margin:0 auto;">
            <div style="background:#003366; color:white; padding:20px; text-align:center;">
                <img src="{logo_url}" alt="MSU Logo" style="height:44px;width:auto;display:block;margin:0 auto 6px;">
                <div style="font-family:Arial,sans-serif;font-weight:700;color:white;">Lost & Found</div>
            </div>
            <div style="padding:30px; background:#f9f9f9;">
                <p>Dear <strong>{user_name}</strong>,</p>
                <p>An agent has reported that your item <strong>{item_name}</strong> was delivered successfully.</p>
                <p>Please confirm whether you received the item.</p>
                <div style="margin-top:24px; text-align:center;">
                    <a href="{confirm_url}"
                       style="background:#2ECC71; color:white; padding:12px 22px; text-decoration:none; border-radius:5px;">
                        Yes, I received it
                    </a>
                    &nbsp;
                    <a href="{dispute_url}"
                       style="background:#E74C3C; color:white; padding:12px 22px; text-decoration:none; border-radius:5px;">
                        No, I did not
                    </a>
                </div>
            </div>
            <div style="background:#003366; color:white; padding:10px; text-align:center;">
                    <div style="font-size:0.8rem;opacity:0.95;">
                        Lost & Found — Midlands State University
                    </div>
            </div>
        </div>
        """
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Delivery confirmation email error: {e}")
        return False


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name  = request.form['full_name']
        email      = (request.form['email'] or '').strip()
        phone      = request.form.get('phone', '').strip()
        student_id = request.form.get('student_id', '').strip()
        password   = request.form['password']

        if not is_valid_email(email):
            flash('Please enter a valid email address (example: name@domain.com).', 'danger')
            return redirect(url_for('register'))

        if student_id and not is_valid_student_id(student_id):
            flash('Invalid Student/Reg Number format. Example: R245109P (R + 6 digits + letter).', 'danger')
            return redirect(url_for('register'))

        if phone and not is_valid_phone(phone):
            flash('Invalid phone number. Enter exactly 10 digits (Zimbabwe format).', 'danger')
            return redirect(url_for('register'))

        # store normalized phone digits (and keep it empty if user left phone blank)
        phone = normalize_phone_digits(phone) if phone else ''






        db = get_db()
        try:
            with db.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
                if cursor.fetchone():
                    flash('Email already registered!', 'danger')
                    return redirect(url_for('register'))

                hashed_pw = generate_password_hash(password)
                cursor.execute("""
                    INSERT INTO users (full_name, email, phone, student_id, password_hash)
                    VALUES (%s,%s,%s,%s,%s)
                """, (full_name, email, phone, student_id, hashed_pw))
                db.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
        finally:
            db.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = (request.form.get('email') or '').strip()
        password = request.form['password']

        if not is_valid_email(email):
            flash('Please enter a valid email address (example: name@domain.com).', 'danger')
            return redirect(url_for('login'))

        db = get_db()
        try:
            ensure_settings_tables(db)
            with db.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
                user = cursor.fetchone()

            if user is None:
                flash('Email not recognized. Please register before logging in.', 'danger')
            elif not check_password_hash(user['password_hash'], password):
                flash('Invalid password for this email. If you are not registered yet, please sign up.', 'danger')
            elif is_user_suspended(db, user['id']):
                flash('This account has been suspended. Please contact support.', 'danger')
            else:
                session['user_id']    = user['id']
                session['user_name']  = user['full_name']
                session['user_email'] = user['email']
                session['role']       = user['role']
                prefs = get_user_preferences(db, user['id'])
                session['dark_mode'] = int(prefs.get('dark_mode', 0))
                session['theme_color'] = prefs.get('theme_color', '#003366')
                log_activity(db, user['id'], user['email'], user['role'], 'admin_login' if user['role'] == 'admin' else 'login')
                flash(f"Welcome back, {user['full_name']}!", 'success')

                # login_destination is only for redirect UX.
                # Actual access is still determined by session['role'] in protected routes.
                destination = request.form.get('login_destination', 'student')
                if session.get('role') == 'admin':
                    return redirect(url_for('admin'))
                return redirect(url_for('dashboard') if destination == 'student' else url_for('dashboard'))
        finally:
            db.close()

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    try:
        ensure_match_status_column(db)
        with db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM lost_items WHERE user_id=%s",
                           (session['user_id'],))
            lost_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM found_items WHERE user_id=%s",
                           (session['user_id'],))
            found_count = cursor.fetchone()['count']

            cursor.execute("""
                SELECT m.*, l.item_name as lost_name, f.item_name as found_name
                FROM matches m
                JOIN lost_items l  ON m.lost_item_id  = l.id
                JOIN found_items f ON m.found_item_id = f.id
                WHERE l.user_id=%s OR f.user_id=%s
                ORDER BY m.matched_at DESC LIMIT 5
            """, (session['user_id'], session['user_id']))
            recent_matches = cursor.fetchall()

            cursor.execute("""
                SELECT id, item_name, category, description, location_lost,
                       date_lost, status, created_at, matched_at
                FROM lost_items
                WHERE user_id=%s
                ORDER BY created_at DESC
                LIMIT 10
            """, (session['user_id'],))
            lost_reports = cursor.fetchall()
    finally:
        db.close()

    expiry_days = 90
    expiry_cutoff = datetime.now() - timedelta(days=expiry_days)
    for report in lost_reports:
        report_date = report.get('date_lost') or report.get('created_at')
        report['tracking_date'] = report_date

        if report.get('status') == 'matched':
            report['tracking_status'] = 'Found'
            report['tracking_badge'] = 'success'
            report['tracking_note'] = 'Your lost item has been matched with a found item. Open Matches to claim it.'
        elif report.get('created_at') and report['created_at'] < expiry_cutoff:
            report['tracking_status'] = 'Expired Request'
            report['tracking_badge'] = 'danger'
            report['tracking_note'] = 'This report has been open for a long time and may no longer be valid.'
        else:
            report['tracking_status'] = 'Pending'
            report['tracking_badge'] = 'warning'
            report['tracking_note'] = 'We will notify you when a matching found item is reported.'

    return render_template('dashboard.html',
                           lost_count=lost_count,
                           found_count=found_count,
                           recent_matches=recent_matches,
                           lost_reports=lost_reports,
                           expiry_days=expiry_days)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    try:
        ensure_settings_tables(db)

        if request.method == 'POST':
            section = request.form.get('section')
            with db.cursor() as cursor:
                if section == 'general':
                    new_student_id = (request.form.get('student_id', '') or '').strip()
                    new_phone = (request.form.get('phone', '') or '').strip()

                    if new_student_id and not is_valid_student_id(new_student_id):
                        flash('Invalid Student/Reg Number format.', 'danger')
                        return redirect(url_for('settings'))

                    if new_phone and not is_valid_phone(new_phone):
                        flash('Invalid phone number. Enter exactly 10 digits (Zimbabwe format).', 'danger')
                        return redirect(url_for('settings'))

                    new_phone_norm = normalize_phone_digits(new_phone)

                    cursor.execute("""
                        UPDATE users SET student_id=%s, phone=%s WHERE id=%s
                    """, (
                        new_student_id,
                        new_phone_norm,
                        session['user_id']
                    ))

                    if session.get('role') == 'admin':
                        for key in ['support_phone', 'default_language', 'time_zone']:
                            cursor.execute("""
                                INSERT INTO app_settings (setting_key, setting_value, updated_by)
                                VALUES (%s,%s,%s)
                                ON DUPLICATE KEY UPDATE
                                    setting_value=VALUES(setting_value),
                                    updated_by=VALUES(updated_by)
                            """, (key, request.form.get(key, ''), session['user_id']))
                    flash('General settings updated.', 'success')

                elif section == 'notifications':
                    cursor.execute("""
                        INSERT INTO user_preferences
                            (user_id, email_notifications, match_alerts, report_status_updates, sound_notifications)
                        VALUES (%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE
                            email_notifications=VALUES(email_notifications),
                            match_alerts=VALUES(match_alerts),
                            report_status_updates=VALUES(report_status_updates),
                            sound_notifications=VALUES(sound_notifications)
                    """, (
                        session['user_id'],
                        1 if request.form.get('email_notifications') else 0,
                        1 if request.form.get('match_alerts') else 0,
                        1 if request.form.get('report_status_updates') else 0,
                        1 if request.form.get('sound_notifications') else 0,
                    ))
                    if session.get('role') == 'admin':
                        cursor.execute("""
                            INSERT INTO app_settings (setting_key, setting_value, updated_by)
                            VALUES ('admin_alerts', %s, %s)
                            ON DUPLICATE KEY UPDATE
                                setting_value=VALUES(setting_value),
                                updated_by=VALUES(updated_by)
                        """, ('1' if request.form.get('admin_alerts') else '0', session['user_id']))
                    flash('Notification settings updated.', 'success')

                elif section == 'reports' and session.get('role') == 'admin':
                    report_keys = [
                        'archive_resolved_reports',
                        'auto_delete_old_reports',
                        'max_image_upload_size_mb',
                        'allowed_file_types',
                        'enable_anonymous_reporting',
                    ]
                    for key in report_keys:
                        value = request.form.get(key, '')
                        if key in ['archive_resolved_reports', 'auto_delete_old_reports', 'enable_anonymous_reporting']:
                            value = '1' if request.form.get(key) else '0'
                        cursor.execute("""
                            INSERT INTO app_settings (setting_key, setting_value, updated_by)
                            VALUES (%s,%s,%s)
                            ON DUPLICATE KEY UPDATE
                                setting_value=VALUES(setting_value),
                                updated_by=VALUES(updated_by)
                        """, (key, value, session['user_id']))
                    flash('Report management settings updated.', 'success')

                elif section == 'security':
                    current_password = request.form.get('current_password', '')
                    new_password = request.form.get('new_password', '')
                    if new_password:
                        cursor.execute("SELECT password_hash FROM users WHERE id=%s", (session['user_id'],))
                        user = cursor.fetchone()
                        if not user or not check_password_hash(user['password_hash'], current_password):
                            flash('Current password is incorrect.', 'danger')
                            return redirect(url_for('settings'))
                        cursor.execute(
                            "UPDATE users SET password_hash=%s WHERE id=%s",
                            (generate_password_hash(new_password), session['user_id'])
                        )
                    cursor.execute("""
                        INSERT INTO user_preferences (user_id, two_factor_enabled)
                        VALUES (%s,%s)
                        ON DUPLICATE KEY UPDATE two_factor_enabled=VALUES(two_factor_enabled)
                    """, (session['user_id'], 1 if request.form.get('two_factor_enabled') else 0))
                    if session.get('role') == 'admin':
                        for key in ['session_timeout_minutes', 'login_attempt_limit']:
                            cursor.execute("""
                                INSERT INTO app_settings (setting_key, setting_value, updated_by)
                                VALUES (%s,%s,%s)
                                ON DUPLICATE KEY UPDATE
                                    setting_value=VALUES(setting_value),
                                    updated_by=VALUES(updated_by)
                            """, (key, request.form.get(key, ''), session['user_id']))
                    flash('Security settings updated.', 'success')

                elif section == 'appearance':
                    theme_color = request.form.get('theme_color', '#003366')
                    dark_mode = 1 if request.form.get('dark_mode') else 0
                    cursor.execute("""
                        INSERT INTO user_preferences (user_id, dark_mode, theme_color)
                        VALUES (%s,%s,%s)
                        ON DUPLICATE KEY UPDATE
                            dark_mode=VALUES(dark_mode),
                            theme_color=VALUES(theme_color)
                    """, (session['user_id'], dark_mode, theme_color))
                    session['dark_mode'] = dark_mode
                    session['theme_color'] = theme_color
                    flash('Appearance settings updated.', 'success')
                else:
                    flash('You do not have access to that settings section.', 'danger')
            db.commit()
            return redirect(url_for('settings'))

        with db.cursor() as cursor:
            cursor.execute("SELECT id, full_name, email, phone, student_id, role FROM users WHERE id=%s", (session['user_id'],))
            user = cursor.fetchone()
        app_settings = get_app_settings(db)
        prefs = get_user_preferences(db, session['user_id'])

        users = []
        activity_logs = []
        if session.get('role') == 'admin':
            with db.cursor() as cursor:
                cursor.execute("""
                    SELECT u.id, u.full_name, u.email, u.phone, u.student_id, u.role,
                           COALESCE(s.suspended, 0) AS suspended,
                           MAX(a.created_at) AS last_seen
                    FROM users u
                    LEFT JOIN user_admin_state s ON s.user_id = u.id
                    LEFT JOIN activity_logs a ON a.user_id = u.id
                    GROUP BY u.id, u.full_name, u.email, u.phone, u.student_id, u.role, s.suspended
                    ORDER BY u.full_name ASC
                """)
                users = cursor.fetchall()
                cursor.execute("""
                    SELECT * FROM activity_logs
                    WHERE action IN ('admin_login', 'login', 'user_management')
                    ORDER BY created_at DESC LIMIT 30
                """)
                activity_logs = cursor.fetchall()

        return render_template('settings.html',
                               user=user,
                               app_settings=app_settings,
                               prefs=prefs,
                               users=users,
                               activity_logs=activity_logs)
    finally:
        db.close()


@app.route('/settings/users', methods=['POST'])
def settings_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Admin access required.', 'danger')
        return redirect(url_for('dashboard'))

    target_id = request.form.get('user_id')
    action = request.form.get('action')
    db = get_db()
    try:
        ensure_settings_tables(db)
        with db.cursor() as cursor:
            if action == 'assign_role':
                role = request.form.get('role', 'student')
                cursor.execute("UPDATE users SET role=%s WHERE id=%s", (role, target_id))
                flash('User role updated.', 'success')
            elif action == 'suspend':
                cursor.execute("""
                    INSERT INTO user_admin_state (user_id, suspended, updated_by)
                    VALUES (%s,1,%s)
                    ON DUPLICATE KEY UPDATE suspended=1, updated_by=VALUES(updated_by)
                """, (target_id, session['user_id']))
                flash('User suspended.', 'success')
            elif action == 'activate':
                cursor.execute("""
                    INSERT INTO user_admin_state (user_id, suspended, updated_by)
                    VALUES (%s,0,%s)
                    ON DUPLICATE KEY UPDATE suspended=0, updated_by=VALUES(updated_by)
                """, (target_id, session['user_id']))
                flash('User reactivated.', 'success')
            elif action == 'reset_password':
                new_password = request.form.get('reset_password', '').strip()
                if not new_password:
                    flash('Enter a temporary password first.', 'danger')
                    return redirect(url_for('settings'))
                cursor.execute(
                    "UPDATE users SET password_hash=%s WHERE id=%s",
                    (generate_password_hash(new_password), target_id)
                )
                flash('Password reset.', 'success')
            elif action == 'remove':
                if str(target_id) == str(session['user_id']):
                    flash('You cannot remove your own account here.', 'danger')
                    return redirect(url_for('settings'))
                try:
                    cursor.execute("DELETE FROM users WHERE id=%s", (target_id,))
                    flash('User removed.', 'success')
                except pymysql.err.IntegrityError:
                    cursor.execute("""
                        INSERT INTO user_admin_state (user_id, suspended, updated_by)
                        VALUES (%s,1,%s)
                        ON DUPLICATE KEY UPDATE suspended=1, updated_by=VALUES(updated_by)
                    """, (target_id, session['user_id']))
                    flash('User has linked reports, so the account was suspended instead of deleted.', 'info')
            else:
                flash('Unknown user action.', 'danger')

            log_activity(db, session['user_id'], session['user_email'], session['role'], 'user_management')
            db.commit()
    finally:
        db.close()

    return redirect(url_for('settings'))

@app.route('/report_lost', methods=['GET', 'POST'])
def report_lost():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        item_name     = request.form.get('item_name', '')
        item_name_other = request.form.get('item_name_other', '')
        category      = request.form['category']
        description   = request.form['description']
        colour        = request.form['colour']
        location_lost = request.form['location_lost']
        date_lost_raw  = request.form['date_lost']

        # Validate date: allow only today or earlier.
        parsed_date_lost = parse_and_validate_report_date(date_lost_raw)
        if parsed_date_lost is None:
            flash('Invalid Date Lost. Please select today or an earlier date.', 'danger')
            return redirect(url_for('report_lost'))

        date_lost = parsed_date_lost.isoformat()

        resolved_item_name = resolve_item_name(category, item_name, item_name_other)
        if not resolved_item_name:
            flash('Please select a valid item type for the selected category (or type a value in Other).', 'danger')
            return redirect(url_for('report_lost'))

        item_name = resolved_item_name

        img_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_dir = os.path.join(UPLOAD_FOLDER, 'lost')
                os.makedirs(save_dir, exist_ok=True)
                full_path = os.path.join(save_dir, filename)
                file.save(full_path)
                img_path = f"data/raw/images/lost/{filename}"

        db = get_db()
        try:
            with db.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE id=%s", (session['user_id'],))
                student = cursor.fetchone()

            with db.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO lost_items
                    (user_id, item_name, category, description, colour, image_path, location_lost, date_lost)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (session['user_id'], item_name, category, description,
                      colour, img_path, location_lost, date_lost))
                db.commit()
                lost_id = cursor.lastrowid

            append_report_to_csv(
                item_name=item_name,
                description=description,
                category=category,
                status='lost',
                image_path=img_path
            )

            # Notify admin with student details and image
            notify_admin_lost_item(
                student_name  = student['full_name'],
                student_email = student['email'],
                student_phone = student.get('phone', ''),
                student_id    = student.get('student_id', ''),
                item_name     = item_name,
                category      = category,
                description   = description,
                colour        = colour,
                location      = location_lost,
                date_lost     = date_lost,
                img_path      = img_path
            )

            # Run hybrid matching
            full_img_path = os.path.join(BASE_DIR, img_path) if img_path else None
            matches = find_matches(description, full_img_path, search_in='found', category=category)

            if matches:
                best = matches[0]
                with db.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO matches
                        (lost_item_id, found_item_id, nlp_score, cv_score, hybrid_score)
                        VALUES (%s,%s,%s,%s,%s)
                    """, (lost_id, best['item']['id'],
                          best['nlp_score'], best['cv_score'], best['hybrid_score']))
                    match_id = cursor.lastrowid
                    db.commit()

                    cursor.execute("UPDATE lost_items  SET status='matched' WHERE id=%s", (lost_id,))
                    cursor.execute("UPDATE found_items SET status='matched' WHERE id=%s",
                                   (best['item']['id'],))
                    db.commit()

                    cursor.execute("""
                        INSERT INTO notifications (user_id, match_id, message)
                        VALUES (%s,%s,%s)
                    """, (session['user_id'], match_id,
                          f"Match found for your {item_name}!"))
                    db.commit()

                with db.cursor() as cursor:
                    cursor.execute("SELECT * FROM lost_items WHERE id=%s", (lost_id,))
                    lost_item = cursor.fetchone()

                send_match_notification(
                    session['user_email'],
                    session['user_name'],
                    lost_item,
                    best['item'],
                    best['hybrid_score']
                )

                flash(f"✅ Match found with {best['hybrid_score']*100:.1f}% confidence! "
                      f"Check your email and go to Matches to claim your item.", 'success')
            else:
                send_no_match_notification(
                    session['user_email'],
                    session['user_name'],
                    item_name
                )
                flash('📧 Item reported! No match found yet. '
                      'You will receive an email if a match is found. '
                      'The Lost & Found office has been notified.', 'info')

            return redirect(url_for('dashboard'))
        finally:
            db.close()

    return render_template('report_lost.html', item_categories=DYNAMIC_ITEM_CATEGORIES or ITEM_CATEGORIES)


@app.route('/report_found', methods=['GET', 'POST'])
def report_found():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Only administrators can report found items.', 'danger')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        item_name      = request.form.get('item_name', '')
        item_name_other = request.form.get('item_name_other', '')
        category       = request.form['category']
        description    = request.form['description']
        colour         = request.form['colour']
        location_found = request.form['location_found']
        date_found     = request.form['date_found']

        resolved_item_name = resolve_item_name(category, item_name, item_name_other)
        if not resolved_item_name:
            flash('Please select a valid item type for the selected category (or type a value in Other).', 'danger')
            return redirect(url_for('report_found'))

        item_name = resolved_item_name

        img_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_dir = os.path.join(UPLOAD_FOLDER, 'found')
                os.makedirs(save_dir, exist_ok=True)
                full_path = os.path.join(save_dir, filename)
                file.save(full_path)
                img_path = f"data/raw/images/found/{filename}"

        db = get_db()
        try:
            with db.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO found_items
                    (user_id, item_name, category, description, colour, image_path, location_found, date_found)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (session['user_id'], item_name, category, description,
                      colour, img_path, location_found, date_found))
                db.commit()
                found_id = cursor.lastrowid

            append_report_to_csv(
                item_name=item_name,
                description=description,
                category=category,
                status='found',
                image_path=img_path
            )

            full_img_path = os.path.join(BASE_DIR, img_path) if img_path else None
            matches = find_matches(description, full_img_path, search_in='lost', category=category)

            if matches:
                best = matches[0]
                with db.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO matches
                        (lost_item_id, found_item_id, nlp_score, cv_score, hybrid_score)
                        VALUES (%s,%s,%s,%s,%s)
                    """, (best['item']['id'], found_id,
                          best['nlp_score'], best['cv_score'], best['hybrid_score']))
                    db.commit()

                    cursor.execute("UPDATE found_items SET status='matched' WHERE id=%s", (found_id,))
                    cursor.execute("UPDATE lost_items  SET status='matched' WHERE id=%s",
                                   (best['item']['id'],))
                    db.commit()

                with db.cursor() as cursor:
                    cursor.execute("""
                        SELECT u.email, u.full_name, l.*
                        FROM lost_items l
                        JOIN users u ON l.user_id = u.id
                        WHERE l.id=%s
                    """, (best['item']['id'],))
                    lost_owner = cursor.fetchone()

                    cursor.execute("SELECT * FROM found_items WHERE id=%s", (found_id,))
                    found_item = cursor.fetchone()

                if lost_owner:
                    send_match_notification(
                        lost_owner['email'],
                        lost_owner['full_name'],
                        best['item'],
                        found_item,
                        best['hybrid_score']
                    )

                flash(f"✅ Found item reported! Match found with "
                      f"{best['hybrid_score']*100:.1f}% confidence. Student notified.", 'success')
            else:
                flash('✅ Found item reported! No matching lost item yet. '
                      'Students will be notified automatically when a match is found.', 'info')

            return redirect(url_for('dashboard'))
        finally:
            db.close()

    return render_template('report_found.html', item_categories=DYNAMIC_ITEM_CATEGORIES or ITEM_CATEGORIES)


@app.route('/matches')
def matches():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT m.*,
                       l.item_name as lost_name, l.description as lost_desc,
                       l.image_path as lost_image, l.category as lost_category,
                       f.item_name as found_name, f.description as found_desc,
                       f.image_path as found_image, f.category as found_category
                FROM matches m
                JOIN lost_items l  ON m.lost_item_id  = l.id
                JOIN found_items f ON m.found_item_id = f.id
                WHERE l.user_id=%s OR f.user_id=%s
                ORDER BY m.matched_at DESC
            """, (session['user_id'], session['user_id']))
            all_matches = cursor.fetchall()
    finally:
        db.close()

    return render_template('matches.html', matches=all_matches)

@app.route('/claim/<int:match_id>', methods=['GET', 'POST'])
def claim(match_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        proof_details = request.form['proof_details']
        proof_image   = None

        if 'proof_image' in request.files:
            file = request.files['proof_image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_dir = os.path.join(BASE_DIR, 'data', 'raw', 'images', 'proofs')
                os.makedirs(save_dir, exist_ok=True)
                file.save(os.path.join(save_dir, filename))
                proof_image = f"data/raw/images/proofs/{filename}"

        db = get_db()
        try:
            ensure_match_status_column(db)
            with db.cursor() as cursor:
                # 1) Create claim record (for admin review)
                cursor.execute("""
                    INSERT INTO claims (match_id, user_id, proof_details, proof_image, status)
                    VALUES (%s,%s,%s,%s,'pending')
                """, (match_id, session['user_id'], proof_details, proof_image))

                # 2) Update match status immediately so the UI shows "Claimed"
                cursor.execute("UPDATE matches SET status='claimed' WHERE id=%s", (match_id,))
                db.commit()

            flash('✅ Claim submitted! Your match is now marked as Claimed. Admin will review your proof.', 'success')
            return redirect(url_for('matches'))
        finally:
            db.close()

    return render_template('claim.html', match_id=match_id)

@app.route('/admin')
def admin():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Admin access required!', 'danger')
        return redirect(url_for('dashboard'))

    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as c FROM users")
            total_users = cursor.fetchone()['c']
            cursor.execute("SELECT COUNT(*) as c FROM lost_items")
            total_lost = cursor.fetchone()['c']
            cursor.execute("SELECT COUNT(*) as c FROM found_items")
            total_found = cursor.fetchone()['c']
            cursor.execute("SELECT COUNT(*) as c FROM matches")
            total_matches = cursor.fetchone()['c']

            cursor.execute("""
                SELECT l.*, u.full_name, u.email, u.phone, u.student_id
                FROM lost_items l
                JOIN users u ON l.user_id = u.id
                ORDER BY l.created_at DESC LIMIT 10
            """)
            recent_lost = cursor.fetchall()

            cursor.execute("""
                SELECT c.*, u.full_name, u.email,
                       l.item_name as lost_name, f.item_name as found_name
                FROM claims c
                JOIN users u    ON c.user_id     = u.id
                JOIN matches m  ON c.match_id    = m.id
                JOIN lost_items  l ON m.lost_item_id  = l.id
                JOIN found_items f ON m.found_item_id = f.id
                WHERE c.status='pending'
                ORDER BY c.created_at DESC
            """)
            pending_claims = cursor.fetchall()

            cursor.execute("""
                SELECT ht.*, u.full_name AS claimant_name, u.email AS claimant_email,
                       agent.full_name AS agent_name,
                       l.item_name AS lost_name, f.item_name AS found_name
                FROM handling_tasks ht
                JOIN users u ON ht.user_id = u.id
                LEFT JOIN users agent ON ht.agent_id = agent.id
                JOIN lost_items l ON ht.lost_item_id = l.id
                JOIN found_items f ON ht.found_item_id = f.id
                ORDER BY
                    FIELD(ht.status, 'pending', 'accepted', 'failed', 'delivered', 'user_disputed', 'user_confirmed'),
                    ht.created_at DESC
            """)
            handling_tasks = cursor.fetchall()
    finally:
        db.close()

    return render_template('admin.html',
                           total_users=total_users,
                           total_lost=total_lost,
                           total_found=total_found,
                           total_matches=total_matches,
                           recent_lost=recent_lost,
                           pending_claims=pending_claims,
                           handling_tasks=handling_tasks)

@app.route('/admin/claim/<int:claim_id>/<action>')
def process_claim(claim_id, action):
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))

    status = 'approved' if action == 'approve' else 'rejected'
    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("UPDATE claims SET status=%s WHERE id=%s", (status, claim_id))
            db.commit()

            cursor.execute("""
                SELECT u.email, u.full_name FROM claims c
                JOIN users u ON c.user_id = u.id WHERE c.id=%s
            """, (claim_id,))
            user = cursor.fetchone()

        if user:
            logo_url = _build_external_url('static/Logo.jpeg')
            msg = Message(
                subject=f"MSU Lost & Found — Claim {status.title()}",
                recipients=[user['email']]
            )
            if status == 'approved':
                msg.html = f"""
                <div style="font-family:Arial,sans-serif; max-width:600px; margin:0 auto;">
                    <div style="background:#003366; color:white; padding:20px; text-align:center;">
                <img src="{logo_url}" alt="MSU Logo" style="height:44px;width:auto;display:block;margin:0 auto 6px;">
                <div style="font-family:Arial,sans-serif;font-weight:700;color:white;">Lost & Found</div>
                    </div>
                    <div style="padding:30px; background:#f9f9f9;">
                        <p>Dear <strong>{user['full_name']}</strong>,</p>
                        <div style="background:#d4edda; border-left:4px solid #2ECC71;
                                    padding:16px; border-radius:8px; margin:20px 0;">
                            <strong>✅ Your claim has been APPROVED!</strong><br><br>
                            Please visit the MSU Lost & Found office with your student ID
                            to collect your item.<br><br>
                            <strong>Office Location:</strong> Administration Block, Room 101<br>
                            <strong>Office Hours:</strong> Monday–Friday, 8:00 AM – 4:30 PM
                        </div>
                    </div>
                    <div style="background:#003366; color:white; padding:10px; text-align:center;">
                <div style="font-size:0.8rem;opacity:0.95;">
                    Lost & Found — Midlands State University
                </div>
                    </div>
                </div>
                """
            else:
                msg.html = f"""
                <div style="font-family:Arial,sans-serif; max-width:600px; margin:0 auto;">
                    <div style="background:#003366; color:white; padding:20px; text-align:center;">
                <img src="{logo_url}" alt="MSU Logo" style="height:44px;width:auto;display:block;margin:0 auto 6px;">
                <div style="font-family:Arial,sans-serif;font-weight:700;color:white;">Lost & Found</div>
                    </div>
                    <div style="padding:30px; background:#f9f9f9;">
                        <p>Dear <strong>{user['full_name']}</strong>,</p>
                        <div style="background:#f8d7da; border-left:4px solid #E74C3C;
                                    padding:16px; border-radius:8px; margin:20px 0;">
                            <strong>❌ Your claim has been REJECTED</strong><br><br>
                            Unfortunately your claim was not approved.
                            Please visit the Lost & Found office for further assistance.
                        </div>
                    </div>
                    <div style="background:#003366; color:white; padding:10px; text-align:center;">
                        <div style="font-size:0.8rem;opacity:0.95;">
                            Lost & Found — Midlands State University
                        </div>
                    </div>
                </div>
                """
            mail.send(msg)

        if status == 'approved':
            create_handling_task(db, claim_id)

        flash(f'Claim {status} successfully! Student has been notified.', 'success')
    finally:
        db.close()

    return redirect(url_for('admin'))


@app.route('/admin/handling/<int:task_id>/accept', methods=['POST'])
def accept_handling_task(task_id):
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))

    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                UPDATE handling_tasks
                SET status='accepted', agent_id=%s, accepted_at=NOW(), updated_at=NOW()
                WHERE id=%s AND status IN ('pending','failed')
            """, (session['user_id'], task_id))
            db.commit()
        flash('Handling task accepted.', 'success')
    finally:
        db.close()

    return redirect(url_for('admin'))


@app.route('/admin/handling/<int:task_id>/report', methods=['POST'])
def report_handling_task(task_id):
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))

    outcome = request.form.get('outcome')
    agent_report = (request.form.get('agent_report') or '').strip()
    if outcome not in ('delivered', 'failed'):
        flash('Invalid handling outcome.', 'danger')
        return redirect(url_for('admin'))

    db = get_db()
    try:
        token = secrets.token_urlsafe(24) if outcome == 'delivered' else None
        with db.cursor() as cursor:
            cursor.execute("""
                UPDATE handling_tasks
                SET status=%s,
                    agent_id=COALESCE(agent_id, %s),
                    agent_report=%s,
                    confirmation_token=COALESCE(%s, confirmation_token),
                    completed_at=NOW(),
                    updated_at=NOW()
                WHERE id=%s AND status IN ('pending','accepted','failed')
            """, (outcome, session['user_id'], agent_report, token, task_id))
            db.commit()

            cursor.execute("""
                SELECT ht.id, ht.confirmation_token,
                       u.email, u.full_name, l.item_name
                FROM handling_tasks ht
                JOIN users u ON ht.user_id = u.id
                JOIN lost_items l ON ht.lost_item_id = l.id
                WHERE ht.id=%s
            """, (task_id,))
            task = cursor.fetchone()

        if outcome == 'delivered' and task and task.get('confirmation_token'):
            send_delivery_confirmation_email(
                task['email'],
                task['full_name'],
                task['item_name'],
                task['id'],
                task['confirmation_token']
            )
            flash('Delivery marked successful. Student confirmation email sent.', 'success')
        else:
            flash('Delivery issue recorded for clarity.', 'info')
    finally:
        db.close()

    return redirect(url_for('admin'))


@app.route('/delivery/confirm/<int:task_id>/<token>')
def confirm_delivery(task_id, token):
    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                UPDATE handling_tasks
                SET status='user_confirmed', user_confirmed_at=NOW(), updated_at=NOW()
                WHERE id=%s AND confirmation_token=%s AND status='delivered'
            """, (task_id, token))
            db.commit()
            ok = cursor.rowcount > 0
    finally:
        db.close()

    flash('Thank you. Your delivery confirmation has been recorded.' if ok else
          'This delivery confirmation link is invalid or has already been used.', 'success' if ok else 'danger')
    return redirect(url_for('login'))


@app.route('/delivery/dispute/<int:task_id>/<token>')
def dispute_delivery(task_id, token):
    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                UPDATE handling_tasks
                SET status='user_disputed', user_confirmed_at=NOW(), updated_at=NOW()
                WHERE id=%s AND confirmation_token=%s AND status='delivered'
            """, (task_id, token))
            db.commit()
            ok = cursor.rowcount > 0
    finally:
        db.close()

    flash('Thank you. The delivery issue has been recorded for admin follow-up.' if ok else
          'This delivery confirmation link is invalid or has already been used.', 'info' if ok else 'danger')
    return redirect(url_for('login'))


@app.route('/chat')
def chat_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')

# Chat backend (WhatsApp-like via polling)
# Mode: per-user private room with admin replies.

def _get_or_create_support_room(db, user_id: int) -> int:
    # Ensure one room per user
    with db.cursor() as cursor:
        cursor.execute(
            "SELECT id FROM support_rooms WHERE user_id=%s AND room_type='user_admin' LIMIT 1",
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return row['id']

        cursor.execute(
            "INSERT INTO support_rooms (user_id, room_type, created_at) VALUES (%s,'user_admin',NOW())",
            (user_id,)
        )
        db.commit()
        return cursor.lastrowid


def _fetch_messages(db, room_id: int, after_id: int = 0):
    with db.cursor() as cursor:
        cursor.execute(
            """
                SELECT id, room_id, sender_user_id, message, created_at
                FROM chat_messages
                WHERE room_id=%s AND id>%s
                ORDER BY id ASC
            """,
            (room_id, after_id)
        )
        return cursor.fetchall()


def _room_exists(db, room_id: int) -> bool:
    with db.cursor() as cursor:
        cursor.execute(
            "SELECT id FROM support_rooms WHERE id=%s AND room_type='user_admin' LIMIT 1",
            (room_id,)
        )
        return cursor.fetchone() is not None


def _get_auto_reply_sender_id(db):
    with db.cursor() as cursor:
        cursor.execute("SELECT id FROM users WHERE role='admin' ORDER BY id ASC LIMIT 1")
        admin = cursor.fetchone()
        return admin['id'] if admin else None


def _chat_table_supports_attachments(db) -> bool:
    with db.cursor() as cursor:
        cursor.execute("SHOW COLUMNS FROM chat_messages LIKE 'attachment_type'")
        return cursor.fetchone() is not None


def _room_has_messages(db, room_id: int) -> bool:
    with db.cursor() as cursor:
        cursor.execute("SELECT 1 FROM chat_messages WHERE room_id=%s LIMIT 1", (room_id,))
        return cursor.fetchone() is not None


def _insert_admin_auto_reply(db, room_id: int):
    admin_id = _get_auto_reply_sender_id(db)
    if not admin_id:
        return None

    auto_reply = (
        "Thank you for contacting MSU Lost & Found. "
        "Any available agent will be in touch as soon as possible."
    )
    with db.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO chat_messages (room_id, sender_user_id, message, created_at)
            VALUES (%s,%s,%s,NOW())
            """,
            (room_id, admin_id, auto_reply)
        )
        db.commit()
        return cursor.lastrowid


@app.route('/chat/send', methods=['POST'])
def chat_send():
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401

    message = ''
    if request.content_type and 'application/json' in request.content_type.lower():
        payload = request.get_json(silent=True) or {}
        message = (payload.get('message') or '').strip()
    else:
        message = (request.form.get('message') or '').strip()

    image_file = request.files.get('image')
    attachment_type = None
    attachment_path = None

    if image_file and image_file.filename:
        if allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            save_dir = os.path.join(BASE_DIR, 'data', 'raw', 'images', 'chat')
            os.makedirs(save_dir, exist_ok=True)
            # save flat; collisions are acceptable, but we can reduce by prefixing user+time
            filename = f"{session['user_id']}_{int(datetime.utcnow().timestamp())}_{filename}"
            full_path = os.path.join(save_dir, filename)
            image_file.save(full_path)
            attachment_type = 'image'
            attachment_path = f"data/raw/images/chat/{filename}"

    # allow image-only messages (clarification) with empty text
    if not message and not attachment_path:
        return jsonify({'error': 'empty_message'}), 400
    if attachment_path:
        message = (message or '').strip() + f"\n[image]{attachment_path}"

    db = get_db()
    try:
        room_id = _get_or_create_support_room(db, session['user_id'])
        should_auto_reply = not _room_has_messages(db, room_id)

        with db.cursor() as cursor:
            if _chat_table_supports_attachments(db):
                cursor.execute(
                    """
                    INSERT INTO chat_messages
                        (room_id, sender_user_id, message, attachment_type, attachment_path, created_at)
                    VALUES (%s,%s,%s,%s,%s,NOW())
                    """,
                    (room_id, session['user_id'], message, attachment_type, attachment_path)
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO chat_messages
                        (room_id, sender_user_id, message, created_at)
                    VALUES (%s,%s,%s,NOW())
                    """,
                    (room_id, session['user_id'], message)
                )
            db.commit()
            msg_id = cursor.lastrowid

        auto_reply_id = None
        if should_auto_reply:
            auto_reply_id = _insert_admin_auto_reply(db, room_id)
        return jsonify({'ok': True, 'message_id': msg_id, 'auto_reply_id': auto_reply_id})
    except Exception as exc:
        app.logger.exception('chat_send failed')
        os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
        with open(os.path.join(BASE_DIR, 'logs', 'chat_send_error.log'), 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.utcnow().isoformat()}] chat_send error: {exc}\n")
        return jsonify({'error': 'server_error', 'message': str(exc)}), 500
    finally:
        db.close()


@app.route('/chat/admin/send', methods=['POST'])
def chat_admin_send():
    if session.get('role') != 'admin':
        return jsonify({'error': 'unauthorized'}), 401

    # Support both JSON (legacy) and FormData (current admin_chat.html)
    room_id_raw = None
    message_raw = None
    if request.content_type and 'application/json' in request.content_type.lower():
        payload = request.get_json(silent=True) or {}
        room_id_raw = payload.get('room_id')
        message_raw = payload.get('message')
    else:
        room_id_raw = request.form.get('room_id')
        message_raw = request.form.get('message')

    try:
        room_id = int(room_id_raw)
    except Exception:
        return jsonify({'error': 'invalid_room_id'}), 400

    message = (message_raw or '').strip()

    image_file = request.files.get('image')
    attachment_type = None
    attachment_path = None

    if image_file and image_file.filename and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        save_dir = os.path.join(BASE_DIR, 'data', 'raw', 'images', 'chat')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{session['user_id']}_{int(datetime.utcnow().timestamp())}_{filename}"
        full_path = os.path.join(save_dir, filename)
        image_file.save(full_path)

        attachment_type = 'image'
        attachment_path = f"data/raw/images/chat/{filename}"

    # allow image-only messages
    if not message and not attachment_path:
        return jsonify({'error': 'empty_message'}), 400
    if attachment_path:
        message = (message or '').strip() + f"\n[image]{attachment_path}"

    db = get_db()
    try:
        if not _room_exists(db, room_id):
            return jsonify({'error': 'room_not_found'}), 404

        with db.cursor() as cursor:
            if _chat_table_supports_attachments(db):
                cursor.execute(
                    """
                    INSERT INTO chat_messages
                        (room_id, sender_user_id, message, attachment_type, attachment_path, created_at)
                    VALUES (%s,%s,%s,%s,%s,NOW())
                    """,
                    (room_id, session['user_id'], message, attachment_type, attachment_path)
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO chat_messages
                        (room_id, sender_user_id, message, created_at)
                    VALUES (%s,%s,%s,NOW())
                    """,
                    (room_id, session['user_id'], message)
                )
            db.commit()
            msg_id = cursor.lastrowid

        return jsonify({'ok': True, 'message_id': msg_id})
    finally:
        db.close()


@app.route('/chat/poll', methods=['GET'])
def chat_poll():
    # User polls their private room.
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401

    after_id = request.args.get('after_id', '0')
    try:
        after_id = int(after_id)
    except Exception:
        after_id = 0

    db = get_db()
    try:
        room_id = _get_or_create_support_room(db, session['user_id'])
        rows = _fetch_messages(db, room_id, after_id=after_id)
        return jsonify({
            'room_id': room_id,
            'messages': [
                {
                    'id': r['id'],
                    'sender_user_id': r['sender_user_id'],
                    'message': r['message'],
                    'created_at': r['created_at'].strftime('%Y-%m-%d %H:%M:%S') if r.get('created_at') else None
                } for r in rows
            ]
        })
    except Exception as e:
        # Return JSON instead of Flask HTML error page so the frontend can display it.
        return jsonify({'error': 'poll_failed', 'message': str(e)}), 500
    finally:
        db.close()



@app.route('/admin/chat')
def admin_chat_page():
    if session.get('role') != 'admin':
        flash('Admin access required!', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('admin_chat.html')


@app.route('/admin/chat/rooms', methods=['GET'])
def admin_chat_rooms():
    if session.get('role') != 'admin':
        return jsonify({'error': 'unauthorized'}), 401

    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute(
                """
                SELECT sr.id AS room_id, sr.user_id, u.full_name, u.email
                FROM support_rooms sr
                JOIN users u ON u.id = sr.user_id
                WHERE sr.room_type='user_admin' AND u.role!='admin'
                ORDER BY sr.created_at DESC
                """
            )
            rows = cursor.fetchall()
        return jsonify({
            'rooms': [
                {
                    'room_id': r['room_id'],
                    'user_id': r['user_id'],
                    'full_name': r['full_name'],
                    'email': r['email']
                }
                for r in rows
            ]
        })
    finally:
        db.close()


@app.route('/admin/chat/messages', methods=['GET'])
def admin_chat_messages():
    if session.get('role') != 'admin':
        return jsonify({'error': 'unauthorized'}), 401

    room_id = request.args.get('room_id')
    after_id = request.args.get('after_id', '0')

    try:
        room_id = int(room_id)
        after_id = int(after_id)
    except Exception:
        return jsonify({'error': 'invalid_params'}), 400

    db = get_db()
    try:
        if not _room_exists(db, room_id):
            return jsonify({'error': 'room_not_found'}), 404

        rows = _fetch_messages(db, room_id, after_id=after_id)
        return jsonify({
            'messages': [
                {
                    'id': r['id'],
                    'sender_user_id': r['sender_user_id'],
                    'message': r['message'],
                    'created_at': r['created_at'].strftime('%Y-%m-%d %H:%M:%S') if r.get('created_at') else None
                } for r in rows
            ]
        })
    finally:
        db.close()

@app.route('/admin/chat/clear', methods=['POST'])
def admin_chat_clear():
    if session.get('role') != 'admin':
        return jsonify({'error': 'unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    room_id = payload.get('room_id')
    try:
        room_id = int(room_id)
    except Exception:
        return jsonify({'error': 'invalid_room_id'}), 400

    db = get_db()
    try:
        if not _room_exists(db, room_id):
            return jsonify({'error': 'room_not_found'}), 404

        with db.cursor() as cursor:
            cursor.execute("DELETE FROM chat_messages WHERE room_id=%s", (room_id,))
            db.commit()
        return jsonify({'ok': True})
    finally:
        db.close()


@app.route('/admin/chat/delete', methods=['POST'])
def admin_chat_delete():
    if session.get('role') != 'admin':
        return jsonify({'error': 'unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    room_id = payload.get('room_id')
    try:
        room_id = int(room_id)
    except Exception:
        return jsonify({'error': 'invalid_room_id'}), 400

    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute(
                """
                SELECT sr.id AS room_id, sr.user_id, u.role
                FROM support_rooms sr
                JOIN users u ON u.id = sr.user_id
                WHERE sr.id=%s AND sr.room_type='user_admin'
                LIMIT 1
                """,
                (room_id,)
            )
            row = cursor.fetchone()
            if not row:
                return jsonify({'error': 'room_not_found'}), 404
            if row.get('role') == 'admin':
                return jsonify({'error': 'cannot_delete_admin'}), 403

            # Delete the user; FOREIGN KEY constraints will cascade and remove support_rooms and chat_messages
            cursor.execute("DELETE FROM users WHERE id=%s", (row['user_id'],))
            db.commit()
            if cursor.rowcount == 0:
                return jsonify({'error': 'delete_failed'}), 500

        return jsonify({'ok': True})
    finally:
        db.close()


@app.route('/uploaded/<path:filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(BASE_DIR, filename)


if __name__ == '__main__':
    app.run(debug=config['flask']['debug'], host='0.0.0.0', port=5000)
