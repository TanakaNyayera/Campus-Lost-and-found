import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

WEB_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "web_app")
sys.path.insert(0, WEB_APP_DIR)

from app import app  # noqa: E402


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
