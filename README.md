# MSU Lost and Found ML System

A Flask-based lost and found web application for Midlands State University. The system lets students report lost items, administrators manage found items, and both sides communicate through a support chat workflow.

## Features

- Student registration, login, dashboard, and lost item reporting
- Admin dashboard for found item reporting, matching, user management, and handling tasks
- NLP-based text similarity using Sentence Transformers
- Computer vision image feature extraction using MobileNetV2
- Hybrid matching between lost and found item reports
- WhatsApp-style support chat with polling and image attachments
- Email notifications for reports, matches, and delivery confirmation
- MySQL-backed persistence for users, reports, matches, settings, and chat messages

## Tech Stack

- Python, Flask, Jinja templates
- MySQL and PyMySQL
- HTML, CSS, JavaScript, Bootstrap
- TensorFlow/Keras, Sentence Transformers, scikit-learn, OpenCV
- Flask-Mail

## Project Structure

```text
config/      Example configuration
data/        Dataset files and upload folders
docs/        Project documentation
models/      Generated/trained model artifacts
notebooks/   Exploration and model development notebooks
reports/     Evaluation results and figures
scripts/     Utility and smoke-test scripts
sql/         Database schema files
src/         Flask application source
```

## Local Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create the local configuration file:

```powershell
Copy-Item config\config.example.yaml config\config.yaml
```

4. Update `config/config.yaml` with your MySQL and email settings.
5. Create the MySQL database using the SQL files in `sql/`.
6. Run the application:

```bash
python run_flask_local.py
```

The app runs at `http://127.0.0.1:5000`.

## Configuration

The application reads local settings from `config/config.yaml`. That file is ignored by Git so database passwords, Flask secret keys, and email credentials stay on the developer machine.

You can also override sensitive values with environment variables:

- `MSU_DB_HOST`
- `MSU_DB_PORT`
- `MSU_DB_NAME`
- `MSU_DB_USER`
- `MSU_DB_PASSWORD`
- `MSU_FLASK_SECRET_KEY`
- `MSU_MAIL_SERVER`
- `MSU_MAIL_PORT`
- `MSU_MAIL_USERNAME`
- `MSU_MAIL_PASSWORD`
- `MSU_MAIL_SENDER`

See `.env.example` and `config/config.example.yaml` for safe placeholder values.


## GitHub Notes

The real `config/config.yaml`, `.env`, logs, temporary scripts, chat uploads, proof uploads, and generated feature files are intentionally ignored by Git. This keeps credentials and local runtime data out of the repository.

Before publishing, review `docs/GITHUB_CHECKLIST.md`.

## CV Summary

Developed a Flask-based Lost and Found system with authentication, admin dashboards, item reporting workflows, ML-powered matching, support chat, image uploads, email notifications, and MySQL data storage.
