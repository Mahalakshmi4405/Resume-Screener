import os
import requests
import uuid
import sqlite3
import csv
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from pdfminer.high_level import extract_text as pdf_extract_text
import docx
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Gemini imports
from google import genai
from google.genai import types

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app = Flask(__name__)
app.secret_key = "supersecret"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Gemini Client ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB6bvdq2cYkbQlAfPv-NN4VpqxVqqoTCB4")
client = genai.Client(api_key=GEMINI_API_KEY)

# --- DB Init ---
def init_db():
    conn = sqlite3.connect("resumes.db")
    c = conn.cursor()
    # Added extra columns for Gemini insights
    c.execute('''CREATE TABLE IF NOT EXISTS resumes
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  resume_text TEXT UNIQUE,
                  job_desc TEXT,
                  job_link TEXT,
                  similarity REAL,
                  gemini_good TEXT,
                  gemini_bad TEXT,
                  gemini_recommendations TEXT,
                  gemini_rating REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_resume(filepath):
    if filepath.lower().endswith(".pdf"):
        try:
            return pdf_extract_text(filepath) or ""
        except Exception as e:
            return f"[Error extracting PDF: {e}]"
    elif filepath.lower().endswith(".docx") or filepath.lower().endswith(".doc"):
        try:
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs]) or ""
        except Exception as e:
            return f"[Error extracting DOCX: {e}]"
    return ""

def scrape_text_from_url(url, max_chars=5000):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        text = " ".join(soup.stripped_strings)
        return text[:max_chars]
    except Exception as e:
        return f"[Error scraping URL: {e}]"

def compute_similarity(text1, text2):
    try:
        if not text1:
            text1 = ""
        if not text2:
            text2 = ""
        if text1.strip() == "" and text2.strip() == "":
            return 0.0
        vectorizer = TfidfVectorizer(stop_words="english").fit([text1, text2])
        tfidf = vectorizer.transform([text1, text2])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(round(sim * 100, 2))
    except Exception:
        return 0.0

def exists_resume_text(text):
    conn = sqlite3.connect("resumes.db")
    c = conn.cursor()
    c.execute("SELECT id FROM resumes WHERE resume_text = ?", (text,))
    row = c.fetchone()
    conn.close()
    return row is not None

def save_metadata(resume_id, filename, resume_text, job_desc, job_link, similarity,
                  good, bad, recs, rating):
    conn = sqlite3.connect("resumes.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO resumes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                  (resume_id, filename, resume_text, job_desc, job_link, similarity,
                   good, bad, recs, rating))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False
    conn.close()
    export_to_csv()
    return True

def export_to_csv():
    conn = sqlite3.connect("resumes.db")
    c = conn.cursor()
    c.execute("SELECT id, filename, resume_text, job_desc, job_link, similarity, gemini_good, gemini_bad, gemini_recommendations, gemini_rating FROM resumes")
    rows = c.fetchall()
    conn.close()

    with open("resumes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Filename", "Resume_Content", "Job_Description", "Job_Link", "Similarity", "Good_Points", "Bad_Points", "Recommendations", "Rating"])
        writer.writerows(rows)

# --- Gemini Resume Review ---
def analyze_resume_with_gemini(filepath):
    try:
        uploaded_file = client.files.upload(file=filepath)
        prompt = """
        Analyze this resume. Provide:
        1. Good points (strengths)
        2. Bad points (weaknesses)
        3. Recommendations for improvement
        4. A rating out of 10
        Return the response in JSON with keys: good, bad, recommendations, rating
        """
        contents = [uploaded_file, prompt]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        import json
        try:
            result = json.loads(response.text)
        except Exception:
            result = {"good": "N/A", "bad": "N/A", "recommendations": "N/A", "rating": 0}

        return (
            result.get("good", ""),
            result.get("bad", ""),
            result.get("recommendations", ""),
            float(result.get("rating", 0))
        )
    except Exception as e:
        return ("Error analyzing resume", str(e), "Try again later", 0)

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        job_desc = request.form.get("job_desc", "").strip()
        job_link = request.form.get("job_link", "").strip()
        resume_file = request.files.get("resume_file")
        resume_link = request.form.get("resume_link", "").strip()

        if job_link:
            scraped = scrape_text_from_url(job_link)
            if scraped.startswith("[Error"):
                flash(f"❌ Failed to scrape job link: {scraped}")
                return redirect(request.url)
            job_desc = scraped

        if not job_desc:
            flash("❌ Please enter a job description (text or job link).")
            return redirect(request.url)

        resume_id = str(uuid.uuid4())
        filename, filepath = None, None

        if resume_file and resume_file.filename != "" and allowed_file(resume_file.filename):
            safe_name = secure_filename(resume_file.filename)
            filename = f"{resume_id}_{safe_name}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(filepath)

        elif resume_link:
            if not (resume_link.lower().endswith(".pdf") or resume_link.lower().endswith(".docx")):
                flash("❌ Only .pdf or .docx resume links are allowed.")
                return redirect(request.url)
            try:
                response = requests.get(resume_link, timeout=10)
                response.raise_for_status()
                ext = resume_link.split(".")[-1].lower()
                filename = f"{resume_id}_link.{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                flash(f"❌ Error fetching resume link: {e}")
                return redirect(request.url)

        else:
            flash("❌ Please upload a resume file or provide a resume link.")
            return redirect(request.url)

        resume_text = extract_text_from_resume(filepath).strip()
        if resume_text.startswith("[Error"):
            flash(f"❌ {resume_text}")
            return redirect(request.url)

        if exists_resume_text(resume_text):
            flash("⚠️ Duplicate resume detected (text already in DB). Skipping insert.")
            return redirect(request.url)

        sim_score = compute_similarity(resume_text, job_desc)

        # Gemini review
        good, bad, recs, rating = analyze_resume_with_gemini(filepath)

        saved = save_metadata(resume_id, filename, resume_text, job_desc, job_link or None, sim_score,
                              good, bad, recs, rating)
        if not saved:
            flash("⚠️ Duplicate detected or insert failed.")
            return redirect(request.url)

        flash(f"✅ Resume stored! Similarity: {sim_score}%, Gemini Rating: {rating}/10")
        return redirect(url_for("upload_resume"))

    return render_template("upload.html")

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"content": "", "error": "No URL provided"}), 400
    try:
        content = scrape_text_from_url(url, max_chars=5000)
        if content.startswith("[Error"):
            return jsonify({"content": "", "error": content}), 500
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"content": "", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
