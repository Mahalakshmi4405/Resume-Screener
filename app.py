import os
import re
import json
import uuid
import sqlite3
import csv
import logging
import requests
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Gemini imports
from google import genai

# ---------- CONFIG ----------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
DB_PATH = "resumes.db"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Flask init
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "supersecret")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini client configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_ENABLED = True
    logger.info("Gemini client enabled.")
else:
    client = None
    GEMINI_ENABLED = False
    logger.warning("GEMINI_API_KEY not set — Gemini calls will be skipped. Set GEMINI_API_KEY env var to enable.")

# ---------- DB INIT & SCHEMA ----------
def init_db():
    logger.info("Initializing database...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
                  gemini_rating REAL,
                  gemini_match_score REAL,
                  gemini_comparison TEXT,
                  status TEXT,
                  progress INTEGER)''')
    conn.commit()
    conn.close()
    logger.info("Database ready.")
init_db()

def update_db_columns(resume_id, **kwargs):
    if not kwargs:
        return
    keys = list(kwargs.keys())
    placeholders = ", ".join([f"{k} = ?" for k in keys])
    vals = [kwargs[k] for k in keys] + [resume_id]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    sql = f"UPDATE resumes SET {placeholders} WHERE id = ?"
    c.execute(sql, vals)
    conn.commit()
    conn.close()

# ---------- HELPERS ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_resume(filepath):
    logger.info(f"Extracting text from resume: {filepath}")
    if filepath.lower().endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip() or ""
        except Exception as e:
            logger.exception("PDF extraction failed")
            return f"[Error extracting PDF: {e}]"
    elif filepath.lower().endswith((".docx", ".doc")):
        try:
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs]) or ""
        except Exception as e:
            logger.exception("DOCX extraction failed")
            return f"[Error extracting DOCX: {e}]"
    return ""

def scrape_text_from_url(url, max_chars=8000):
    logger.info(f"Scraping job description from {url}")
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        text = " ".join(soup.stripped_strings)
        return text[:max_chars]
    except Exception as e:
        logger.exception("Error scraping URL")
        return f"[Error scraping URL: {e}]"

def tokenize_keywords(text, top_k=40):
    text = (text or "").lower()
    words = re.findall(r"\b[a-zA-Z0-9\+\#\.\-]{2,}\b", text)
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS and not w.isdigit()]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]

def compute_similarity(text1, text2):
    logger.info("Computing similarity between resume and job description...")
    try:
        t1 = (text1 or "").strip()
        t2 = (text2 or "").strip()
        if t1 == "" and t2 == "":
            return 0.0
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        tfidf = vectorizer.fit_transform([t1, t2])
        tfidf_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        kw1 = set(tokenize_keywords(t1, top_k=60))
        kw2 = set(tokenize_keywords(t2, top_k=60))
        jac = len(kw1 & kw2) / len(kw1 | kw2) if kw1 and kw2 else 0.0
        combined = (0.75 * tfidf_sim) + (0.25 * jac)
        sim = round(combined * 100, 2)
        logger.info(f"Similarity score: {sim}%")
        return sim
    except Exception:
        logger.exception("Similarity computation failed")
        return 0.0

def exists_resume_text(text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM resumes WHERE resume_text = ?", (text,))
    row = c.fetchone()
    conn.close()
    return row is not None

def save_initial_metadata(resume_id, filename, resume_text, job_desc, job_link, similarity):
    logger.info("Inserting initial metadata row (processing)...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""INSERT INTO resumes
                     (id, filename, resume_text, job_desc, job_link, similarity, status, progress)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (resume_id, filename, resume_text, job_desc, job_link, similarity, "processing", 5))
        conn.commit()
    except sqlite3.IntegrityError:
        logger.warning("Duplicate resume detected while inserting initial metadata.")
        conn.close()
        return False
    conn.close()
    return True

def export_to_csv():
    logger.info("Exporting database to CSV...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, resume_text, job_desc, job_link, similarity, gemini_good, gemini_bad, gemini_recommendations, gemini_rating, gemini_match_score, gemini_comparison FROM resumes")
    rows = c.fetchall()
    conn.close()
    with open("resumes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Filename", "Resume_Content", "Job_Description", "Job_Link",
                         "Similarity", "Good_Points", "Bad_Points", "Recommendations",
                         "Rating", "Gemini_Match_Score", "Gemini_Comparison"])
        writer.writerows(rows)
    logger.info("CSV export complete.")

# ---------- GEMINI (text-only) ----------
def gemini_generate_from_text(prompt, model="gemini-2.5-flash"):
    if not GEMINI_ENABLED:
        logger.warning("Gemini not enabled — skipping API call.")
        return "[Gemini disabled: API key not configured]"
    try:
        logger.info("Calling Gemini (text prompt)...")
        # pass a single text prompt in contents (text only)
        response = client.models.generate_content(model=model, contents=[prompt])
        text = getattr(response, "text", "") or str(response)
        logger.debug("Gemini response length: %d", len(text))
        return text
    except Exception as e:
        logger.exception("Gemini API error")
        return f"[Gemini API error: {e}]"

def gemini_field_from_text(resume_text, job_desc_text, instruction_text, expect_json=False, truncate_resume=8000, truncate_jd=8000):
    rtxt = (resume_text or "")[:truncate_resume]
    jtxt = (job_desc_text or "")[:truncate_jd]
    prompt = f"""You are an expert resume screener.
Resume (candidate):
---
{rtxt}
---
Job description:
---
{jtxt}
---
Task: {instruction_text}

Return the requested output. If JSON is requested, return valid JSON only.
"""
    raw = gemini_generate_from_text(prompt)
    if expect_json:
        try:
            parsed = json.loads(raw)
            return parsed
        except Exception:
            m = re.search(r'(\{.*\})', raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    return parsed
                except Exception:
                    pass
            return {"raw": raw}
    else:
        return raw.strip()

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        logger.info("New resume upload request received.")
        job_desc = request.form.get("job_desc", "").strip()
        job_link = request.form.get("job_link", "").strip()
        resume_file = request.files.get("resume_file")
        resume_link = request.form.get("resume_link", "").strip()

        # If job link provided, scrape and use that text
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

        # Handle resume file upload or link
        if resume_file and resume_file.filename != "" and allowed_file(resume_file.filename):
            safe_name = secure_filename(resume_file.filename)
            filename = f"{resume_id}_{safe_name}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(filepath)
            logger.info(f"Saved uploaded resume to {filepath}")
        elif resume_link:
            if not (resume_link.lower().endswith(".pdf") or resume_link.lower().endswith(".docx")):
                flash("❌ Only .pdf or .docx resume links are allowed.")
                return redirect(request.url)
            try:
                response = requests.get(resume_link, timeout=15)
                response.raise_for_status()
                ext = resume_link.split(".")[-1].lower()
                filename = f"{resume_id}_link.{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded resume from link to {filepath}")
            except Exception as e:
                logger.exception("Error fetching resume link")
                flash(f"❌ Error fetching resume link: {e}")
                return redirect(request.url)
        else:
            flash("❌ Please upload a resume file or provide a resume link.")
            return redirect(request.url)

        # extract resume text
        resume_text = extract_text_from_resume(filepath).strip()
        if resume_text.startswith("[Error"):
            flash(f"❌ {resume_text}")
            return redirect(request.url)

        if exists_resume_text(resume_text):
            flash("⚠️ Duplicate resume detected (text already in DB). Skipping insert.")
            return redirect(request.url)

        # Compute local similarity
        sim_score = compute_similarity(resume_text, job_desc)

        # Insert initial DB row in processing state
        ok = save_initial_metadata(resume_id, filename, resume_text, job_desc, job_link or None, sim_score)
        if not ok:
            flash("⚠️ Duplicate detected or insert failed.")
            return redirect(request.url)

        # Steps definitions
        steps = [
            ("gemini_good", 
            "List the top 3 strengths of the resume for this job. Keep each point short and specific (max 12 words). Format as bullet points (•)."),

            ("gemini_bad", 
            "List up to 3 weaknesses or issues in the resume. Keep each point short and specific (max 12 words). Format as bullet points (•)."),

            ("gemini_recommendations", 
            "Suggest 3 very short, actionable improvements for this resume. Use concise bullet points (•), focused on making the resume stronger for the given job."),

            ("gemini_rating", 
            "Rate the resume for this job on a scale from 0–10. Return ONLY the number."),

            ("gemini_match_and_comparison", 
            "Evaluate how well the resume matches the job description. Return JSON only:\n"
            "{\n"
            "  \"match_score\": <0-100>,\n"
            "  \"comparison\": \"2–4 sentence summary comparing resume to job requirements\"\n"
            "}")
        ]


        try:
            # strengths
            update_db_columns(resume_id, status="processing", progress=10)
            strengths = gemini_field_from_text(resume_text, job_desc, steps[0][1], expect_json=False)
            update_db_columns(resume_id, gemini_good=str(strengths), progress=25)

            # weaknesses
            weaknesses = gemini_field_from_text(resume_text, job_desc, steps[1][1], expect_json=False)
            update_db_columns(resume_id, gemini_bad=str(weaknesses), progress=40)

            # recommendations
            recs = gemini_field_from_text(resume_text, job_desc, steps[2][1], expect_json=False)
            update_db_columns(resume_id, gemini_recommendations=str(recs), progress=55)

            # rating (expect single number)
            rating_text = gemini_field_from_text(resume_text, job_desc, steps[3][1], expect_json=False)
            rating_val = 0.0
            try:
                m = re.search(r"(\d+(\.\d+)?)", str(rating_text))
                if m:
                    rating_val = float(m.group(1))
            except Exception:
                rating_val = 0.0
            update_db_columns(resume_id, gemini_rating=rating_val, progress=70)

            # match score + comparison (expect json)
            match_json = gemini_field_from_text(resume_text, job_desc, steps[4][1], expect_json=True)
            match_score = 0.0
            comparison_text = ""
            if isinstance(match_json, dict):
                match_score = float(match_json.get("match_score") or match_json.get("match") or match_json.get("score") or 0)
                comparison_text = str(match_json.get("comparison") or match_json.get("summary") or match_json.get("raw") or "")
            else:
                try:
                    text = str(match_json)
                    m = re.search(r'(\d+(\.\d+)?)\s*%', text)
                    if m: match_score = float(m.group(1))
                    comparison_text = text
                except Exception:
                    comparison_text = str(match_json)

            # combine with local sim for final match
            if match_score and match_score > 0:
                combined_match = round((0.6 * float(match_score)) + (0.4 * float(sim_score)), 2)
            else:
                combined_match = sim_score

            update_db_columns(resume_id, gemini_match_score=combined_match, gemini_comparison=comparison_text, progress=95)
            update_db_columns(resume_id, status="done", progress=100)
            export_to_csv()
        except Exception as e:
            logger.exception("Error during Gemini sequential processing")
            update_db_columns(resume_id, status="failed", progress=0)
            flash("❌ Error while analyzing resume with Gemini. Check logs.")
            return redirect(url_for("report", resume_id=resume_id))

        flash(f"✅ Resume stored! Local similarity: {sim_score}%, Gemini Match: {combined_match}%, Rating: {rating_val}/10")
        return redirect(url_for("report", resume_id=resume_id))

    return render_template("upload.html")

@app.route("/report/<resume_id>")
def report(resume_id):
    logger.info(f"Fetching report for resume {resume_id}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, resume_text, job_desc, job_link, similarity, gemini_good, gemini_bad, gemini_recommendations, gemini_rating, gemini_match_score, gemini_comparison, status, progress FROM resumes WHERE id = ?", (resume_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        flash("Report not found.")
        return redirect(url_for("upload_resume"))

    (rid, filename, resume_text, job_desc, job_link, similarity,
     gemini_good, gemini_bad, gemini_recommendations, gemini_rating,
     gemini_match_score, gemini_comparison, status, progress) = row

    resume_snippet = (resume_text or "")[:600] + ("..." if len(resume_text or "") > 600 else "")
    job_snippet = (job_desc or "")[:1000] + ("..." if len(job_desc or "") > 1000 else "")

    # Provide clear placeholders when Gemini is disabled
    if not GEMINI_ENABLED:
        if not gemini_good: gemini_good = "[Gemini disabled]"
        if not gemini_bad: gemini_bad = "[Gemini disabled]"
        if not gemini_recommendations: gemini_recommendations = "[Gemini disabled]"
        if not gemini_comparison: gemini_comparison = "[Gemini disabled]"

    return render_template("report.html",
                           id=rid,
                           filename=filename,
                           resume_snippet=resume_snippet,
                           job_snippet=job_snippet,
                           job_link=job_link,
                           similarity=similarity,
                           gemini_good=gemini_good,
                           gemini_bad=gemini_bad,
                           gemini_recommendations=gemini_recommendations,
                           gemini_rating=gemini_rating,
                           gemini_match_score=gemini_match_score,
                           gemini_comparison=gemini_comparison,
                           status=status,
                           progress=progress)

@app.route("/status/<resume_id>")
def status(resume_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, similarity, gemini_good, gemini_bad, gemini_recommendations, gemini_rating, gemini_match_score, gemini_comparison, status, progress, job_link FROM resumes WHERE id = ?", (resume_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "not found"}), 404

    (rid, filename, similarity, gemini_good, gemini_bad, gemini_recommendations, gemini_rating, gemini_match_score, gemini_comparison, status, progress, job_link) = row
    return jsonify({
        "id": rid,
        "filename": filename,
        "similarity": similarity,
        "gemini_good": gemini_good,
        "gemini_bad": gemini_bad,
        "gemini_recommendations": gemini_recommendations,
        "gemini_rating": gemini_rating,
        "gemini_match_score": gemini_match_score,
        "gemini_comparison": gemini_comparison,
        "status": status,
        "progress": progress,
        "job_link": job_link
    })

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"content": "", "error": "No URL provided"}), 400
    content = scrape_text_from_url(url)
    return jsonify({"content": content})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(debug=True)
