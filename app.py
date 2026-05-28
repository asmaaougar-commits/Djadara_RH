import re
import unicodedata
import os
from datetime import datetime
from functools import wraps
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# ── OCR & PDF ──────────────────────────────────────────────────────────────
import fitz
import pytesseract
from PIL import Image
import io
import shutil

tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    for path in ['/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract', '/usr/bin/tesseract']:
        if os.path.exists(path):
            tesseract_path = path
            break
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Tesseract trouvé : {tesseract_path}")
else:
    print("⚠️ Tesseract introuvable — OCR désactivé.")

# ── APPLICATION CONFIGURATION ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "recrutai-ultra-secret-2026")

UPLOAD_AVATAR_FOLDER = os.path.join('static', 'uploads', 'avatars')
os.makedirs(UPLOAD_AVATAR_FOLDER, exist_ok=True)

# ── DATABASE CONFIGURATION ─────────────────────────────────────────────────
DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_NAME     = os.environ.get("DB_NAME",     "yassir_rh")
DB_USER     = os.environ.get("DB_USER",     "asma")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_PORT     = os.environ.get("DB_PORT",     "5432")

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        nom VARCHAR(100) NOT NULL,
        entreprise VARCHAR(150),
        email VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        avatar VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS offres (
        id SERIAL PRIMARY KEY,
        titre VARCHAR(255) NOT NULL,
        description TEXT,
        competences TEXT NOT NULL,
        experience VARCHAR(100),
        formation VARCHAR(150),
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cvs (
        id SERIAL PRIMARY KEY,
        nom_fichier VARCHAR(255) NOT NULL,
        contenu TEXT,
        score SMALLINT CHECK (score >= 0 AND score <= 100),
        offre_id INTEGER REFERENCES offres(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")
    cur.execute("SELECT id FROM users WHERE email = %s", ("recruteur@ummto.dz",))
    if not cur.fetchone():
        hashed_pwd = generate_password_hash("tizi2026")
        cur.execute(
            "INSERT INTO users (nom, entreprise, email, password) VALUES (%s, %s, %s, %s)",
            ("Recruteur UMMTO", "UMMTO", "recruteur@ummto.dz", hashed_pwd)
        )
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Base de données initialisée.")

try:
    init_db()
except Exception as e:
    print(f"⚠️ Erreur d'initialisation DB : {e}")

# ── LOGIN DECORATOR ────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authed'):
            flash("Veuillez vous connecter.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ── NLP & TEXT PROCESSING ──────────────────────────────────────────────────
def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s@\.]', ' ', text)
    return " ".join(text.split())

def extract_text_from_pdf(stream_bytes):
    text = ""
    try:
        doc = fitz.open(stream=stream_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        pass
    if len(clean_text(text)) < 50 and tesseract_path:
        try:
            text = ""
            doc = fitz.open(stream=stream_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img, lang="fra+eng")
            doc.close()
        except Exception:
            pass
    return text

def compute_knn_rank(scores_arr):
    if len(scores_arr) < 2:
        return np.zeros_like(scores_arr)
    distances = np.abs(1.0 - scores_arr)
    max_d = np.max(distances) if np.max(distances) > 0 else 1.0
    return 1.0 - (distances / max_d)

# ── ROUTES ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close(); conn.close()
        if user and check_password_hash(user['password'], password):
            session['authed'] = True
            session['user_id'] = user['id']
            session['nom'] = user['nom']
            session['email'] = user['email']
            return redirect(url_for('dashboard'))
        else:
            flash("Identifiants incorrects.", "error")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nom = request.form.get('nom', '').strip()
        entreprise = request.form.get('entreprise', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        avatar_file = request.files.get('avatar')
        if not nom or not email or not password:
            flash("Tous les champs obligatoires doivent être remplis.", "error")
            return render_template('register.html')
        avatar_filename = None
        if avatar_file and avatar_file.filename != '':
            ext = os.path.splitext(avatar_file.filename)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                avatar_filename = f"avatar_{int(datetime.now().timestamp())}{ext}"
                upload_dir = os.path.join(app.root_path, 'static', 'uploads', 'avatars')
                os.makedirs(upload_dir, exist_ok=True)
                avatar_file.save(os.path.join(upload_dir, avatar_filename))
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute(
                "INSERT INTO users (nom, entreprise, email, password, avatar) VALUES (%s, %s, %s, %s, %s) RETURNING id;",
                (nom, entreprise, email, generate_password_hash(password), avatar_filename)
            )
            new_user = cur.fetchone()
            conn.commit()
            session['authed'] = True
            session['user_id'] = new_user['id']
            session['nom'] = nom
            session['email'] = email
            return redirect(url_for('dashboard'))
        except psycopg2.IntegrityError:
            conn.rollback()
            flash("Cet email est déjà enregistré.", "error")
        except Exception as e:
            conn.rollback()
            flash(f"Erreur : {str(e)}", "error")
        finally:
            cur.close(); conn.close()
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM offres WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
    offres = [dict(r) for r in cur.fetchall()]
    cur.execute("""
        SELECT cvs.*, offres.titre as offre_titre
        FROM cvs
        JOIN offres ON cvs.offre_id = offres.id
        WHERE offres.user_id = %s
        ORDER BY cvs.created_at DESC
        LIMIT 5
    """, (session['user_id'],))
    candidats_recents = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return render_template('dashboard.html', offres=offres, candidats_recents=candidats_recents)

# ── OFFRES D'EMPLOI ────────────────────────────────────────────────────────
@app.route('/offres')
@login_required
def offres():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM offres WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
    offres_list = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return render_template('offres.html', offres=offres_list)

@app.route('/creer-offre', methods=['POST'])
@login_required
def creer_offre():
    titre = request.form.get('titre', '').strip()
    competences = request.form.get('competences', '').strip()
    description = request.form.get('description', '').strip()
    experience = request.form.get('experience', '').strip()
    if not titre or not competences:
        flash("Le titre et les compétences sont obligatoires.", "error")
        return redirect(request.referrer or url_for('offres'))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO offres (titre, description, competences, experience, user_id) VALUES (%s, %s, %s, %s, %s)",
        (titre, description, competences, experience, session['user_id'])
    )
    conn.commit(); cur.close(); conn.close()
    flash("Offre publiée avec succès !", "success")
    return redirect(url_for('offres'))

# ── UPLOAD CV ──────────────────────────────────────────────────────────────
@app.route('/upload-cv')
@login_required
def upload_cv():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM offres WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
    offres = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return render_template('upload_cv.html', offres=offres)

# ── CANDIDATS ──────────────────────────────────────────────────────────────
@app.route('/candidats')
@login_required
def candidats():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT cvs.* FROM cvs
        JOIN offres ON cvs.offre_id = offres.id
        WHERE offres.user_id = %s
        ORDER BY cvs.created_at DESC
    """, (session['user_id'],))
    candidats_list = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return render_template('candidats.html', candidats=candidats_list)

@app.route('/candidats/supprimer/<int:cv_id>', methods=['POST'])
@login_required
def supprimer_candidat(cv_id):
    conn = get_db_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    # Vérifier que le CV appartient bien à l'utilisateur connecté
    cur.execute("""
        SELECT cvs.id FROM cvs
        JOIN offres ON cvs.offre_id = offres.id
        WHERE cvs.id = %s AND offres.user_id = %s
    """, (cv_id, session['user_id']))
    row = cur.fetchone()
    if row:
        cur.execute("DELETE FROM cvs WHERE id = %s", (cv_id,))
        conn.commit()
        flash("Candidat supprimé avec succès.", "success")
    else:
        flash("Candidat introuvable ou accès refusé.", "error")
    cur.close(); conn.close()
    return redirect(url_for('candidats'))

# ── MATCHING IA ────────────────────────────────────────────────────────────
@app.route('/matching')
@login_required
def matching():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM offres WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
    offres = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return render_template('matching.html', offres=offres)

# Garde la compatibilité avec /analyse
@app.route('/analyse')
@login_required
def analyse():
    return redirect(url_for('matching'))

# ── ANALYSE DIMENSIONS CV ─────────────────────────────────────────────────

def extract_candidate_name(raw_text, filename):
    """
    Tente d'extraire le nom du candidat depuis le texte du CV.
    Stratégies par ordre de priorité :
    1. Ligne contenant "Nom :" ou "Name :" dans le CV
    2. Premières lignes du CV (souvent le nom est en haut)
    3. Nom du fichier PDF nettoyé comme fallback
    """
    # Stratégie 1 : pattern "Nom : Prénom NOM" ou "Name: ..."
    patterns = [
        r'(?:nom\s*[:\-]\s*)([A-ZÀ-Ÿa-zà-ÿ]+(?:\s+[A-ZÀ-Ÿa-zà-ÿ]+){1,3})',
        r'(?:name\s*[:\-]\s*)([A-Za-z]+(?:\s+[A-Za-z]+){1,3})',
        r'(?:prénom\s*[:\-]\s*)([A-ZÀ-Ÿa-zà-ÿ]+(?:\s+[A-ZÀ-Ÿa-zà-ÿ]+){0,2})',
    ]
    for pat in patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            if 3 < len(name) < 50:
                return name

    # Stratégie 2 : premières lignes non vides (le nom est souvent en haut du CV)
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    for line in lines[:6]:
        # Ligne courte (2-5 mots), que des lettres/espaces → probablement un nom
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.match(r'^[A-ZÀ-Ÿa-zà-ÿ\-]+$', w) for w in words):
            # Exclure les mots génériques
            generics = {'curriculum', 'vitae', 'cv', 'resume', 'profil', 'contact', 'coordonnees'}
            if not any(w.lower() in generics for w in words):
                return line.strip().title()

    # Stratégie 3 : fallback sur le nom du fichier
    name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    # Retirer les chiffres et underscores en trop
    name = re.sub(r'\d+', '', name).strip().title()
    return name if name else "Candidat inconnu"

def score_competences(cleaned_text, keywords):
    """Score basé sur les mots-clés de l'offre trouvés dans le CV."""
    if not keywords:
        return 0, [], []
    found     = [kw for kw in keywords if kw in cleaned_text]
    missing   = [kw for kw in keywords if kw not in cleaned_text]
    score     = int(len(found) / len(keywords) * 100)
    return score, found, missing

def score_experience(raw_text):
    """
    Détecte les années d'expérience et les mentions de postes/stages.
    Retourne un score /100 et une liste de détails trouvés.
    """
    text_lower = raw_text.lower()

    # Patterns : "3 ans d'expérience", "5 years", "2 ans", "depuis 2019"...
    exp_patterns = [
        r'(\d+)\s*ans?\s+d[\'\u2019]exp[eé]rience',
        r'(\d+)\s*years?\s+of\s+experience',
        r'(\d+)\s*ans?\s+d[\'\u2019]anciennet[eé]',
        r'exp[eé]rience\s*[:\-]?\s*(\d+)\s*ans?',
    ]
    years_found = []
    for pat in exp_patterns:
        for m in re.finditer(pat, text_lower):
            years_found.append(int(m.group(1)))

    # Mots-clés de postes professionnels
    job_keywords = [
        'ingénieur', 'développeur', 'analyste', 'consultant', 'chef de projet',
        'technicien', 'responsable', 'manager', 'directeur', 'coordinateur',
        'engineer', 'developer', 'analyst', 'architect', 'lead', 'senior',
        'stagiaire', 'stage', 'alternance', 'apprentissage', 'intern',
        'cdi', 'cdd', 'freelance', 'mission',
    ]
    job_hits = sum(1 for kw in job_keywords if kw in text_lower)

    # Dates de travail : "jan 2020 – déc 2022", "2018-2021"
    date_ranges = re.findall(
        r'(jan|fév|mar|avr|mai|jun|jul|aoû|sep|oct|nov|déc|january|february|march|april|'
        r'june|july|august|september|october|november|december)[\s\.\-]*\d{4}',
        text_lower
    )

    # Calcul du score expérience
    max_years = max(years_found) if years_found else 0
    details   = []

    if max_years >= 5:
        exp_score = 90
        details.append(f"{max_years} ans d'expérience détectés")
    elif max_years >= 3:
        exp_score = 70
        details.append(f"{max_years} ans d'expérience détectés")
    elif max_years >= 1:
        exp_score = 50
        details.append(f"{max_years} an(s) d'expérience détecté(s)")
    elif max_years == 0 and (job_hits >= 3 or len(date_ranges) >= 2):
        exp_score = 40
        details.append("Postes/missions mentionnés sans durée explicite")
    elif job_hits >= 1 or len(date_ranges) >= 1:
        exp_score = 25
        details.append("Mentions professionnelles limitées")
    else:
        exp_score = 10
        details.append("Aucune expérience professionnelle détectée")

    if 'stage' in text_lower or 'stagiaire' in text_lower or 'intern' in text_lower:
        details.append("Stage(s) mentionné(s)")
        exp_score = max(exp_score, 30)

    return min(exp_score, 100), details

def score_formation(raw_text):
    """
    Détecte les diplômes et leur niveau.
    Retourne un score /100 et une liste de diplômes trouvés.
    """
    text_lower = raw_text.lower()

    diplomes = {
        # Niveau doctorat → 100
        'doctorat': 100, 'phd': 100, 'ph.d': 100, 'thèse': 100,
        # Master → 90
        'master': 90, 'mastère': 90, 'm2': 90, 'mba': 90,
        'diplôme d\'ingénieur': 90, "diplome d'ingenieur": 90,
        'ingénieur': 85,
        # Licence / Bachelor → 70
        'licence': 70, 'bachelor': 70, 'l3': 70, 'bsc': 70,
        # BTS / DUT / BUT → 55
        'bts': 55, 'dut': 55, 'but': 55, 'hnd': 55,
        # Bac → 40
        'baccalauréat': 40, 'bac': 40, 'lycée': 35,
        # Certificat / Formation → 30
        'certification': 30, 'certificat': 30, 'diplôme': 30,
        'formation': 25,
    }

    best_score = 0
    found_diplomes = []
    for diplome, val in diplomes.items():
        if diplome in text_lower and val > best_score:
            best_score = val
            found_diplomes.append(diplome.title())

    # Universités / grandes écoles
    ecoles = [
        'université', 'university', 'école nationale', 'ecole nationale',
        'école polytechnique', 'ecole polytechnique', 'ens', 'insa',
        'ummto', 'usthb', 'usto', 'enp', 'esi', 'hmoud',
        'sorbonne', 'harvard', 'mit', 'stanford',
    ]
    ecole_hits = [e for e in ecoles if e in text_lower]
    if ecole_hits and best_score < 50:
        best_score = max(best_score, 50)
        found_diplomes += [e.title() for e in ecole_hits[:2]]

    if best_score == 0:
        return 10, ["Aucune formation détectée"]

    return min(best_score, 100), list(dict.fromkeys(found_diplomes))[:4]

def score_langues(raw_text):
    """
    Détecte les langues et leur niveau.
    Retourne un score /100 et liste des langues trouvées.
    """
    text_lower = raw_text.lower()

    langues_map = {
        'français': 'Français', 'french': 'Français', 'francais': 'Français',
        'anglais': 'Anglais',   'english': 'Anglais',
        'arabe': 'Arabe',       'arabic': 'Arabe',
        'espagnol': 'Espagnol', 'spanish': 'Espagnol',
        'allemand': 'Allemand', 'german': 'Allemand',
        'italien': 'Italien',   'italian': 'Italien',
        'chinois': 'Chinois',   'chinese': 'Chinois',
        'portugais': 'Portugais', 'portuguese': 'Portugais',
        'russe': 'Russe',       'russian': 'Russe',
    }
    niveaux_bonus = {
        'courant': 15, 'bilingue': 20, 'natif': 20, 'native': 20,
        'fluent': 15, 'professionnel': 12, 'professional': 12,
        'b2': 10, 'c1': 15, 'c2': 20,
        'intermédiaire': 5, 'notions': 0, 'débutant': 0,
    }

    found = []
    for key, label in langues_map.items():
        if key in text_lower and label not in found:
            found.append(label)

    if not found:
        return 10, []

    # Score de base : nombre de langues
    base = min(len(found) * 20, 60)

    # Bonus niveaux
    bonus = sum(v for k, v in niveaux_bonus.items() if k in text_lower)
    bonus = min(bonus, 40)

    return min(base + bonus, 100), found

def build_points_forts(found_kws, exp_details, diplomes, langues, scores):
    points = []
    if scores['competences'] >= 60:
        points.append(f"Maîtrise de {len(found_kws)} compétence(s) requise(s) : {', '.join(found_kws[:4])}.")
    if scores['experience'] >= 50:
        points.append(exp_details[0] if exp_details else "Expérience professionnelle solide.")
    if scores['formation'] >= 70:
        points.append(f"Formation de niveau élevé : {', '.join(diplomes[:2])}.")
    if scores['langues'] >= 40 and langues:
        points.append(f"Maîtrise linguistique : {', '.join(langues)}.")
    if not points:
        points.append("Profil avec des bases à développer.")
    return points

def build_points_ameliorer(missing_kws, exp_details, scores):
    points = []
    if scores['competences'] < 80 and missing_kws:
        points.append(f"Compétences manquantes : {', '.join(missing_kws[:3])}.")
    if scores['experience'] < 50:
        points.append("Expérience professionnelle insuffisante ou non détaillée.")
    if scores['formation'] < 50:
        points.append("Formation académique peu ou mal renseignée.")
    if scores['langues'] < 40:
        points.append("Peu de langues mentionnées ou niveaux non précisés.")
    if not points:
        points.append("Continuer à développer une veille technologique active.")
    return points

@app.route('/match', methods=['POST'])
@login_required
def match():
    offre_id = request.form.get('offre_id')
    files    = request.files.getlist('cv_files')
    if not offre_id or not files or files[0].filename == '':
        flash("Veuillez sélectionner une offre et ajouter au moins un fichier PDF.", "error")
        return redirect(url_for('matching'))

    conn = get_db_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM offres WHERE id = %s AND user_id = %s", (offre_id, session['user_id']))
    offer_row = cur.fetchone()
    offer = dict(offer_row) if offer_row else None
    if not offer:
        cur.close(); conn.close()
        flash("Offre introuvable.", "error")
        return redirect(url_for('matching'))

    try:
        # Mots-clés de l'offre
        keywords = [clean_text(k) for k in offer['competences'].split(',') if clean_text(k)]

        # Niveau d'expérience requis par l'offre (ex: "3 ans", "junior", "senior")
        exp_requise = clean_text(offer.get('experience') or '')

        candidats = []
        for f in files:
            if not f.filename.lower().endswith('.pdf'):
                continue
            stream   = f.read()
            raw_text = extract_text_from_pdf(stream)
            cleaned  = clean_text(raw_text)
            if not cleaned:
                continue

            # ── Scores par dimension ──────────────────────────────
            s_comp,  found_kws, missing_kws = score_competences(cleaned, keywords)
            s_exp,   exp_details            = score_experience(raw_text)
            s_form,  diplomes               = score_formation(raw_text)
            s_lang,  langues                = score_langues(raw_text)

            # ── Score global pondéré ──────────────────────────────
            # Compétences 40% | Expérience 30% | Formation 20% | Langues 10%
            global_score = int(
                s_comp * 0.40 +
                s_exp  * 0.30 +
                s_form * 0.20 +
                s_lang * 0.10
            )
            global_score = min(100, max(0, global_score))

            # ── Points forts / à améliorer ────────────────────────
            dim_scores = {
                'competences': s_comp,
                'experience':  s_exp,
                'formation':   s_form,
                'langues':     s_lang,
            }
            points_forts    = build_points_forts(found_kws, exp_details, diplomes, langues, dim_scores)
            points_ameliorer = build_points_ameliorer(missing_kws, exp_details, dim_scores)

            nom_candidat = extract_candidate_name(raw_text, f.filename)
            candidats.append({
                "fichier":          secure_filename(f.filename),
                "nom_candidat":     nom_candidat,
                "texte_brut":       raw_text,
                "global_score":     global_score,
                "score_competences": s_comp,
                "score_experience":  s_exp,
                "score_formation":   s_form,
                "score_langues":     s_lang,
                "competences_trouvees": [kw.title() for kw in found_kws],
                "competences_manquantes": [kw.title() for kw in missing_kws],
                "exp_details":      exp_details,
                "diplomes":         diplomes,
                "langues":          langues,
                "points_forts":     points_forts,
                "points_ameliorer": points_ameliorer,
                "niveau":           "À évaluer",
            })

        if not candidats:
            cur.close(); conn.close()
            flash("Aucun texte extrait des PDF fournis.", "error")
            return redirect(url_for('matching'))

        # ── KNN boost si plusieurs candidats ─────────────────────
        if len(candidats) >= 2:
            raw_scores = np.array([c["global_score"] / 100.0 for c in candidats])
            boost = compute_knn_rank(raw_scores)
            for i, c in enumerate(candidats):
                c["global_score"] = min(100, max(0, int(
                    (c["global_score"] / 100 * 0.85 + boost[i] * 0.15) * 100
                )))

        # ── Tri et rang ───────────────────────────────────────────
        candidats.sort(key=lambda x: x["global_score"], reverse=True)
        for rank, c in enumerate(candidats, 1):
            c["rang"] = rank
            s = c["global_score"]
            c["niveau"] = (
                "Excellent"   if s >= 75 else
                "Bon profil"  if s >= 50 else
                "Partiel"     if s >= 30 else
                "Insuffisant"
            )
            cur.execute(
                "INSERT INTO cvs (nom_fichier, contenu, score, offre_id) VALUES (%s, %s, %s, %s)",
                (c["fichier"], c["texte_brut"], c["global_score"], offre_id)
            )

        conn.commit()
        return render_template("resultats.html", offer=offer, candidats=candidats)

    except Exception as e:
        conn.rollback()
        flash(f"Erreur lors du matching : {str(e)}", "error")
        return redirect(url_for('matching'))
    finally:
        cur.close(); conn.close()

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)