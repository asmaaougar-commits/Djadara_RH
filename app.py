"""
╔══════════════════════════════════════════════════════════════╗
║  Djadara RH  —  app.py  v4.1                                ║
║  Plateforme intelligente de recrutement                      ║
║                                                              ║
║  3 rôles :                                                   ║
║    • Candidat   → inscription, CV, candidatures, suivi       ║
║    • Entreprise → offres, matching IA, entretiens            ║
║    • Admin      → connexion seule, gestion globale + stats   ║
║                                                              ║
║  ✅ Sans SpaCy — matching par regex multi-couche             ║
╚══════════════════════════════════════════════════════════════╝
"""
import re
import unicodedata
import os
import json
import shutil
import secrets
from datetime import datetime, timedelta

from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# ── PDF & OCR ─────────────────────────────────────────────────
import fitz          # pymupdf
import pytesseract
from PIL import Image
import io
from pypdf import PdfReader

# ── NLP & ML ──────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ── Tesseract auto-détection ───────────────────────────────────
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    for path in ['/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract']:
        if os.path.exists(path):
            tesseract_path = path
            break
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Tesseract : {tesseract_path}")
else:
    print("⚠️  Tesseract introuvable — OCR désactivé")


# ══════════════════════════════════════════════════════════════
#   CONFIGURATION FLASK
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "djadara_rh_secret_key_2026")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# ── Flask-Mail (Gmail) ─────────────────────────────────────────
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD']       = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)
print("MAIL_USERNAME =", app.config['MAIL_USERNAME'])
print("MAIL_PASSWORD =", "OK" if app.config['MAIL_PASSWORD'] else "VIDE")

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ══════════════════════════════════════════════════════════════
#   BASE DE DONNÉES
# ══════════════════════════════════════════════════════════════
def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        database=os.environ.get("DB_NAME", "djadara_rh"),
        user=os.environ.get("DB_USER", "asma"),
        password=os.environ.get("DB_PASSWORD", ""),
        cursor_factory=RealDictCursor
    )


# ══════════════════════════════════════════════════════════════
#   NLP UTILITIES
# ══════════════════════════════════════════════════════════════
def normalize(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_pdf_text(file_path: str) -> str:
    text = ""

    # Stratégie 1 : pypdf (PDF texte normal)
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        print(f"[PDF] pypdf → {len(text)} caractères")
    except Exception as e:
        print(f"[PDF] pypdf error: {e}")

    # Stratégie 2 : OCR si texte insuffisant (< 200 car.)
    if len(text) < 200 and tesseract_path:
        try:
            doc = fitz.open(file_path)
            ocr_pages = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                img = img.convert("L")
                ocr_pages.append(
                    pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")
                )
            doc.close()
            ocr_text = "\n".join(ocr_pages).strip()
            print(f"[PDF] OCR → {len(ocr_text)} caractères")
            if len(ocr_text) > len(text):
                text = ocr_text
        except Exception as e:
            print(f"[PDF] OCR error: {e}")

    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    print(f"[PDF] Final → {len(text)} car. | aperçu : {repr(text[:100])}")
    return text


def extraire_nom_candidat(texte: str, nom_fichier: str) -> str:
    lignes = [l.strip() for l in texte.split('\n') if l.strip()]
    for ligne in lignes[:8]:
        mots = ligne.split()
        if 2 <= len(mots) <= 4 and all(m[0].isupper() for m in mots if m.isalpha()):
            if not any(kw in ligne.lower() for kw in
                       ['curriculum', 'vitae', 'cv', 'résumé', 'resume', '@', 'tel', 'email', 'www']):
                return ligne.title()
    return nom_fichier.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()


# ── TF-IDF cosine ──────────────────────────────────────────────
def compute_tfidf_cosine(texts: list, query: str) -> np.ndarray:
    corpus = texts + [query]
    normalized = [normalize(t) for t in corpus]
    if not any(normalized):
        return np.zeros(len(texts))
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        matrix = vec.fit_transform(normalized)
        return cosine_similarity(matrix[-1], matrix[:-1])[0]
    except ValueError:
        return np.zeros(len(texts))


# ── KNN boost ──────────────────────────────────────────────────
def compute_knn_boost(scores: np.ndarray, k: int = 5) -> np.ndarray:
    if len(scores) < 2:
        return np.ones(len(scores))
    X = scores.reshape(-1, 1)
    k_eff = min(k, len(X))
    knn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    mean_dist = distances.mean(axis=1)
    return 1.0 - (mean_dist / (mean_dist.max() + 1e-9))


PATTERNS_EXPERIENCE = [
    r'\b(\d+)\s*an[s]?\s*(d\'|de\s*)?(expérience|exp\.?)',
    r'\b(stage|stagiaire|intern)\b',
    r'\b(chef de projet|lead|manager|directeur|responsable)\b',
    r'\b(cdd|cdi|freelance|consultant)\b',
    r'\b(\d{4})\s*[-–]\s*(\d{4}|\bprésent\b|\bactuel\b)',
]
PATTERNS_FORMATION = [
    r'\b(master|licence|bachelor|bts|dut|bac\+\d|ingénieur|doctorat|phd|mba)\b',
    r'\b(université|ecole|école|faculté|institut|ensa|esi|usthb|enp)\b',
    r'\b(diplômé|diplôme|formation|certifi)\b',
]
PATTERNS_LANGUES = [
    r'\b(français|anglais|arabe|espagnol|allemand|italien)\b',
    r'\b(french|english|arabic|spanish|german)\b',
    r'\b(bilingue|trilingue|natif|courant|intermédiaire)\b',
    r'\b(toeic|toefl|ielts|delf|dalf)\b',
    r'\b(b[12]|c[12]|a[12])\b',
]

# FIX 1 — Patterns pour les certifications (manquaient complètement)
PATTERNS_CERTIFICATIONS = [
    r'\b(aws|azure|gcp|google cloud)\b',
    r'\b(cisco|ccna|ccnp|comptia|security\+|network\+)\b',
    r'\b(pmp|prince2|scrum|agile|safe)\b',
    r'\b(oracle|microsoft certified|mcsa|mcse)\b',
    r'\b(certification|certifié|certificat|badge)\b',
    r'\b(itil|iso|lean|six sigma)\b',
]


def _pattern_score(texte: str, patterns: list) -> int:
    t = texte.lower()
    hits = sum(1 for p in patterns if re.search(p, t))
    return min(int((hits / len(patterns)) * 100 * 1.8), 95)


# ══════════════════════════════════════════════════════════════
#   MATCHING — DÉTECTION DES COMPÉTENCES (sans SpaCy)
#   Fonctionne avec SQL, Python, C++, Node.js, etc.
# ══════════════════════════════════════════════════════════════
def _kw_in_text(kw: str, texte_low: str) -> bool:
    """
    Détecte si un mot-clé est présent dans le texte du CV.
    Stratégie en 2 couches (sans SpaCy) :
      1. Recherche directe entourée de séparateurs → SQL, C++, Node.js...
      2. Variantes morphologiques avec \b → pluriel, tiret/espace, version collée
    """
    kw_low = kw.lower()

    # Couche 1 : entouré de séparateurs courants (espace, virgule, saut de ligne…)
    sep = r'[\s,;:/\.\(\)\[\]\-\+\#\|\n\r]'
    pat_direct = r'(?:^|' + sep + r')' + re.escape(kw_low) + r'(?=$|' + sep + r'|s' + sep + r'|s$)'
    if re.search(pat_direct, texte_low):
        return True

    # Couche 2 : \b sur mots alphanumérique (Python, Django, machine learning…)
    if re.match(r'^[\w\s\-]+$', kw_low):
        variantes = [
            re.escape(kw_low),
            re.escape(kw_low + 's'),
            re.escape(kw_low.replace(' ', '-')),
            re.escape(kw_low.replace('-', ' ')),
            re.escape(kw_low.replace(' ', '')),   # "machine learning" → "machinelearning"
        ]
        pat_word = r'(?i)\b(' + '|'.join(variantes) + r')\b'
        if re.search(pat_word, texte_low):
            return True

    return False


def analyser_competences(texte: str, competences_str: str):
    """
    Compare les compétences de l'offre avec le texte du CV.
    Retourne : (score, liste_trouvées, liste_manquantes)
    """
    keywords = [k.strip() for k in competences_str.split(',') if k.strip()]
    if not keywords:
        return 0, [], []

    texte_low = texte.lower()
    trouvees, manquantes = [], []

    for kw in keywords:
        if _kw_in_text(kw, texte_low):
            trouvees.append(kw.title())
        else:
            manquantes.append(kw.title())

    ratio = len(trouvees) / len(keywords)
    # FIX 2 — Avant : score plafonné à 95 (ratio*85 + 10 max).
    # Maintenant : score linéaire sur 100, bonus +5 si >= 80% trouvées.
    score = int(ratio * 100)
    if ratio >= 0.8:
        score = min(score + 5, 100)
    return score, trouvees, manquantes


def generer_points(trouvees, manquantes, score_exp, score_form, score_lang):
    forts, ameliorer = [], []
    if trouvees:
        forts.append(f"Maîtrise de {', '.join(trouvees[:3])}" + (" et plus" if len(trouvees) > 3 else ""))
    if score_form >= 60:
        forts.append("Formation académique solide et adaptée au poste")
    if score_exp >= 60:
        forts.append("Expérience professionnelle significative")
    if score_lang >= 70:
        forts.append("Bonne maîtrise des langues")
    if manquantes:
        ameliorer.append(f"Compétences manquantes : {', '.join(manquantes[:3])}" + (" et autres" if len(manquantes) > 3 else ""))
    if score_exp < 40:
        ameliorer.append("Expérience professionnelle limitée ou non détaillée")
    if score_form < 40:
        ameliorer.append("Formation non précisée ou peu en lien avec le poste")
    if score_lang < 40:
        ameliorer.append("Compétences linguistiques non mentionnées")
    if not forts:
        forts = ["Candidature soumise à évaluation"]
    if not ameliorer:
        ameliorer = ["Continuer à enrichir le CV avec des projets concrets"]
    return forts[:4], ameliorer[:4]


def evaluer_cv_complet(texte: str, offre: dict, tfidf_score: float) -> dict:
    comp_str = offre.get('competences', '') or ''

    score_comp, trouvees, manquantes = analyser_competences(texte, comp_str)
    score_exp   = _pattern_score(texte, PATTERNS_EXPERIENCE)
    score_form  = _pattern_score(texte, PATTERNS_FORMATION)
    score_lang  = _pattern_score(texte, PATTERNS_LANGUES)
    # FIX 3 — score_certifications était toujours 0 (jamais calculé)
    score_certif = _pattern_score(texte, PATTERNS_CERTIFICATIONS)

    formation_cible = offre.get('niveau_etude', '') or ''
    if formation_cible:
        for mot in re.split(r'[,\s]+', formation_cible.lower()):
            if mot and len(mot) > 3 and mot in texte.lower():
                score_form = min(score_form + 15, 100)
                break

    keywords = [k.strip() for k in comp_str.split(',') if k.strip()]
    overlap = len(trouvees) / max(len(keywords), 1)

    # FIX 4 — Les poids bruts de la DB ne somment pas forcément à 100.
    # On les normalise pour que leur somme soit toujours égale à 1,
    # ce qui garantit que le score_global est bien dans [0, 100].
    p_comp   = offre.get('poids_competences',  40)
    p_exp    = offre.get('poids_experience',   25)
    p_form   = offre.get('poids_formation',    20)
    p_lang   = offre.get('poids_langues',      10)
    p_certif = offre.get('poids_certif',        5)

    total = p_comp + p_exp + p_form + p_lang + p_certif
    if total == 0:
        total = 100  # sécurité anti-division par zéro

    # Score de compétences = 87.5% TF-IDF sémantique + 12.5% overlap exact
    score_comp_final = tfidf_score * 100 * 0.875 + overlap * 100 * 0.125

    score_global = int(
        score_comp_final * (p_comp   / total) +
        score_exp        * (p_exp    / total) +
        score_form       * (p_form   / total) +
        score_lang       * (p_lang   / total) +
        score_certif     * (p_certif / total)
    )
    score_global = max(0, min(score_global, 100))

    if score_global >= 75:   niveau = "Excellent"
    elif score_global >= 60: niveau = "Bon profil"
    elif score_global >= 40: niveau = "Partiel"
    else:                    niveau = "Insuffisant"

    forts, ameliorer = generer_points(trouvees, manquantes, score_exp, score_form, score_lang)

    return {
        'score':                  score_global,
        'score_competences':      score_comp,
        'score_experience':       score_exp,
        'score_formation':        score_form,
        'score_langues':          score_lang,
        'score_certifications':   score_certif,   # FIX 3 — maintenant renseigné
        'competences_trouvees':   json.dumps(trouvees, ensure_ascii=False),
        'competences_manquantes': json.dumps(manquantes, ensure_ascii=False),
        'points_forts':           json.dumps(forts, ensure_ascii=False),
        'points_ameliorer':       json.dumps(ameliorer, ensure_ascii=False),
        'niveau':                 niveau,
    }


# ══════════════════════════════════════════════════════════════
#   HELPERS AUTH
# ══════════════════════════════════════════════════════════════
def candidat_auth_required():
    if 'candidat_id' not in session:
        flash("Veuillez vous connecter.", "warning")
        return redirect(url_for('login') + '?role=candidat')
    return None


def entreprise_auth_required():
    if 'entreprise_id' not in session:
        flash("Veuillez vous connecter.", "warning")
        return redirect(url_for('login') + '?role=entreprise')
    return None


def admin_auth_required():
    if 'admin' not in session:
        flash("Accès réservé à l'administrateur.", "warning")
        return redirect(url_for('admin_login'))
    return None


# ══════════════════════════════════════════════════════════════
#   ROUTES PUBLIQUES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


# ── REGISTER (Candidat + Entreprise) ──────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        role = request.form.get('role', 'candidat')

        if role not in ('candidat', 'entreprise'):
            flash("Rôle invalide.", "danger")
            return redirect(url_for('register'))

        email    = request.form.get('email', '').strip().lower()
        password = generate_password_hash(request.form.get('password'))
        code     = str(secrets.randbelow(900000) + 100000)

        conn = get_db_connection()
        cur  = conn.cursor()
        try:
            if role == 'entreprise':
                nom_entreprise = request.form.get('nom_entreprise', '').strip()
                secteur   = request.form.get('secteur', '').strip()
                adresse   = request.form.get('adresse', '').strip()
                telephone = request.form.get('telephone', '').strip()
                cur.execute("""
                    INSERT INTO entreprises
                      (nom, secteur, adresse, email, telephone, password, verif_token, is_verified)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE);
                """, (nom_entreprise, secteur, adresse, email, telephone, password, code))
                prenom_affichage = nom_entreprise
                nom_affichage    = ''
            else:
                nom    = request.form.get('nom', '').strip()
                prenom = request.form.get('prenom', '').strip()
                cur.execute("""
                    INSERT INTO candidats
                      (nom, prenom, email, password, verif_token, is_verified)
                    VALUES (%s, %s, %s, %s, %s, FALSE);
                """, (nom, prenom, email, password, code))
                prenom_affichage = prenom
                nom_affichage    = nom

            conn.commit()

            msg = Message("Votre code de vérification — Djadara RH", recipients=[email])
            msg.body = (
                f"Bonjour {prenom_affichage} {nom_affichage},\n\n"
                f"Votre code de vérification est :\n\n"
                f"  {code}\n\n"
                f"Saisissez ce code sur la page de confirmation pour activer votre compte.\n"
                f"Ce code est valable 24h.\n\n"
                f"Si vous n'êtes pas à l'origine de cette inscription, ignorez cet email.\n\n"
                f"— L'équipe Djadara RH"
            )
            try:
               mail.send(msg)
            except Exception as mail_error:
                print(f"Error sending email: {mail_error}")

            session['pending_email'] = email
            session['pending_role']  = role
            flash("Inscription réussie ! Un code à 6 chiffres a été envoyé à votre adresse email.", "success")
            return redirect(url_for('verify_code'))

        except Exception as e:
            conn.rollback()
            print(f"Register error: {e}")
            if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
                flash("Cet email est déjà enregistré. Connectez-vous ou utilisez un autre email.", "danger")
            else:
                flash("Une erreur est survenue lors de l'inscription.", "danger")
        finally:
            cur.close()
            conn.close()

    return render_template('register.html')


# ── LOGIN (Candidat + Entreprise) ─────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        role     = request.form.get('role', 'candidat')
        password = request.form.get('password', '')

        if role == 'admin':
            return redirect(url_for('admin_login'))
        if role not in ('candidat', 'entreprise'):
            flash("Rôle invalide.", "danger")
            return redirect(url_for('login'))

        conn = get_db_connection()
        cur  = conn.cursor()

        if role == 'entreprise':
            cur.execute("SELECT * FROM entreprises WHERE email = %s;", (email,))
            user = cur.fetchone()
            cur.close(); conn.close()
            if user and check_password_hash(user['password'], password):
                if not user['is_verified']:
                    session['pending_email'] = email
                    session['pending_role']  = role
                    flash("Veuillez d'abord vérifier votre compte.", "warning")
                    return redirect(url_for('verify_code'))
                if not user.get('is_active', True):
                    flash("Votre compte a été désactivé. Contactez l'administrateur.", "danger")
                    return redirect(url_for('login'))
                session['entreprise_id']  = user['id']
                session['entreprise_nom'] = user['nom']
                return redirect(url_for('entreprise_dashboard'))

        elif role == 'candidat':
            cur.execute("SELECT * FROM candidats WHERE email = %s;", (email,))
            user = cur.fetchone()
            cur.close(); conn.close()
            if user and check_password_hash(user['password'], password):
                if not user['is_verified']:
                    session['pending_email'] = email
                    session['pending_role']  = role
                    flash("Veuillez d'abord vérifier votre compte.", "warning")
                    return redirect(url_for('verify_code'))
                if not user.get('is_active', True):
                    flash("Votre compte a été désactivé. Contactez l'administrateur.", "danger")
                    return redirect(url_for('login'))
                session['candidat_id']     = user['id']
                session['candidat_nom']    = user['nom']
                session['candidat_prenom'] = user['prenom']
                return redirect(url_for('candidat_dashboard'))
        else:
            cur.close(); conn.close()

        flash("Email ou mot de passe incorrect.", "danger")

    selected_role = request.args.get('role', request.form.get('role', 'candidat'))
    return render_template('login.html', selected_role=selected_role)


# ── LOGOUT ────────────────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ── FORGOT PASSWORD (Candidat + Entreprise) ───────────────────
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        role  = request.form.get('role', 'candidat')

        if role not in ('candidat', 'entreprise'):
            flash("Rôle invalide.", "danger")
            return redirect(url_for('forgot_password'))

        conn = get_db_connection()
        cur  = conn.cursor()
        try:
            tables = {'candidat': 'candidats', 'entreprise': 'entreprises'}
            table = tables[role]

            if role == 'entreprise':
                cur.execute(f"SELECT id, nom, '' AS prenom FROM {table} WHERE email = %s;", (email,))
            else:
                cur.execute(f"SELECT id, nom, prenom FROM {table} WHERE email = %s;", (email,))

            user = cur.fetchone()
            if user:
                code   = str(secrets.randbelow(900000) + 100000)
                expiry = datetime.now() + timedelta(minutes=15)
                cur.execute(
                    f"UPDATE {table} SET reset_code = %s, reset_expiry = %s WHERE id = %s;",
                    (code, expiry, user['id'])
                )
                conn.commit()

                msg = Message("Réinitialisation de votre mot de passe — Djadara RH", recipients=[email])
                msg.body = (
                    f"Bonjour {user.get('prenom', '')} {user['nom']},\n\n"
                    f"Voici votre code de réinitialisation :\n\n"
                    f"    {code}\n\n"
                    f"Ce code est valable 15 minutes.\n"
                    f"— L'équipe Djadara RH"
                )
                mail.send(msg)

                session['reset_user_id'] = user['id']
                session['reset_role']    = role
                session['reset_email']   = email
                flash("Un code de réinitialisation a été envoyé à votre adresse email.", "info")
                return redirect(url_for('reset_password'))
            else:
                flash("Si cet email est enregistré, vous recevrez un code sous peu.", "info")
                return redirect(url_for('forgot_password'))

        except Exception as e:
            conn.rollback()
            print(f"Forgot password error: {e}")
            flash("Une erreur est survenue.", "danger")
        finally:
            cur.close()
            conn.close()

    return render_template('forgot_password.html')


# ── RESET PASSWORD ────────────────────────────────────────────
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    user_id = session.get('reset_user_id')
    role    = session.get('reset_role', 'candidat')
    email   = session.get('reset_email', '')

    if not user_id:
        flash("Session expirée. Veuillez recommencer.", "warning")
        return redirect(url_for('forgot_password'))

    tables = {'candidat': 'candidats', 'entreprise': 'entreprises'}
    table  = tables.get(role, 'candidats')

    if request.method == 'POST':
        code             = request.form.get('code', '').strip()
        new_password     = request.form.get('password', '')
        confirm_password = request.form.get('confirm', '')

        if new_password != confirm_password:
            flash("Les mots de passe ne correspondent pas.", "danger")
            return render_template('reset_password.html', email=email)
        if len(new_password) < 6:
            flash("Le mot de passe doit contenir au moins 6 caractères.", "danger")
            return render_template('reset_password.html', email=email)

        conn = get_db_connection()
        cur  = conn.cursor()
        try:
            cur.execute(
                f"SELECT id FROM {table} WHERE id = %s AND reset_code = %s AND reset_expiry > NOW();",
                (user_id, code)
            )
            rec = cur.fetchone()
            if rec:
                cur.execute(
                    f"UPDATE {table} SET password = %s, reset_code = NULL, reset_expiry = NULL WHERE id = %s;",
                    (generate_password_hash(new_password), user_id)
                )
                conn.commit()
                session.pop('reset_user_id', None)
                session.pop('reset_role', None)
                session.pop('reset_email', None)
                flash("Mot de passe mis à jour ! Vous pouvez vous connecter.", "success")
                return redirect(url_for('login') + f'?role={role}')
            flash("Code invalide ou expiré.", "danger")
        except Exception as e:
            conn.rollback()
            print(f"Reset password error: {e}")
            flash("Une erreur est survenue.", "danger")
        finally:
            cur.close()
            conn.close()

    return render_template('reset_password.html', email=email)


# ── VERIFY (inscription) ──────────────────────────────────────
@app.route('/verify', methods=['GET', 'POST'])
def verify_code():
    email = session.get('pending_email')
    role  = session.get('pending_role', 'candidat')
    if not email:
        flash("Session expirée. Veuillez vous inscrire à nouveau.", "warning")
        return redirect(url_for('register'))

    if request.method == 'POST':
        code_saisi = request.form.get('code', '').strip()
        conn = get_db_connection()
        cur  = conn.cursor()
        try:
            tables = {'candidat': 'candidats', 'entreprise': 'entreprises'}
            table  = tables.get(role, 'candidats')
            cur.execute(
                f"SELECT id FROM {table} WHERE email = %s AND verif_token = %s AND is_verified = FALSE;",
                (email, code_saisi)
            )
            rec = cur.fetchone()
            if rec:
                cur.execute(
                    f"UPDATE {table} SET is_verified = TRUE, verif_token = NULL WHERE id = %s;",
                    (rec['id'],)
                )
                conn.commit()
                session.pop('pending_email', None)
                session.pop('pending_role', None)
                flash("Email vérifié ! Vous pouvez maintenant vous connecter.", "success")
                return redirect(url_for('login'))
            else:
                flash("Code incorrect ou expiré.", "danger")
        except Exception as e:
            conn.rollback()
            print(f"Verify error: {e}")
            flash("Une erreur est survenue.", "danger")
        finally:
            cur.close()
            conn.close()

    return render_template('verify.html', email=email)


# ── RESEND CODE ───────────────────────────────────────────────
@app.route('/resend-code')
def resend_code():
    email = session.get('pending_email')
    role  = session.get('pending_role', 'candidat')
    if not email:
        return redirect(url_for('register'))

    tables = {'candidat': 'candidats', 'entreprise': 'entreprises'}
    table  = tables.get(role, 'candidats')
    nouveau_code = str(secrets.randbelow(900000) + 100000)

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            f"UPDATE {table} SET verif_token = %s WHERE email = %s AND is_verified = FALSE;",
            (nouveau_code, email)
        )
        conn.commit()
        msg = Message("Votre nouveau code de vérification — Djadara RH", recipients=[email])
        msg.body = f"Votre nouveau code de vérification est :\n\n  {nouveau_code}\n\nCe code est valable 24h."
        try : 
         mail.send(msg)
        except Exception as mail_error:
         print(f"Error sending email: {mail_error}")
         flash("Erreur lors de l'envoi de l'email. Veuillez réessayer plus tard.", "danger")
         return redirect(url_for('verify_code'))
        flash("Un nouveau code a été envoyé à votre adresse email.", "info")
    except Exception as e:
        conn.rollback()
        print(f"Resend error: {e}")
        flash("Impossible de renvoyer le code.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('verify_code'))


# ══════════════════════════════════════════════════════════════
#   ESPACE ADMIN
# ══════════════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        admin_email = os.environ.get('ADMIN_EMAIL', 'admin@djadara.com')
        admin_pass  = os.environ.get('ADMIN_PASSWORD', 'admin123')

        if email == admin_email and password == admin_pass:
            session['admin']       = True
            session['admin_email'] = email
            flash("Bienvenue, Administrateur.", "success")
            return redirect(url_for('admin_dashboard'))

        flash("Identifiants administrateur incorrects.", "danger")

    return render_template('admin/login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    session.pop('admin_email', None)
    return redirect(url_for('admin_login'))


@app.route('/admin/dashboard')
def admin_dashboard():
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) AS n FROM candidats;")
    nb_candidats = cur.fetchone()['n']
    cur.execute("SELECT COUNT(*) AS n FROM entreprises;")
    nb_entreprises = cur.fetchone()['n']
    cur.execute("SELECT COUNT(*) AS n FROM offres;")
    nb_offres = cur.fetchone()['n']
    cur.execute("SELECT COUNT(*) AS n FROM candidatures;")
    nb_candidatures = cur.fetchone()['n']
    cur.execute("SELECT COUNT(*) AS n FROM entretiens;")
    nb_entretiens = cur.fetchone()['n']
    cur.execute("SELECT ROUND(AVG(score_global), 1) AS moy FROM candidatures;")
    row = cur.fetchone()
    score_moyen = row['moy'] if row and row['moy'] else 0
    cur.execute("SELECT COUNT(*) AS n FROM offres WHERE statut = 'active';")
    nb_offres_actives = cur.fetchone()['n']

    cur.close()
    conn.close()

    return render_template('admin/dashboard.html',
                           nb_candidats=nb_candidats,
                           nb_entreprises=nb_entreprises,
                           nb_offres=nb_offres,
                           nb_offres_actives=nb_offres_actives,
                           nb_candidatures=nb_candidatures,
                           nb_entretiens=nb_entretiens,
                           score_moyen=score_moyen)


@app.route('/admin/candidats')
def admin_candidats():
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, nom, prenom, email, telephone, is_verified, is_active,
               cv_fichier, created_at,
               (SELECT COUNT(*) FROM candidatures WHERE candidat_id = candidats.id) AS nb_candidatures
        FROM candidats
        ORDER BY created_at DESC;
    """)
    candidats = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('admin/candidats.html', candidats=candidats)


@app.route('/admin/candidats/<int:candidat_id>/toggle', methods=['POST'])
def admin_toggle_candidat(candidat_id):
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE candidats SET is_active = NOT is_active WHERE id = %s RETURNING is_active;",
            (candidat_id,)
        )
        result = cur.fetchone()
        conn.commit()
        etat = "activé" if result['is_active'] else "désactivé"
        flash(f"Compte candidat {etat}.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Toggle candidat error: {e}")
        flash("Erreur lors de la modification.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('admin_candidats'))


@app.route('/admin/candidats/<int:candidat_id>')
def admin_voir_candidat(candidat_id):
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM candidats WHERE id = %s;", (candidat_id,))
    candidat = cur.fetchone()
    cur.execute("""
        SELECT ca.*, o.titre AS offre_titre, e.nom AS entreprise_nom
        FROM candidatures ca
        JOIN offres o ON ca.offre_id = o.id
        LEFT JOIN entreprises e ON o.entreprise_id = e.id
        WHERE ca.candidat_id = %s
        ORDER BY ca.created_at DESC;
    """, (candidat_id,))
    candidatures = cur.fetchall()
    cur.close()
    conn.close()
    if not candidat:
        flash("Candidat introuvable.", "danger")
        return redirect(url_for('admin_candidats'))
    return render_template('admin/voir_candidat.html', candidat=candidat, candidatures=candidatures)


@app.route('/admin/candidats/<int:candidat_id>/supprimer', methods=['POST'])
def admin_supprimer_candidat(candidat_id):
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM candidats WHERE id = %s;", (candidat_id,))
        conn.commit()
        flash("Candidat supprimé définitivement.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Admin delete candidat error: {e}")
        flash("Erreur lors de la suppression.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('admin_candidats'))


@app.route('/admin/entreprises')
def admin_entreprises():
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, nom, secteur, email, telephone, is_verified, is_active, created_at,
               (SELECT COUNT(*) FROM offres WHERE entreprise_id = entreprises.id) AS nb_offres
        FROM entreprises
        ORDER BY created_at DESC;
    """)
    entreprises = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('admin/entreprises.html', entreprises=entreprises)


@app.route('/admin/entreprises/<int:entreprise_id>/toggle', methods=['POST'])
def admin_toggle_entreprise(entreprise_id):
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE entreprises SET is_active = NOT is_active WHERE id = %s RETURNING is_active;",
            (entreprise_id,)
        )
        result = cur.fetchone()
        conn.commit()
        etat = "activée" if result['is_active'] else "désactivée"
        flash(f"Entreprise {etat}.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Toggle entreprise error: {e}")
        flash("Erreur lors de la modification.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('admin_entreprises'))


@app.route('/admin/entreprises/<int:entreprise_id>/supprimer', methods=['POST'])
def admin_supprimer_entreprise(entreprise_id):
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM entreprises WHERE id = %s;", (entreprise_id,))
        conn.commit()
        flash("Entreprise supprimée définitivement.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Admin delete entreprise error: {e}")
        flash("Erreur lors de la suppression.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('admin_entreprises'))


@app.route('/admin/offres')
def admin_offres():
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT o.id, o.titre, o.type_contrat, o.localisation, o.statut, o.created_at,
               e.nom AS entreprise_nom,
               (SELECT COUNT(*) FROM candidatures WHERE offre_id = o.id) AS nb_candidatures
        FROM offres o
        LEFT JOIN entreprises e ON o.entreprise_id = e.id
        ORDER BY o.created_at DESC;
    """)
    offres = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('admin/offres.html', offres=offres)


@app.route('/admin/offres/<int:offre_id>/supprimer', methods=['POST'])
def admin_supprimer_offre(offre_id):
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM offres WHERE id = %s;", (offre_id,))
        conn.commit()
        flash("Offre supprimée.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Admin delete offre error: {e}")
        flash("Erreur lors de la suppression.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('admin_offres'))


@app.route('/admin/stats')
def admin_stats():
    r = admin_auth_required()
    if r: return r

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM v_stats_plateforme;")
    stats = cur.fetchone()
    cur.close()
    conn.close()
    return render_template('admin/stats.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    if 'candidat_id' in session:
        return redirect(url_for('candidat_dashboard'))
    if 'entreprise_id' in session:
        return redirect(url_for('entreprise_dashboard'))
    if 'admin' in session:
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('login'))


# ══════════════════════════════════════════════════════════════
#   ESPACE ENTREPRISE
# ══════════════════════════════════════════════════════════════

@app.route('/entreprise/dashboard')
def entreprise_dashboard():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    cur.execute("SELECT * FROM entreprises WHERE id = %s;", (entreprise_id,))
    entreprise = cur.fetchone()
    cur.execute("SELECT * FROM v_stats_entreprise WHERE entreprise_id = %s;", (entreprise_id,))
    stats = cur.fetchone()
    cur.execute("""
        SELECT o.id, o.titre, o.type_contrat, o.statut, o.created_at,
               COUNT(ca.id) AS nb_candidatures
        FROM offres o
        LEFT JOIN candidatures ca ON ca.offre_id = o.id
        WHERE o.entreprise_id = %s
        GROUP BY o.id, o.titre, o.type_contrat, o.statut, o.created_at
        ORDER BY o.created_at DESC
        LIMIT 10;
    """, (entreprise_id,))
    offres = cur.fetchall()
    cur.execute("""
        SELECT titre, message, type, created_at
        FROM notifications
        WHERE entreprise_id = %s AND lue = FALSE
        ORDER BY created_at DESC LIMIT 10;
    """, (entreprise_id,))
    notifs = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('entreprise/dashboard.html',
                           entreprise=entreprise, stats=stats,
                           offres=offres, notifs=notifs)


@app.route('/entreprise/profil', methods=['GET', 'POST'])
def entreprise_profil():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    if request.method == 'POST':
        try:
            new_password     = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            cur.execute("""
                UPDATE entreprises SET
                    nom = %s, secteur = %s, adresse = %s, telephone = %s
                WHERE id = %s;
            """, (
                request.form.get('nom'), request.form.get('secteur'),
                request.form.get('adresse'), request.form.get('telephone'),
                entreprise_id
            ))
            if new_password:
                if new_password != confirm_password:
                    flash("Les mots de passe ne correspondent pas.", "danger")
                    conn.rollback()
                elif len(new_password) < 6:
                    flash("Le mot de passe doit contenir au moins 6 caractères.", "danger")
                else:
                    cur.execute(
                        "UPDATE entreprises SET password = %s WHERE id = %s;",
                        (generate_password_hash(new_password), entreprise_id)
                    )
            conn.commit()
            session['entreprise_nom'] = request.form.get('nom')
            flash("Profil mis à jour avec succès.", "success")
        except Exception as e:
            conn.rollback()
            print(f"Entreprise profil update error: {e}")
            flash("Erreur lors de la mise à jour.", "danger")

    cur.execute("SELECT * FROM entreprises WHERE id = %s;", (entreprise_id,))
    entreprise = cur.fetchone()
    cur.close(); conn.close()
    return render_template('entreprise/profil.html', entreprise=entreprise)


@app.route('/entreprise/offres', methods=['GET', 'POST'])
def entreprise_offres():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    if request.method == 'POST':
        try:
            cur.execute("""
                INSERT INTO offres (
                    titre, competences, experience_min, niveau_etude,
                    description, entreprise_id, localisation, type_contrat, salaire,
                    langues_requises, certifications,
                    poids_competences, poids_experience, poids_formation, poids_langues, poids_certif,
                    statut
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """, (
                request.form.get('titre'), request.form.get('competences'),
                request.form.get('experience_min', 0) or 0, request.form.get('niveau_etude', ''),
                request.form.get('description'), entreprise_id,
                request.form.get('localisation', ''), request.form.get('type_contrat', ''),
                request.form.get('salaire', ''), request.form.get('langues_requises', ''),
                request.form.get('certifications', ''),
                request.form.get('poids_competences', 40) or 40,
                request.form.get('poids_experience', 25) or 25,
                request.form.get('poids_formation', 20) or 20,
                request.form.get('poids_langues', 10) or 10,
                request.form.get('poids_certif', 5) or 5,
                request.form.get('statut', 'active')
            ))
            conn.commit()
            flash("Offre publiée avec succès !", "success")
        except Exception as e:
            conn.rollback()
            print(f"Entreprise offre insert error: {e}")
            flash("Erreur lors de la publication de l'offre.", "danger")
        finally:
            cur.close(); conn.close()
        return redirect(url_for('entreprise_offres'))

    cur.execute("""
        SELECT o.*, (SELECT COUNT(*) FROM candidatures ca WHERE ca.offre_id = o.id) AS nb_candidatures
        FROM offres o
        WHERE o.entreprise_id = %s
        ORDER BY o.created_at DESC;
    """, (entreprise_id,))
    offres_list = cur.fetchall()
    cur.close(); conn.close()
    return render_template('entreprise/offres.html', offres=offres_list)


@app.route('/entreprise/offres/<int:offre_id>/edit', methods=['GET', 'POST'])
def entreprise_offre_edit(offre_id: int):
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM offres WHERE id = %s AND entreprise_id = %s;", (offre_id, entreprise_id))
    offre = cur.fetchone()

    if not offre:
        cur.close(); conn.close()
        flash("Offre introuvable.", "danger")
        return redirect(url_for('entreprise_offres'))

    if request.method == 'POST':
        try:
            cur.execute("""
                UPDATE offres SET
                    titre = %s, competences = %s, description = %s,
                    experience_min = %s, niveau_etude = %s,
                    langues_requises = %s, certifications = %s,
                    localisation = %s, type_contrat = %s, salaire = %s,
                    poids_competences = %s, poids_experience = %s,
                    poids_formation = %s, poids_langues = %s,
                    poids_certif = %s, statut = %s, updated_at = NOW()
                WHERE id = %s AND entreprise_id = %s;
            """, (
                request.form.get('titre'), request.form.get('competences'),
                request.form.get('description'),
                request.form.get('experience_min', 0) or 0, request.form.get('niveau_etude', ''),
                request.form.get('langues_requises', ''), request.form.get('certifications', ''),
                request.form.get('localisation', ''), request.form.get('type_contrat', ''),
                request.form.get('salaire', ''),
                request.form.get('poids_competences', 40) or 40,
                request.form.get('poids_experience', 25) or 25,
                request.form.get('poids_formation', 20) or 20,
                request.form.get('poids_langues', 10) or 10,
                request.form.get('poids_certif', 5) or 5,
                request.form.get('statut', 'active'),
                offre_id, entreprise_id
            ))
            conn.commit()
            flash("Offre mise à jour.", "success")
            return redirect(url_for('entreprise_offres'))
        except Exception as e:
            conn.rollback()
            print(f"Entreprise offre update error: {e}")
            flash("Erreur lors de la mise à jour.", "danger")
        finally:
            cur.close(); conn.close()
        return redirect(url_for('entreprise_offres'))

    cur.close(); conn.close()
    return render_template('entreprise/offre_edit.html', offre=offre)


@app.route('/entreprise/offres/<int:offre_id>/fermer', methods=['POST'])
def entreprise_offre_fermer(offre_id: int):
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE offres SET statut = 'fermée', updated_at = NOW() WHERE id = %s AND entreprise_id = %s;",
            (offre_id, entreprise_id)
        )
        conn.commit()
        flash("Offre fermée.", "info")
    except Exception as e:
        conn.rollback()
        flash("Erreur lors de la fermeture.", "danger")
    finally:
        cur.close(); conn.close()
    return redirect(url_for('entreprise_offres'))


@app.route('/entreprise/offres/<int:offre_id>/supprimer', methods=['POST'])
def entreprise_offre_supprimer(offre_id: int):
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM offres WHERE id = %s AND entreprise_id = %s;", (offre_id, entreprise_id))
        conn.commit()
        flash("Offre supprimée.", "success")
    except Exception as e:
        conn.rollback()
        flash("Erreur lors de la suppression.", "danger")
    finally:
        cur.close(); conn.close()
    return redirect(url_for('entreprise_offres'))


@app.route('/entreprise/resultats/<int:offre_id>')
def entreprise_resultats(offre_id: int):
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM offres WHERE id = %s AND entreprise_id = %s;", (offre_id, entreprise_id))
    offre = cur.fetchone()
    if not offre:
        cur.close(); conn.close()
        flash("Accès refusé.", "danger")
        return redirect(url_for('entreprise_offres'))

    cur.execute("""
        SELECT ca.*, cand.nom, cand.prenom, cand.email
        FROM candidatures ca
        JOIN candidats cand ON ca.candidat_id = cand.id
        WHERE ca.offre_id = %s
        ORDER BY ca.score_global DESC;
    """, (offre_id,))
    candidatures = cur.fetchall()

    cur.execute("SELECT * FROM cvs WHERE offre_id = %s ORDER BY score_global DESC;", (offre_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    candidats_ia = []
    for idx, row in enumerate(rows):
        candidats_ia.append({
            'rang':                   idx + 1,
            'nom_candidat':           row['nom_fichier'].replace('.pdf', '').replace('_', ' ').title(),
            'fichier':                row['nom_fichier'],
            'global_score':           row['score_global'],
            'score_competences':      row['score_competences'],
            'score_experience':       row['score_experience'],
            'score_formation':        row['score_formation'],
            'score_langues':          row['score_langues'],
            'competences_trouvees':   json.loads(row['competences_trouvees'] or '[]'),
            'competences_manquantes': json.loads(row['competences_manquantes'] or '[]'),
            'points_forts':           json.loads(row['points_forts'] or '[]'),
            'points_ameliorer':       json.loads(row['points_ameliorer'] or '[]'),
            'niveau':                 row['niveau_qualitatif'],
        })
    return render_template('entreprise/resultats.html',
                           candidats=candidats_ia, offre=dict(offre), candidatures=candidatures)


@app.route('/entreprise/matching')
def entreprise_matching_index():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT o.id, o.titre,
               (SELECT COUNT(*) FROM candidatures ca WHERE ca.offre_id = o.id) AS nb_candidatures
        FROM offres o
        WHERE o.entreprise_id = %s
        ORDER BY o.created_at DESC;
    """, (entreprise_id,))
    offres_list = cur.fetchall()
    cur.close(); conn.close()
    return render_template('entreprise/matching.html', offre=None, candidatures=[], offres=offres_list)


@app.route('/entreprise/matching/<int:offre_id>', methods=['GET', 'POST'])
def entreprise_matching(offre_id: int):
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    cur.execute("SELECT * FROM offres WHERE id = %s AND entreprise_id = %s;", (offre_id, entreprise_id))
    offre = cur.fetchone()
    if not offre:
        cur.close(); conn.close()
        flash("Offre introuvable.", "danger")
        return redirect(url_for('entreprise_offres'))

    # Jointure avec cvs pour récupérer competences_trouvees / manquantes
    cur.execute("""
        SELECT ca.*, cand.nom, cand.prenom, cand.email, cand.cv_fichier, cand.cv_texte_brut,
               cv.competences_trouvees, cv.competences_manquantes,
               cv.points_forts, cv.points_ameliorer
        FROM candidatures ca
        JOIN candidats cand ON ca.candidat_id = cand.id
        LEFT JOIN cvs cv ON ca.cv_id = cv.id
        WHERE ca.offre_id = %s
        ORDER BY ca.score_global DESC;
    """, (offre_id,))
    candidatures = cur.fetchall()

    if request.method == 'POST':
        try:
            offre_dict = dict(offre)
            offer_text = f"{offre_dict.get('titre','')} {offre_dict.get('description','')} {offre_dict.get('competences','')}"
            candidatures_list = [dict(c) for c in candidatures]
            textes = [c.get('cv_texte_brut') or '' for c in candidatures_list]

            if not candidatures_list:
                flash("Aucun candidat pour cette offre.", "warning")
            else:
                tfidf_sims = compute_tfidf_cosine(textes, offer_text)
                if len(tfidf_sims) >= 2:
                    knn_boost   = compute_knn_boost(tfidf_sims)
                    tfidf_final = tfidf_sims * 0.8 + knn_boost * 0.2
                else:
                    tfidf_final = tfidf_sims

                for i, c in enumerate(candidatures_list):
                    texte = c.get('cv_texte_brut') or ''
                    res   = evaluer_cv_complet(texte, offre_dict, float(tfidf_final[i]))

                    # INSERT dans cvs (ou UPDATE si déjà existant)
                    # Fonctionne même si cv_id est NULL (candidatures anciennes)
                    cur.execute("""
                        INSERT INTO cvs (
                            candidat_id, nom_fichier,
                            score_global, score_competences, score_experience,
                            score_formation, score_langues, score_certifications,
                            competences_trouvees, competences_manquantes,
                            points_forts, points_ameliorer,
                            niveau_qualitatif, offre_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (nom_fichier, offre_id) DO UPDATE SET
                            score_global           = EXCLUDED.score_global,
                            score_competences      = EXCLUDED.score_competences,
                            score_experience       = EXCLUDED.score_experience,
                            score_formation        = EXCLUDED.score_formation,
                            score_langues          = EXCLUDED.score_langues,
                            score_certifications   = EXCLUDED.score_certifications,
                            competences_trouvees   = EXCLUDED.competences_trouvees,
                            competences_manquantes = EXCLUDED.competences_manquantes,
                            points_forts           = EXCLUDED.points_forts,
                            points_ameliorer       = EXCLUDED.points_ameliorer,
                            niveau_qualitatif      = EXCLUDED.niveau_qualitatif
                        RETURNING id;
                    """, (
                        c['candidat_id'], c.get('cv_fichier') or str(c['candidat_id']),
                        res['score'], res['score_competences'], res['score_experience'],
                        res['score_formation'], res['score_langues'], res['score_certifications'],
                        res['competences_trouvees'], res['competences_manquantes'],
                        res['points_forts'], res['points_ameliorer'],
                        res['niveau'], offre_id
                    ))
                    cv_id = cur.fetchone()['id']

                    # Mettre à jour candidatures avec scores + cv_id lié
                    cur.execute("""
                        UPDATE candidatures SET
                            cv_id                = %s,
                            score_global         = %s,
                            score_competences    = %s,
                            score_experience     = %s,
                            score_formation      = %s,
                            score_langues        = %s,
                            score_certifications = %s,
                            niveau               = %s
                        WHERE id = %s;
                    """, (
                        cv_id,
                        res['score'], res['score_competences'], res['score_experience'],
                        res['score_formation'], res['score_langues'], res['score_certifications'],
                        res['niveau'], c['id']
                    ))

                cur.execute("""
                    UPDATE candidatures ca
                    SET rang = sub.rang
                    FROM (
                        SELECT id, RANK() OVER (ORDER BY score_global DESC) AS rang
                        FROM candidatures
                        WHERE offre_id = %s
                    ) sub
                    WHERE ca.id = sub.id;
                """, (offre_id,))
                conn.commit()
                flash("Scores recalculés avec succès.", "success")

        except Exception as e:
            conn.rollback()
            print(f"Matching error: {e}")
            flash(f"Erreur lors du calcul des scores : {e}", "danger")
        finally:
            cur.close(); conn.close()
        return redirect(url_for('entreprise_matching', offre_id=offre_id))

    conn2 = get_db_connection()
    cur2  = conn2.cursor()
    cur2.execute("""
        SELECT o.id, o.titre,
               (SELECT COUNT(*) FROM candidatures ca WHERE ca.offre_id = o.id) AS nb_candidatures
        FROM offres o
        WHERE o.entreprise_id = %s
        ORDER BY o.created_at DESC;
    """, (entreprise_id,))
    offres_list = cur2.fetchall()
    cur2.close(); conn2.close()

    return render_template('entreprise/matching.html',
                           offre=offre, candidatures=candidatures, offres=offres_list)


@app.route('/entreprise/candidatures')
def entreprise_candidatures():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT ca.id, ca.score_global, ca.statut, ca.rang, ca.created_at,
               cand.nom, cand.prenom, cand.email,
               o.titre AS offre_titre, o.id AS offre_id
        FROM candidatures ca
        JOIN candidats cand ON ca.candidat_id = cand.id
        JOIN offres    o    ON ca.offre_id    = o.id
        WHERE o.entreprise_id = %s
        ORDER BY ca.created_at DESC;
    """, (entreprise_id,))
    candidatures = cur.fetchall()
    cur.close(); conn.close()
    return render_template('entreprise/candidatures.html', candidatures=candidatures)


@app.route('/entreprise/candidatures/<int:candidature_id>/statut', methods=['POST'])
def entreprise_update_statut(candidature_id: int):
    r = entreprise_auth_required()
    if r: return r

    nouveau_statut = request.form.get('statut', 'soumise')
    note           = request.form.get('note_recruteur', '')
    offre_id       = request.form.get('offre_id')

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            UPDATE candidatures SET statut = %s, note_recruteur = %s, updated_at = NOW()
            WHERE id = %s;
        """, (nouveau_statut, note, candidature_id))

        cur.execute("SELECT candidat_id FROM candidatures WHERE id = %s;", (candidature_id,))
        row = cur.fetchone()
        if row:
            messages_statut = {
                'vue':             ("Votre candidature a été vue",
                                    "Un responsable RH a consulté votre candidature."),
                'présélectionnée': ("Vous êtes présélectionné(e) !",
                                    "Félicitations ! Votre profil a été retenu pour la prochaine étape."),
                'entretien':       ("Convocation à un entretien",
                                    "Vous avez été convoqué(e) à un entretien. Consultez vos convocations."),
                'offre':           ("Offre d'emploi !",
                                    "Félicitations ! Une offre d'emploi vous a été proposée."),
                'refusée':         ("Candidature non retenue",
                                    "Votre candidature n'a pas été retenue pour ce poste."),
            }
            if nouveau_statut in messages_statut:
                titre_notif, msg_notif = messages_statut[nouveau_statut]
                cur.execute("""
                    INSERT INTO notifications (candidat_id, titre, message, type)
                    VALUES (%s, %s, %s, 'info');
                """, (row['candidat_id'], titre_notif, msg_notif))

        conn.commit()
        flash("Statut mis à jour.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Update statut error: {e}")
        flash("Erreur lors de la mise à jour.", "danger")
    finally:
        cur.close(); conn.close()

    if offre_id:
        return redirect(url_for('entreprise_matching', offre_id=offre_id))
    return redirect(url_for('entreprise_candidatures'))


@app.route('/entreprise/entretiens', methods=['GET', 'POST'])
def entreprise_entretiens():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    if request.method == 'POST':
        candidature_id = request.form.get('candidature_id')
        date_heure     = request.form.get('date_heure')
        duree_min      = request.form.get('duree_min', 60)
        type_entretien = request.form.get('type', 'présentiel')
        lien_visio     = request.form.get('lien_visio', '')
        lieu           = request.form.get('lieu', '')
        notes          = request.form.get('notes', '')

        try:
            cur.execute("""
                INSERT INTO entretiens
                    (candidature_id, entreprise_id, date_heure, duree_min,
                     type, lien_visio, lieu, notes, statut)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'planifié');
            """, (candidature_id, entreprise_id, date_heure, duree_min,
                  type_entretien, lien_visio, lieu, notes))

            cur.execute("SELECT candidat_id FROM candidatures WHERE id = %s;", (candidature_id,))
            row = cur.fetchone()
            if row:
                cur.execute("""
                    INSERT INTO notifications (candidat_id, titre, message, type)
                    VALUES (%s, %s, %s, 'entretien');
                """, (row['candidat_id'],
                      "Convocation à un entretien",
                      f"Vous êtes convoqué(e) à un entretien le {date_heure}. Type : {type_entretien}."))

            cur.execute(
                "UPDATE candidatures SET statut = 'entretien', updated_at = NOW() WHERE id = %s;",
                (candidature_id,)
            )
            conn.commit()
            flash("Entretien planifié avec succès.", "success")
        except Exception as e:
            conn.rollback()
            print(f"Entretien insert error: {e}")
            flash("Erreur lors de la planification de l'entretien.", "danger")

    cur.execute("""
        SELECT et.*, ca.statut AS statut_candidature,
               cand.nom, cand.prenom, cand.email,
               o.titre AS offre_titre
        FROM entretiens et
        JOIN candidatures ca   ON et.candidature_id = ca.id
        JOIN candidats    cand ON ca.candidat_id    = cand.id
        JOIN offres       o    ON ca.offre_id       = o.id
        WHERE o.entreprise_id = %s
        ORDER BY et.date_heure DESC;
    """, (entreprise_id,))
    entretiens = cur.fetchall()

    cur.execute("""
        SELECT ca.id, ca.statut, cand.nom, cand.prenom, o.titre AS offre_titre
        FROM candidatures ca
        JOIN candidats cand ON ca.candidat_id = cand.id
        JOIN offres    o    ON ca.offre_id    = o.id
        WHERE o.entreprise_id = %s
          AND LOWER(TRIM(ca.statut)) NOT IN ('refusée', 'refusee')
        ORDER BY cand.nom;
    """, (entreprise_id,))
    candidatures_disponibles = cur.fetchall()

    cur.close(); conn.close()
    return render_template('entreprise/entretiens.html',
                           entretiens=entretiens,
                           candidatures_disponibles=candidatures_disponibles)


@app.route('/entreprise/notifier', methods=['POST'])
def entreprise_notifier():
    r = entreprise_auth_required()
    if r: return r

    candidat_id = request.form.get('candidat_id')
    titre       = request.form.get('titre', '').strip()
    message     = request.form.get('message', '').strip()
    offre_id    = request.form.get('offre_id')

    if not candidat_id or not titre:
        flash("Candidat et titre obligatoires.", "danger")
        return redirect(request.referrer or url_for('entreprise_dashboard'))

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO notifications (candidat_id, titre, message, type)
            VALUES (%s, %s, %s, 'info');
        """, (candidat_id, titre, message))
        conn.commit()
        flash("Notification envoyée.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Notif error: {e}")
        flash("Erreur lors de l'envoi.", "danger")
    finally:
        cur.close(); conn.close()

    if offre_id:
        return redirect(url_for('entreprise_matching', offre_id=offre_id))
    return redirect(url_for('entreprise_dashboard'))


@app.route('/entreprise/stats')
def entreprise_stats():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            CASE
                WHEN ca.score_global >= 75 THEN 'Excellent (75-100)'
                WHEN ca.score_global >= 60 THEN 'Bon (60-74)'
                WHEN ca.score_global >= 40 THEN 'Partiel (40-59)'
                ELSE 'Insuffisant (0-39)'
            END AS tranche,
            COUNT(*) AS nb
        FROM candidatures ca
        JOIN offres o ON ca.offre_id = o.id
        WHERE o.entreprise_id = %s
        GROUP BY tranche
        ORDER BY tranche;
    """, (entreprise_id,))
    distribution = cur.fetchall()

    cur.execute("""
        SELECT o.titre, COUNT(ca.id) AS nb,
               ROUND(AVG(ca.score_global), 1) AS score_moy
        FROM offres o
        LEFT JOIN candidatures ca ON ca.offre_id = o.id
        WHERE o.entreprise_id = %s
        GROUP BY o.id, o.titre
        ORDER BY nb DESC
        LIMIT 10;
    """, (entreprise_id,))
    offres_top = cur.fetchall()

    cur.execute("SELECT * FROM entreprises WHERE id = %s;", (entreprise_id,))
    entreprise = cur.fetchone()
    cur.close(); conn.close()
    return render_template('entreprise/stats.html',
                           distribution=distribution, offres_top=offres_top, entreprise=entreprise)


@app.route('/entreprise/notifications')
def entreprise_notifications():
    r = entreprise_auth_required()
    if r: return r

    entreprise_id = session['entreprise_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT titre, message, type, lue, created_at FROM notifications
        WHERE entreprise_id = %s
        ORDER BY created_at DESC;
    """, (entreprise_id,))
    notifications = cur.fetchall()
    cur.execute(
        "UPDATE notifications SET lue = TRUE WHERE entreprise_id = %s AND lue = FALSE;",
        (entreprise_id,)
    )
    conn.commit()
    cur.close(); conn.close()
    return render_template('entreprise/notifications.html', notifications=notifications)


@app.route('/entreprise/logout')
def entreprise_logout():
    session.pop('entreprise_id', None)
    session.pop('entreprise_nom', None)
    return redirect(url_for('login') + '?role=entreprise')


# ══════════════════════════════════════════════════════════════
#   ESPACE CANDIDAT
# ══════════════════════════════════════════════════════════════

@app.route('/candidat/dashboard')
def candidat_dashboard():
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    cur.execute("SELECT * FROM candidats WHERE id = %s;", (candidat_id,))
    candidat = cur.fetchone()
    cur.execute("""
        SELECT ca.score_global, ca.statut, ca.rang,
               o.titre AS offre_titre, o.type_contrat, o.localisation,
               e.nom AS entreprise_nom
        FROM candidatures ca
        JOIN offres o ON ca.offre_id = o.id
        LEFT JOIN entreprises e ON o.entreprise_id = e.id
        WHERE ca.candidat_id = %s
        ORDER BY ca.created_at DESC
        LIMIT 10;
    """, (candidat_id,))
    candidatures = cur.fetchall()
    cur.execute("""
        SELECT titre, message FROM notifications
        WHERE candidat_id = %s AND lue = FALSE
        ORDER BY created_at DESC LIMIT 10;
    """, (candidat_id,))
    notifs = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('candidat/dashboard.html',
                           candidat=candidat, candidatures=candidatures, notifs=notifs)


@app.route('/candidat/profil', methods=['GET', 'POST'])
def candidat_profil():
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    if request.method == 'POST':
        try:
            new_password     = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()
            cur.execute("""
                UPDATE candidats SET
                    nom = %s, prenom = %s, telephone = %s, titre_poste = %s,
                    resume = %s, niveau_etude = %s, annees_experience = %s
                WHERE id = %s;
            """, (
                request.form.get('nom'), request.form.get('prenom'),
                request.form.get('telephone'), request.form.get('titre_poste'),
                request.form.get('resume'), request.form.get('niveau_etude'),
                request.form.get('annees_experience', 0) or 0,
                candidat_id
            ))
            if new_password:
                if new_password != confirm_password:
                    flash("Les mots de passe ne correspondent pas.", "danger")
                    conn.rollback()
                elif len(new_password) < 6:
                    flash("Le mot de passe doit contenir au moins 6 caractères.", "danger")
                else:
                    cur.execute(
                        "UPDATE candidats SET password = %s WHERE id = %s;",
                        (generate_password_hash(new_password), candidat_id)
                    )
            conn.commit()
            flash("Profil mis à jour.", "success")
        except Exception as e:
            conn.rollback()
            print(f"Profil update error: {e}")
            flash("Erreur lors de la mise à jour.", "danger")

    cur.execute("SELECT * FROM candidats WHERE id = %s;", (candidat_id,))
    candidat = cur.fetchone()
    cur.close()
    conn.close()
    return render_template('candidat/profil.html', candidat=candidat)


@app.route('/candidat/upload-cv', methods=['GET', 'POST'])
def candidat_upload_cv():
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']

    if request.method == 'POST':
        file = request.files.get('cv_files')
        if not file or file.filename == '' or not allowed_file(file.filename):
            flash("Veuillez sélectionner un fichier PDF valide.", "danger")
            return redirect(url_for('candidat_upload_cv'))

        filename  = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        texte = extract_pdf_text(file_path)
        conn  = get_db_connection()
        cur   = conn.cursor()
        try:
            cur.execute("""
                UPDATE candidats SET
                    cv_fichier    = %s,
                    cv_texte_brut = %s,
                    cv_analyse_at = NOW()
                WHERE id = %s;
            """, (filename, texte, candidat_id))
            conn.commit()
            flash("CV déposé et analysé avec succès !", "success")
        except Exception as e:
            conn.rollback()
            print(f"Upload CV error: {e}")
            flash("Erreur lors de l'enregistrement du CV.", "danger")
        finally:
            cur.close()
            conn.close()
        return redirect(url_for('candidat_dashboard'))

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id, titre FROM offres WHERE statut = 'active' ORDER BY created_at DESC;")
    offres = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('candidat/upload_cv.html', offres=offres)


@app.route('/candidat/offres')
def candidat_offres():
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT o.id, o.titre, o.localisation, o.type_contrat, o.description,
               o.competences, o.salaire, o.created_at,
               e.nom AS entreprise_nom,
               EXISTS (
                   SELECT 1 FROM candidatures ca
                   WHERE ca.offre_id = o.id AND ca.candidat_id = %s
               ) AS deja_postule
        FROM offres o
        LEFT JOIN entreprises e ON o.entreprise_id = e.id
        WHERE o.statut = 'active'
        ORDER BY o.created_at DESC;
    """, (candidat_id,))
    offres = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('candidat/offres.html', offres=offres)


@app.route('/candidat/postuler/<int:offre_id>', methods=['POST'])
def candidat_postuler(offre_id):
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT cv_fichier, cv_texte_brut FROM candidats WHERE id = %s;", (candidat_id,))
        candidat = cur.fetchone()
        if not candidat or not candidat['cv_fichier']:
            flash("Vous devez d'abord déposer votre CV avant de postuler.", "warning")
            return redirect(url_for('candidat_upload_cv'))

        cur.execute(
            "SELECT id FROM candidatures WHERE candidat_id = %s AND offre_id = %s;",
            (candidat_id, offre_id)
        )
        if cur.fetchone():
            flash("Vous avez déjà postulé à cette offre.", "info")
            return redirect(url_for('candidat_offres'))

        cur.execute("SELECT * FROM offres WHERE id = %s AND statut = 'active';", (offre_id,))
        offre = cur.fetchone()
        if not offre:
            flash("Cette offre n'est plus disponible.", "danger")
            return redirect(url_for('candidat_offres'))

        offre_dict  = dict(offre)
        texte_cv    = candidat['cv_texte_brut'] or ""
        offer_text  = f"{offre_dict.get('titre','')} {offre_dict.get('description','')} {offre_dict.get('competences','')}"
        tfidf_score = float(compute_tfidf_cosine([texte_cv], offer_text)[0]) if texte_cv else 0.0
        res         = evaluer_cv_complet(texte_cv, offre_dict, tfidf_score)

        # Étape 1 — Insérer dans cvs pour sauvegarder competences_trouvees / manquantes
        # ON CONFLICT : si ce CV a déjà été analysé pour cette offre, on met à jour.
        cur.execute("""
            INSERT INTO cvs (
                candidat_id, nom_fichier,
                score_global, score_competences, score_experience,
                score_formation, score_langues, score_certifications,
                competences_trouvees, competences_manquantes,
                points_forts, points_ameliorer,
                niveau_qualitatif, offre_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (nom_fichier, offre_id) DO UPDATE SET
                score_global           = EXCLUDED.score_global,
                score_competences      = EXCLUDED.score_competences,
                score_experience       = EXCLUDED.score_experience,
                score_formation        = EXCLUDED.score_formation,
                score_langues          = EXCLUDED.score_langues,
                score_certifications   = EXCLUDED.score_certifications,
                competences_trouvees   = EXCLUDED.competences_trouvees,
                competences_manquantes = EXCLUDED.competences_manquantes,
                points_forts           = EXCLUDED.points_forts,
                points_ameliorer       = EXCLUDED.points_ameliorer,
                niveau_qualitatif      = EXCLUDED.niveau_qualitatif
            RETURNING id;
        """, (
            candidat_id, candidat['cv_fichier'],
            res['score'], res['score_competences'], res['score_experience'],
            res['score_formation'], res['score_langues'], res['score_certifications'],
            res['competences_trouvees'], res['competences_manquantes'],
            res['points_forts'], res['points_ameliorer'],
            res['niveau'], offre_id
        ))
        cv_id = cur.fetchone()['id']

        # Étape 2 — Insérer la candidature avec cv_id lié
        cur.execute("""
            INSERT INTO candidatures
                (candidat_id, offre_id, cv_id, score_global, score_competences,
                 score_experience, score_formation, score_langues,
                 score_certifications, niveau, statut, rang)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'soumise', 0);
        """, (
            candidat_id, offre_id, cv_id,
            res['score'], res['score_competences'], res['score_experience'],
            res['score_formation'], res['score_langues'],
            res['score_certifications'], res['niveau']
        ))

        # FIX 5 — Rang recalculé pour TOUS les candidats de cette offre
        # selon score_global DESC (rang 1 = meilleur).
        # Avant : rang = compteur d'arrivée, sans rapport avec le score.
        cur.execute("""
            UPDATE candidatures SET rang = sous_requete.nouveau_rang
            FROM (
                SELECT id,
                       RANK() OVER (ORDER BY score_global DESC) AS nouveau_rang
                FROM candidatures
                WHERE offre_id = %s
            ) AS sous_requete
            WHERE candidatures.id = sous_requete.id
              AND candidatures.offre_id = %s;
        """, (offre_id, offre_id))

        conn.commit()
        flash("Candidature envoyée avec succès !", "success")
    except Exception as e:
        conn.rollback()
        print(f"Postuler error: {e}")
        flash("Une erreur est survenue.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('candidat_offres'))


@app.route('/candidat/candidatures')
def candidat_candidatures():
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT ca.id, ca.score_global, ca.statut, ca.rang, ca.created_at,
               o.titre AS offre_titre, o.type_contrat, o.localisation,
               e.nom AS entreprise_nom,
               et.date_heure AS entretien_dt, et.type AS entretien_type,
               et.lieu AS entretien_lieu, et.lien_visio
        FROM candidatures ca
        JOIN offres o ON ca.offre_id = o.id
        LEFT JOIN entreprises e ON o.entreprise_id = e.id
        LEFT JOIN entretiens et ON et.candidature_id = ca.id
        WHERE ca.candidat_id = %s
        ORDER BY ca.created_at DESC;
    """, (candidat_id,))
    candidatures = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('candidat/mes_candidatures.html', candidatures=candidatures)


@app.route('/candidat/notifications')
def candidat_notifications():
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT titre, message, type, lue, created_at FROM notifications
        WHERE candidat_id = %s
        ORDER BY created_at DESC;
    """, (candidat_id,))
    notifications = cur.fetchall()
    cur.execute(
        "UPDATE notifications SET lue = TRUE WHERE candidat_id = %s AND lue = FALSE;",
        (candidat_id,)
    )
    conn.commit()
    cur.close()
    conn.close()
    return render_template('candidat/notifications.html', notifications=notifications)


@app.route('/candidat/entretiens')
def candidat_entretiens():
    r = candidat_auth_required()
    if r: return r

    candidat_id = session['candidat_id']
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT et.date_heure, et.duree_min, et.type, et.lien_visio, et.lieu,
               et.statut, et.notes,
               o.titre AS offre_titre,
               e.nom   AS entreprise_nom
        FROM entretiens et
        JOIN candidatures ca ON et.candidature_id = ca.id
        JOIN offres       o  ON ca.offre_id       = o.id
        LEFT JOIN entreprises e ON o.entreprise_id = e.id
        WHERE ca.candidat_id = %s
        ORDER BY et.date_heure DESC;
    """, (candidat_id,))
    entretiens = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('candidat/entretiens.html', entretiens=entretiens)


@app.route('/candidat/logout')
def candidat_logout():
    session.pop('candidat_id', None)
    session.pop('candidat_nom', None)
    session.pop('candidat_prenom', None)
    return redirect(url_for('login') + '?role=candidat')


@app.route('/candidat/register')
def candidat_register():
    return redirect(url_for('register') + '?role=candidat')


# ══════════════════════════════════════════════════════════════
#   DEBUG
# ══════════════════════════════════════════════════════════════
@app.route('/debug-entreprises')
def debug_entreprises():
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id, nom, email FROM entreprises ORDER BY id;")
    rows = cur.fetchall()
    cur.close(); conn.close()
    return str(rows)


# ══════════════════════════════════════════════════════════════
#   POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, port=5000)
