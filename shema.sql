-- ============================================================
--  Djadara RH  —  Schéma PostgreSQL v3.0
--  Plateforme intelligente de recrutement
--  Acteurs :
--    • Candidat   → inscription, dépôt CV, candidature, suivi
--    • Entreprise → inscription, offres, candidats, matching IA, entretiens
--    • Admin      → connexion seule, gestion globale + statistiques
-- ============================================================

-- Extensions utiles
CREATE EXTENSION IF NOT EXISTS "unaccent";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- recherche floue

-- ============================================================
--  Nettoyage
-- ============================================================
DROP TABLE IF EXISTS
    entretiens, notifications, candidatures, cvs,
    offres, candidats, entreprises, admins
CASCADE;

-- ============================================================
--  1. ADMINS PLATEFORME  (connexion seule, gestion globale)
-- ============================================================
CREATE TABLE admins (
    id              SERIAL PRIMARY KEY,
    nom             VARCHAR(100) NOT NULL,
    prenom          VARCHAR(100) NOT NULL,
    email           VARCHAR(100) UNIQUE NOT NULL,
    password        VARCHAR(255) NOT NULL,
    reset_code      VARCHAR(6),
    reset_expiry    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
--  2. ENTREPRISES  (inscription / connexion / gestion offres)
-- ============================================================
CREATE TABLE entreprises (
    id          SERIAL PRIMARY KEY,
    nom         VARCHAR(150) NOT NULL,
    secteur     VARCHAR(100),
    adresse     TEXT,
    email       VARCHAR(100) UNIQUE NOT NULL,
    telephone   VARCHAR(30),
    logo_url    VARCHAR(255),
    -- Auth
    password        VARCHAR(255) NOT NULL,
    is_verified     BOOLEAN      DEFAULT FALSE,
    verif_token     VARCHAR(10),
    reset_code      VARCHAR(6),
    reset_expiry    TIMESTAMP,
    is_active       BOOLEAN      DEFAULT TRUE,   -- géré par l'admin plateforme
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
--  3. CANDIDATS
-- ============================================================
CREATE TABLE candidats (
    id                  SERIAL PRIMARY KEY,
    nom                 VARCHAR(100) NOT NULL,
    prenom              VARCHAR(100) NOT NULL,
    email               VARCHAR(100) UNIQUE NOT NULL,
    telephone           VARCHAR(30),
    password            VARCHAR(255) NOT NULL,
    is_verified         BOOLEAN DEFAULT FALSE,
    verif_token         VARCHAR(10),
    reset_code          VARCHAR(6),
    reset_expiry        TIMESTAMP,
    is_active           BOOLEAN DEFAULT TRUE,    -- géré par l'admin plateforme
    -- Profil structuré (extrait du CV + saisie manuelle)
    niveau_etude        VARCHAR(100),
    annees_experience   SMALLINT DEFAULT 0,
    titre_poste         VARCHAR(150),
    resume              TEXT,
    competences         JSONB DEFAULT '[]',
    langues             JSONB DEFAULT '[]',
    certifications      JSONB DEFAULT '[]',
    -- CV principal
    cv_fichier          VARCHAR(255),
    cv_texte_brut       TEXT,
    cv_analyse_at       TIMESTAMP,
    disponible          BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
--  4. OFFRES D'EMPLOI  (créées par l'entreprise)
-- ============================================================
CREATE TABLE offres (
    id                  SERIAL PRIMARY KEY,
    titre               VARCHAR(255) NOT NULL,
    description         TEXT,
    competences         TEXT NOT NULL,
    experience_min      SMALLINT DEFAULT 0,
    niveau_etude        VARCHAR(100),
    langues_requises    TEXT DEFAULT '',
    certifications      TEXT DEFAULT '',
    -- Pondérations matching IA
    poids_competences   SMALLINT DEFAULT 40,
    poids_experience    SMALLINT DEFAULT 25,
    poids_formation     SMALLINT DEFAULT 20,
    poids_langues       SMALLINT DEFAULT 10,
    poids_certif        SMALLINT DEFAULT 5,
    localisation        VARCHAR(150),
    type_contrat        VARCHAR(50),
    salaire             VARCHAR(80),
    statut              VARCHAR(20) DEFAULT 'active',  -- active / fermée / draft
    entreprise_id       INTEGER REFERENCES entreprises(id) ON DELETE CASCADE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
--  5. CVS  (analyses IA liées à un candidat / une offre)
-- ============================================================
CREATE TABLE cvs (
    id                      SERIAL PRIMARY KEY,
    candidat_id             INTEGER REFERENCES candidats(id) ON DELETE CASCADE,
    nom_fichier             VARCHAR(255) NOT NULL,
    score_global            SMALLINT DEFAULT 0  CHECK (score_global  BETWEEN 0 AND 100),
    score_competences       SMALLINT DEFAULT 0  CHECK (score_competences BETWEEN 0 AND 100),
    score_experience        SMALLINT DEFAULT 0  CHECK (score_experience  BETWEEN 0 AND 100),
    score_formation         SMALLINT DEFAULT 0  CHECK (score_formation   BETWEEN 0 AND 100),
    score_langues           SMALLINT DEFAULT 0  CHECK (score_langues     BETWEEN 0 AND 100),
    score_certifications    SMALLINT DEFAULT 0  CHECK (score_certifications BETWEEN 0 AND 100),
    competences_trouvees    JSONB DEFAULT '[]',
    competences_manquantes  JSONB DEFAULT '[]',
    points_forts            JSONB DEFAULT '[]',
    points_ameliorer        JSONB DEFAULT '[]',
    langues_trouvees        JSONB DEFAULT '[]',
    certifications_trouvees JSONB DEFAULT '[]',
    niveau_qualitatif       VARCHAR(50) DEFAULT 'À vérifier',
    offre_id                INTEGER REFERENCES offres(id) ON DELETE SET NULL,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT cvs_nom_fichier_offre_unique UNIQUE (nom_fichier, offre_id)
);

-- ============================================================
--  6. CANDIDATURES  (Candidat ↔ Offre + suivi)
-- ============================================================
CREATE TABLE candidatures (
    id              SERIAL PRIMARY KEY,
    candidat_id     INTEGER REFERENCES candidats(id) ON DELETE CASCADE,
    offre_id        INTEGER REFERENCES offres(id)    ON DELETE CASCADE,
    cv_id           INTEGER REFERENCES cvs(id)       ON DELETE SET NULL,
    score_global        SMALLINT DEFAULT 0,
    score_competences   SMALLINT DEFAULT 0,
    score_experience    SMALLINT DEFAULT 0,
    score_formation     SMALLINT DEFAULT 0,
    score_langues       SMALLINT DEFAULT 0,
    score_certifications SMALLINT DEFAULT 0,
    niveau              VARCHAR(50) DEFAULT 'À évaluer',
    rang                SMALLINT DEFAULT 0,
    -- Workflow : soumise → vue → présélectionnée → entretien → offre → refusée
    statut          VARCHAR(30) DEFAULT 'soumise',
    note_recruteur  TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidat_id, offre_id)
);

-- ============================================================
--  7. ENTRETIENS  (gérés directement par l'entreprise)
-- ============================================================
CREATE TABLE entretiens (
    id              SERIAL PRIMARY KEY,
    candidature_id  INTEGER REFERENCES candidatures(id) ON DELETE CASCADE,
    entreprise_id   INTEGER REFERENCES entreprises(id)  ON DELETE CASCADE,
    date_heure      TIMESTAMP NOT NULL,
    duree_min       SMALLINT DEFAULT 60,
    type            VARCHAR(30) DEFAULT 'présentiel',  -- présentiel / visio / téléphone
    lien_visio      VARCHAR(255),
    lieu            VARCHAR(255),
    statut          VARCHAR(20) DEFAULT 'planifié',    -- planifié / confirmé / annulé / terminé
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
--  8. NOTIFICATIONS  (candidat OU entreprise OU admin)
-- ============================================================
CREATE TABLE notifications (
    id              SERIAL PRIMARY KEY,
    candidat_id     INTEGER REFERENCES candidats(id)    ON DELETE CASCADE,
    entreprise_id   INTEGER REFERENCES entreprises(id)  ON DELETE CASCADE,
    admin_id        INTEGER REFERENCES admins(id)       ON DELETE CASCADE,
    titre           VARCHAR(200) NOT NULL,
    message         TEXT,
    type            VARCHAR(30) DEFAULT 'info',   -- info / success / warning / entretien
    lue             BOOLEAN DEFAULT FALSE,
    lien            VARCHAR(255),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT notif_un_destinataire CHECK (
        (candidat_id IS NOT NULL)::int
      + (entreprise_id IS NOT NULL)::int
      + (admin_id IS NOT NULL)::int = 1
    )
);

-- ============================================================
--  INDEX
-- ============================================================
CREATE INDEX idx_offres_entreprise      ON offres(entreprise_id);
CREATE INDEX idx_offres_statut          ON offres(statut);
CREATE INDEX idx_cvs_candidat           ON cvs(candidat_id);
CREATE INDEX idx_cvs_offre              ON cvs(offre_id);
CREATE INDEX idx_candidatures_candidat  ON candidatures(candidat_id);
CREATE INDEX idx_candidatures_offre     ON candidatures(offre_id);
CREATE INDEX idx_candidatures_statut    ON candidatures(statut);
CREATE INDEX idx_entretiens_candidature ON entretiens(candidature_id);
CREATE INDEX idx_entretiens_entreprise  ON entretiens(entreprise_id);
CREATE INDEX idx_notifs_candidat        ON notifications(candidat_id)   WHERE candidat_id   IS NOT NULL;
CREATE INDEX idx_notifs_entreprise      ON notifications(entreprise_id) WHERE entreprise_id IS NOT NULL;
CREATE INDEX idx_notifs_admin           ON notifications(admin_id)      WHERE admin_id      IS NOT NULL;

CREATE INDEX idx_candidats_competences  ON candidats USING GIN (competences);
CREATE INDEX idx_candidats_langues      ON candidats USING GIN (langues);

-- ============================================================
--  VUES
-- ============================================================

-- Classement candidats par offre (matching IA)
CREATE OR REPLACE VIEW v_matching AS
SELECT
    ca.offre_id,
    o.titre          AS offre_titre,
    o.entreprise_id,
    cand.id          AS candidat_id,
    cand.nom         AS candidat_nom,
    cand.prenom      AS candidat_prenom,
    cand.email       AS candidat_email,
    ca.id            AS candidature_id,
    ca.score_global,
    ca.score_competences,
    ca.score_experience,
    ca.score_formation,
    ca.score_langues,
    ca.niveau,
    ca.statut        AS statut_candidature,
    ca.rang,
    ca.created_at    AS date_candidature,
    RANK() OVER (PARTITION BY ca.offre_id ORDER BY ca.score_global DESC) AS classement
FROM candidatures ca
JOIN candidats cand ON ca.candidat_id = cand.id
JOIN offres    o    ON ca.offre_id    = o.id
LEFT JOIN cvs  c    ON ca.cv_id       = c.id;

-- Statistiques par entreprise
CREATE OR REPLACE VIEW v_stats_entreprise AS
SELECT
    e.id                                        AS entreprise_id,
    e.nom                                       AS entreprise_nom,
    COUNT(DISTINCT o.id)                        AS nb_offres,
    COUNT(DISTINCT ca.id)                       AS nb_candidatures,
    ROUND(AVG(ca.score_global), 1)              AS score_moyen,
    COUNT(DISTINCT CASE WHEN o.statut='active' THEN o.id END) AS offres_actives
FROM entreprises e
LEFT JOIN offres       o  ON o.entreprise_id = e.id
LEFT JOIN candidatures ca ON ca.offre_id     = o.id
GROUP BY e.id, e.nom;

-- Statistiques globales plateforme (pour l'admin)
CREATE OR REPLACE VIEW v_stats_plateforme AS
SELECT
    (SELECT COUNT(*) FROM candidats)                              AS nb_candidats,
    (SELECT COUNT(*) FROM candidats   WHERE is_active)            AS nb_candidats_actifs,
    (SELECT COUNT(*) FROM entreprises)                            AS nb_entreprises,
    (SELECT COUNT(*) FROM entreprises WHERE is_active)            AS nb_entreprises_actives,
    (SELECT COUNT(*) FROM offres)                                 AS nb_offres,
    (SELECT COUNT(*) FROM offres      WHERE statut='active')      AS nb_offres_actives,
    (SELECT COUNT(*) FROM candidatures)                           AS nb_candidatures,
    (SELECT COUNT(*) FROM entretiens)                             AS nb_entretiens,
    (SELECT ROUND(AVG(score_global),1) FROM candidatures)         AS score_moyen_global;
