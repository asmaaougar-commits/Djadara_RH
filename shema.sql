
DROP TABLE IF EXISTS cvs, offres, users CASCADE;

CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    nom        VARCHAR(100)        NOT NULL,
    email      VARCHAR(100) UNIQUE NOT NULL,
    password   VARCHAR(255)        NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS offres (
    id          SERIAL PRIMARY KEY,
    titre       VARCHAR(255) NOT NULL,
    description TEXT,
    competences TEXT         NOT NULL,
    experience  VARCHAR(100),
    formation   VARCHAR(150),
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cvs (
    id          SERIAL PRIMARY KEY,
    nom_fichier VARCHAR(255) NOT NULL,
    contenu     TEXT,
    score       SMALLINT CHECK (score >= 0 AND score <= 100),
    offre_id    INTEGER REFERENCES offres(id) ON DELETE CASCADE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour optimiser les requêtes sans ORM
CREATE INDEX IF NOT EXISTS idx_offres_user ON offres(user_id);
CREATE INDEX IF NOT EXISTS idx_cvs_offre   ON cvs(offre_id);