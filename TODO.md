# TODO

- [x] Ajouter la route Flask de suppression `/candidats/supprimer/<id>` (POST) dans `app.py`
  - [x] Contrôler l’accès (recruteur ou entreprise uniquement)
  - [x] Vérifier que le candidat est lié à au moins une offre appartenant à l’entreprise de l’utilisateur
  - [x] Supprimer le candidat dans la table `candidats` (FK => suppression `candidatures` / `cvs`)
  - [x] (Optionnel) Supprimer le fichier PDF dans `uploads/`
  - [x] Rediriger vers la bonne page après suppression

- [ ] Refactor “plateforme IA recrutement” (sans changer la stack)
  - [x] Étape 1 : créer modules backend (`db.py`, `auth.py`, `cv_extract.py`, `ia.py`)
  - [ ] Étape 2 : migrer helpers DB + extraction CV hors de `app.py`
  - [ ] Étape 3 : migrer module IA/scoring TF-IDF + cosine + spaCy hors de `app.py`
  - [ ] Étape 4 : migrer routes acteur (candidat/entreprise/recruteur) hors de `app.py`
  - [ ] Étape 5 : `app.py` devient point d’entrée (init Flask + enregistrement routes)
  - [ ] Étape 6 : harmoniser templates / urls si besoin
  - [ ] Étape 7 : valider scénario manuel (offre→upload CV→postuler→matching→suppression)

