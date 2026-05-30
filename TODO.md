# TODO

- [ ] Ajouter la route Flask de suppression `/candidats/supprimer/<id>` (POST) dans `app.py`
  - [ ] Vérifier session user_id
  - [ ] Supprimer le candidat dans la table `cvs` en s’assurant qu’il appartient aux offres de l’utilisateur
  - [ ] Rediriger vers `/candidats`
- [ ] (Optionnel) Supprimer le fichier PDF dans `uploads/` lors de la suppression

