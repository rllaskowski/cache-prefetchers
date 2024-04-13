#!/bin/bash

# Ustaw wiadomość commita
commit_message="Automatyczny commit"

# Ustaw nazwę gałęzi
branch_name="main"

# Obserwuj bieżący katalog i podkatalogi, ignorując experiments.ipynb
fswatch -o . | while read f
do
  git add .
  git reset experiments.ipynb
  git commit -m "$commit_message"
  git push origin $branch_name
  # Dodaj opóźnienie 30 sekund
  sleep 30
done
