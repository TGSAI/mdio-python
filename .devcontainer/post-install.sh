git config --global --add safe.directory `pwd`
poetry install --with dev --no-ansi
poetry shell