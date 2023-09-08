cp -f /root/.gitconfig_tmp /root/.gitconfig
cp -f /home/vscode/.gitconfig_tmp /home/vscode/.gitconfig
chmod 777 /home/vscode/.gitconfig
git config --global --add safe.directory `pwd`
# poetry install --with dev --no-ansi
# poetry shell
