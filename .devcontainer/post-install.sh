cp -f /root/.gitconfig_tmp /root/.gitconfig
cp -f /home/vscode/.gitconfig_tmp /home/vscode/.gitconfig
chmod 777 /home/vscode/.gitconfig
git config --global --add safe.directory `pwd`
# Enable ssh with github for git push
ssh-keygen -f "/root/.ssh/known_hosts" -R "github.com"
# poetry install --with dev --no-ansi
# poetry shell
