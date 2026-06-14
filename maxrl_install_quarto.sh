set -e
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.9.38/quarto-1.9.38-linux-amd64.deb
apt-get install -y ./quarto-1.9.38-linux-amd64.deb
rm quarto-1.9.38-linux-amd64.deb