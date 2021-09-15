mkdir -p ./checkpoints/
cd ./checkpoints/
curl --cookie zenodo-cookies.txt "https://zenodo.org/record/4751737/files/DeepLIIF_Latest_Model.zip?download=1" --output DeepLIIF_Latest_Model.zip

sudo apt-get install unzip
unzip DeepLIIF_Latest_Model.zip

cd ..

echo "Downloaded pre-trained files:"
ls ./checkpoints/DeepLIIF_Latest_Model/
rm ./checkpoints/DeepLIIF_Latest_Model.zip
