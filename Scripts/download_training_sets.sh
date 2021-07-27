mkdir -p ./Datasets/Dataset1/train/
cd ./Datasets/Dataset1/train/

curl --cookie zenodo-cookies.txt "https://zenodo.org/record/4751737/files/DeepLIIF_Training_Set.zip?download=1" --output DeepLIIF_Training_Set.zip
curl --cookie zenodo-cookies.txt "https://zenodo.org/record/4751737/files/BC-DeepLIIF_Training_Set.zip?download=1" --output BC-DeepLIIF_Training_Set.zip

sudo apt-get install unzip
unzip DeepLIIF_Training_Set.zip
unzip BC-DeepLIIF_Training_Set.zip

cd ../../../

rm ./Datasets/Dataset1/train/DeepLIIF_Training_Set.zip
rm ./Datasets/Dataset1/train/BC-DeepLIIF_Training_Set.zip


mkdir -p ./Datasets/Dataset1/val/
cd ./Datasets/Dataset1/val/

curl --cookie zenodo-cookies.txt "https://zenodo.org/record/4751737/files/DeepLIIF_Validation_Set.zip?download=1" --output DeepLIIF_Validation_Set.zip
curl --cookie zenodo-cookies.txt "https://zenodo.org/record/4751737/files/BC-DeepLIIF_Validation_Set.zip?download=1" --output BC-DeepLIIF_Validation_Set.zip

unzip DeepLIIF_Validation_Set.zip
unzip BC-DeepLIIF_Validation_Set.zip

cd ../../../

rm ./Datasets/Dataset1/val/DeepLIIF_Validation_Set.zip
rm ./Datasets/Dataset1/val/BC-DeepLIIF_Validation_Set.zip
