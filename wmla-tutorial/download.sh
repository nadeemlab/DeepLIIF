
#!/bin/bash

cd $DATA_DIR
mkdir DeepLIIF_Datasets
cd DeepLIIF_Datasets
wget https://zenodo.org/record/4751737/files/DeepLIIF_Training_Set.zip
wget https://zenodo.org/record/4751737/files/DeepLIIF_Validation_Set.zip
wget https://zenodo.org/record/4751737/files/DeepLIIF_Testing_Set.zip

unzip DeepLIIF_Training_Set.zip
unzip DeepLIIF_Validation_Set.zip
unzip DeepLIIF_Testing_Set.zip
rm -rf DeepLIIF_*.zip

mv DeepLIIF_Training_Set train
mv DeepLIIF_Validation_Set val
mv DeepLIIF_Testing_Set test
