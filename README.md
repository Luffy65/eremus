Original author: https://github.com/SalvoCalcagno/eremus


# EREMUS

## Download dataset
1. Download data from Kaggle at https://www.kaggle.com/datasets/salvatorecalcagno/eremus
2. Unzip the archive. DO NOT place the folder in a path that contains accents or special characters.


## Clone this repository
clone into a directory of your choice.

## Configure
Move to eremus folder (main repository folder) and change the configuration.txt specifying the absolute path to EREMUSdataset. If you are using Windows remember to include escape characters in your path e.g. if your path is "C:\path\to\dataset" write "C:\\\path\\\to\\\dataset" (two backslashes instead of a single one)


## Install the package
You must be in the folder with setup.py. The command is `pip install -e .`. To simultaneously move into the right directory and begin installation, just write:
```
!cd eremus && pip install -e .
```
Or, if you want to clone this repository and install the package at the same time:
```
pip install -e git+https://github.com/Luffy65/eremus.git#egg=eremus
```

## View Code Docs
- switch to /docs directory
```
cd docs
```
- install sphinx
```
pip install sphinx
pip install sphinx-rtd-theme
```
- run the make file
```
make html
```
This will create a `build\html` directory. 
Go to `index.html` and start navigating with your favourite browser!

Enjoy!

