#!/usr/bin/env bash

VENVNAME=NLP

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# install ipython kernel, so we can use JHub
pip install ipython
pip install jupyter
python -m ipykernel install --user --name=$VENVNAME

# if there is a requirement .txt file
test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
