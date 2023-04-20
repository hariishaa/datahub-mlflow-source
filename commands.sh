conda deactivate

python3 -m venv venv

source venv/bin/activate

python3 -m pip install -U pip

pip install mlflow

pip install acryl-datahub

python setup.py sdist bdist_wheel

pip install build

python -m build

pip install -e . -q

datahub ingest -c recipe.dhub.yaml

mlflow ui
