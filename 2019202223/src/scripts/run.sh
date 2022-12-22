python -m pip install torch
python -m pip install metrics
python -m pip install sklearn
python -m pip install nltk
python -m pip install networkx
python -m pip install scipy
python -m pip install stanza
python -m pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
python -m dgl.backend.set_default_backend [pytorch]

python run_model.py --epochs=50 --hidden_lstm=100 --hidden_gcn=64 --batch_size=128 --gcn_layer=2