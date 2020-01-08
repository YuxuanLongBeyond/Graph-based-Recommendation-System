# Graph-based-Recommendation-System
## Structure of our repository
	.
	|-- README.md (here)
	|-- log (saved tensorboard log file)
	|-- ml-100k (dataset)
	|-- parameters (weights)
	|-- text (grid search text and ablation study results)
	|-- dataset.py
	|-- train.py
	|-- model.py
	|-- utils.py
	|-- loss.py

## Guideline to run our code:
### STEP 1: install required packages
pip: pip install -r requirements   
conda: conda env create -f environment. yaml

### STEP 2: run code
```console
sh run.sh -lr 0.01 0.02 -epochs 1000 2000 -hidden_dim 3 5 -side_hidden_dim 3 5 -dropout 0 0.1 0.2 -use_side_feature 0 1 -use_data_whitening 0 1 -use_laplacian_loss 0 1 -laplacian_loss_weight 0.05 0.1
```
