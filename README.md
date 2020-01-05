# Graph-based-Recommendation-System
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


## Codes
### Prerequisites
- Python 3.6+
- Numpy
- Scipy
- Pandas

> #### Note for running implement_surprise.py, some other libraries are required. Please refer to the script for details.

### Introduction
1. Notebook : 
    - Recommender_MF.ipynb: recorded how we analyzed the user-item ratings matrix and how we implemented the regularized MF and the biased MF. The notebook is organized as follows:

        1. Load data, split the ratings matrix into training and testing set
        2. Statistics analysis
        3. Presentation of the MF methods used
        4. Grid search of the best parameters (This part has been reorganized in gs_reg_MF.py and gs_biased_MF.py)
        5. Compute the predictions
        6. Creation of csv file for the submission.
