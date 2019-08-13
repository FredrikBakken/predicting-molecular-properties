# Prediction Molecular Properties
Kaggle competition: https://www.kaggle.com/c/champs-scalar-coupling/overview

## Overview
Prediction of the Scalar Coupling Constant of atom pairs in organic molecules from tabular data using
ensembling of Gradient Boosting Trees and Deep Neural Net methods. Molecular representation with distance
matrices and ACSF-representation with additional generated angle data was found to be paramount for accurate
predictions. Gradient Boosting Algorithms (XGB) and Deep Neural Nets were found to have comparable
accuracy (with Gradient Boosting generally better), and ensembling these methods with a strongly separated
configuration gave satisfactory results. The project used data from the Kaggle Competition 'champs-scalar-coupling'.

### Hierarchy

    .
    ├── .git
    ├── .gitignore
    ├── .gitattributes
    ├── notebooks                         
    │   ├── ?         
    │   └── main.ipynb
    ├── utils                         
    │   ├── download.py         
    │   └── generate.py
    ├── models                         
    │   ├── nn
    |   |    └── ...
    │   └── xgb
    |        └── ...
    ├── input                         
    │   ├── sources
    |   |    └── ...
    │   └── generated
    |        └── ...
    ├── submissions                         
    │   └── best_submission.csv
    └── README.md
   
## Utilities
General utility files/scripts can be found within the `utils` directory.

### Kaggle Download
Downloads and extracts all necessary data source files from the Kaggle competition and organizes it into a data_sources directory,
ready to use.

`python kaggle_download.py`

### Molecule Visualizer
Visualize the molecules found in the dataset.

`python molecule_visualizer.py`
