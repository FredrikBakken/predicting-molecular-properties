<h1 align="center">Predicting Molecular Properties</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Prediction of the Scalar Coupling Constant of atom pairs in organic molecules from tabular data using
ensembling of Gradient Boosting Trees and Deep Neural Net methods. Molecular representation with distance
matrices and ACSF-representation with additional generated angle data was found to be paramount for accurate
predictions. Gradient Boosting Algorithms (XGB) and Deep Neural Nets were found to have comparable
accuracy (with Gradient Boosting generally better), and ensembling these methods with a strongly separated
configuration gave satisfactory results. The project used data from the Kaggle Competition champs-scalar-coupling.
</p>
<br> 

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
See the [Documentation Note](/documentation.pdf)

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites
What things you need to install the software and how to install them.

```
pip install anaconda
```

### Installing

Kaggle Download:

Downloads and extracts all necessary data source files from the Kaggle competition and organizes it into a data_sources directory,
ready to use.

```
python kaggle_download.py
```

### File Structure

The hierarchy should look like this:

    .
    ├── .gitignore
    ├── notebooks                         
    │     ├── ?         
    │     └── main.ipynb
    ├── utils                         
    │     ├── download.py         
    │     └── generate.py
    ├── models                         
    │     ├── nn
    |     │    ├── nn_model_1JHC.hdf5
    |     |    └── ...
    │     └── xgb
    |          ├── xgb_model_1JHC.hdf5
    |          └── ...
    ├── input                         
    │     ├── sources
    |     |    └── ...
    │     └── generated
    |          └── ...
    ├── submissions                         
    │     └── best_submission.csv
    |
    ├── LICENSE
    └── README.md


## 🎈 Usage <a name="usage"></a>
Run the notebook, tweak hyper-parameters, change up the data, see where it goes. 

## ⛏️ Built Using <a name = "built_using"></a>
- [Python 3.7](https://www.python.org/) 
- [Jupyter Notebook](https://jupyter.org/)
- [TensorFlow 2.0](https://www.tensorflow.org/) 
- [Keras](https://keras.io/)
- [Pandas](https://keras.io/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)

## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)
- Fredrik Bakken [@FredrikBakken](https://github.com/FredrikBakken) 

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- Hallvar Gisnås [@hallvagi](https://github.com/hallvagi)
- Lars Aurdal [@larsaurdal](https://github.com/larsaurdal)
- Dennis Christensen [@dennis-christensen](https://github.com/dennis-christensen)
- Niels Aase
- Kyle Lobo [@kylelobo](https://github.com/kylelobo)

### Kaggle Download
Downloads and extracts all necessary data source files from the Kaggle competition and organizes it into a data_sources directory,
ready to use.

`python kaggle_download.py`

### Molecule Visualizer
Visualize the molecules found in the dataset.

`python molecule_visualizer.py`
