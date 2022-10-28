# Machine Learning (CS-433) Project 1
## Table of contents
* [Project description](#Description)
* [Installation of the repository](#Installation)
* [Organisation of the repository](#Organisation)

## Project description
Higgs boson events at CERN are registered only through an indirect observation of the decay products of the collision of particles in the so-called LHC experiment. Machine learning techniques have been applied to a data-set of physical values obtained by scientists at CERN, to learn a model that could predict the presence of the Higgs boson during an experiment. In particular, the models used are the least squares regression with gradient descent, stochastic gradient descent and normal equations, ridge regression, and finally two methods for classification: logistic regression and its regularized version.

## Installation of the repository
To run this repository, please make sure that you have python installed.
To use the code, please open a terminal, go to your desired directory and clone this repository (make sure that [GIT](https://git-scm.com/) is installed on your device). You can use the HTTPS URL as follow: 
```
git clone https://github.com/TitBro/Project_1_ML.git
```
once the repository has been cloned, all the files will be visible and editable.

## Organisation of the repository
This repository consists of 2 directories and others files concerning data and implementation mainly: 

* imgs: directory containing all the plots and figures created for this project and included in the report
* py_files: directory containing all the .py files used by .ipynb files: data_processing.py, plots.py, data_loader.py, test_utils.py, helper.py

The other files are:

* run.ipynb: main file where data are loaded (with data_loader.py), visualized (with plots.py) and preprocessed (with data_processing.py), the functions are loaded (with implementations.py) and the models are run
* submission.ipynb: file were submsissions are created (data are loaded with helper.py): to be run only when run.ipynb has been run !
* tests.ipynb: file containing some basic tests for some implemented functions (with test_utils.py)
* implementation.py: file containing the functions of the predictive models to be implemented for this project
* train.csv: file of the training dataset
* test.csv: file of the test datatset, which submissions are made with
