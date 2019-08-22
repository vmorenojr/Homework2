# Predicting Schools Performance 
**Introduction to Data Science Bootcamp**

**Fundação Getulio Vargas - July 2019**

This contest aimed at predicting how a high schools will perform in the Brazilian national exam ENEM. 

Four datasets containing information about schools in São Paulo city were provided (they are in the Data folder). The file ENEM2015.csv contains the classification of the high schools in ENEM 2015, classifying each school in a scale from 0 to 4, where 4 corresponds to the best schools. The goal was to use information from the different sources in order to predict the performance of the students in the ENEM exam in 2015. 
 
In order to build the classification model, the following steps were taken:
 
   1.	I extracted from each data set information related to the schools listed in the file ENEM2015.csv. Each school had a unique identification number, which was used to find the school in each data set. 
   2.	I built a new data set (schools.csv), where the columns are a subset of the columns from the given data sets. 
   3.	I used the new data set to perform training and test for a variety of tree-based models. 

I used accuracy as the score to 
evaluate the prediction capacity of 
each model.

## Files
Two notebooks were created to perform the analysis:
   1. DataWrangling.ipynb: includes all the steps taken to clean and transform the data previous to the modeling stage; it generates a file (Data/Schools.csv) with the dataset to be used in the subsequent analyses.
   2. Modeling.ipynb: loads and runs the tree-based models; it includes the steps taken to train andtest each model, as well as the search for the optimum configurations of hyperparameters; its final output is a list of the prediction accuracy obtained with each model.

All data files are stores in the /Data folder.

## Requirements
To run the notebooks, it is necessary to have pandas, seaborn and scikit learn installed in a python 3.6 (or a posterior version) environment.

 
