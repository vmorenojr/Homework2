# Predicting Schools Performance 
**Introduction to Data Science Bootcamp**

**Fundação Getulio Vargas - July 2019**

This contest aimed at predicting how a high schools will perform in the Brazilian national exam ENEM. 

Three datasets containing information about schools in São Paulo city were provided (they are in the Data folder). The file ENEM2015.csv contains the classification of the high schools in ENEM 2015, classifying each school in a scale from 0 to 4, where 4 corresponds to the best schools. The goal was to use information from the different sources in order to predict the performance of the students in the ENEM exam in 2015. 
 
In order to build the classification model, the following steps were taken:
 
   1.	I extracted from each data set information related to the schools listed in the file ENEM2015.csv. Each school had a unique identification number, which was used to find the school in each data set. 
   2.	I built a new data set (schools.csv), where the columns are a subset of the columns from the given data sets. 
   3.	I used the new data set to perform training and test. 

The team with the smallest test error won the contest.


 
