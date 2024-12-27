# ML_Algo_Selection:

## How is machine learning best used in sleep science?

With multi-modal data: actigraphy (watch) data, survey data, etc... 
machine learning has proven to be a powerful tool for processing millions 
of data points and extracting meaningful insights.

## Which algorithm should be used?

Machine learning is roughly separable into 3 classes:
- Linear
- Non-Linear
- Deep Learning

In this project, we explore the limitations and trade-offs of each style of algorithm.

#

Here is a poster that Dr. Bradley Wheeler created! We presented together at 
**The Center for Sleep and Circadian
Science Research Day 2024.** :)

![Poster Preview](poster.png)

# Code Guide:
## data/

*All the data is private, but I have listed a few CSV files that may help get an idea 
of what the real data could have looked like.*

1. *dummy_data_all.csv*
2. *dummy_data_train.csv*
3. *dummy_data_validation.csv*
4. *dummy_data_test.csv*

**1** displays all the data we collected across the sleep studies. **2,3,4** represent 
that data is stratified across 3 sets, in our case it was about 70%, 15%, 15% between train,
validation, test.

Models are built from training data **(2)**, and are tuned based on validation data **(3)**.
The best performances on validation data are then tested **(4)** to understand how the 
model can perform on unseen data.

## preprocessing/

### preprocessing/statistics.py
- For linear and non-linear models, feature extraction was performed an then put into the models, rather than the raw data points. this script pulls PyTorch tensor data into sci-kit learn "friendly" data based on the following statistical variables:
   - Mean
   - Max
   - Kurtosis
   - Skewness
   - Shannon Entropy 
   - Standard Deviation

Deep learning models (we used an LSTM) do not require this extraction, as the tensors can
be directly input into the model.

### preprocessing/cleaning/
Code that created PyTorch tensors and compiled all survey data per study participant.

### preprocessing/stratification/
2 different stratification methods defined here. Ensuring that the gender and age
distribution was similar across datasets was crucial, and more importantly the outcome
distribution. Ensuring a roughly equal split between 0s and 1s for each set.

# machine_learning/

### machine_learning/tuning
PCA as well as the suite of models provided by scikit-learn.

### machine_learning/testing
PCA repeated including test data (**pca_all.py**)

**all_models.py** is a systematic way to train, validate, and test our final models:
- includes the final parameters decided from tuning.
- removes inputs one-by-one to determine the feature importance. 
- tracks the training time for each model.
- built to be dynamic (and save me time) as more models are added throughout the development process.



