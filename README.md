# Prediction of Real Estate Prices
## Project Overview
This project aims to predict sale prices of real estate using a dataset of known houses and its prices in the housing market. We implement a Random Forest Regression Algorithm to predict Real Estate prices based on features such as YearBuilt, HouseStyle, Utilities and many more.
In this project, I have implemented data preprocessing using pipeline which fills in missing data on both categorical and numerical columns using SimpleImputer and encodes categorical data using the OneHotEncoder function from ```scikit-learn```. I also implemented GridSearchCV for hyperparameter tuning to achieve the best estimator of models with varying parameters
## Dataset
The dataset used in this project is ```houses.csv```,which contains 1460 observations and 81 total features that determine house sale prices. During data preprocessing, some of these features will be removed for simplicity.
## Data Preprocessing
Several preprocessing steps were implemented on the dataset to allow for accurate predictions
1. Removing categorical columns with high cardinality, as implementation alongside OneHotEncoding will result in too much columns
2. In the pipeline, implement ```SimpleImputer``` and ```OneHotEncoding```
## Model Training and Hyperparameter Tuning
This project utilises GridSearchCV to perform hyperparameter tuning on 3 distinct parameters of RandomForestRegressor, mainly number of estimates, maximum depth of trees and the minimum number of samples required to split a node.
## Results
1. Best Parameters =
   ```
   # Python
   {'model__max_depth': 10, 'model__min_samples_split': 2, 'model__n_estimators': 300}
   ```
3. MAE: 17527.008006470056

## Conclusion
The Random Forest regression model equipped with optimal hyperparameters was able to demonstrate a strong ability in predicting house sale prices through its highly minimzed MAE (Mean Absolute Error). The steps I took during the Data Preprocessing stage proved to be beneficial to the model trained. With these results, it indicates that the model is relatively competent in estimating sale prices based on a large set of features.
