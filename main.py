#Import necessary libraries and functions
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

#Make Target and Predictors
X = pd.read_csv('houses.csv', index_col='Id')
print(X.isnull().sum())
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

#Engineer a "Cluster" feature
features = ["LotArea","TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea"]
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)

#Tune the Cluster feature
wcss = []
for i in range(3,14):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(3,14),wcss)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title('Elbow Method')
plt.show()

#Elbow WCSS Method shows 5 clusters
kmeans = KMeans(n_clusters = 5,n_init = 10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)

X_trainN, X_testN, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

low_cardinality_cols = [cname for cname in X_trainN.columns if X_trainN[cname].nunique() < 10 and 
                        X_trainN[cname].dtype == "object"]

numeric_cols = [cname for cname in X_trainN.columns if X_trainN[cname].dtype in ['int64', 'float64']]

#Removing columns with high cardinality, for simplicity in one-hot encoding
cols = low_cardinality_cols + numeric_cols
X_train = X_trainN[cols].copy() 
X_test = X_testN[cols].copy()

# Construct the model pipeline
numerical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, low_cardinality_cols)
    ])

model = RandomForestRegressor(random_state = 0)
Pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

#Implementation of GridSearchCV for hyperparameter tuning
param_grid = {
    'model__n_estimators':[100,200,300],
    'model__max_depth':[5,10,15],
    'model__min_samples_split':[2,5,10]
}

grid_search = GridSearchCV(estimator = Pipe, param_grid = param_grid, cv = 5, n_jobs = -1)
grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_
print(best_parameters)
print(best_score)

#Evaluation of model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("MAE: ",mean_absolute_error(y_pred, y_test))
r2 = r2_score(y_test, y_pred)
print(f"R-Squared score: {r2:.2f}")
acc = r2 * 100
acc = round(acc)
print(f"Accuracy: {acc}%")

# Output predictions
output = pd.DataFrame({'Id': X_test.index,
                      'Predicted Sale Price': y_pred})
print(output.head())
