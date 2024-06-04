import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


X = pd.read_csv('houses.csv', index_col='Id')
print(X.isnull().sum())
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

X_trainN, X_testN, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

low_cardinality_cols = [cname for cname in X_trainN.columns if X_trainN[cname].nunique() < 10 and 
                        X_trainN[cname].dtype == "object"]

numeric_cols = [cname for cname in X_trainN.columns if X_trainN[cname].dtype in ['int64', 'float64']]

cols = low_cardinality_cols + numeric_cols
X_train = X_trainN[cols].copy() #Removing columns with high cardinality, for simplicity in one-hot encoding
X_test = X_testN[cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')
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

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("MAE: ",mean_absolute_error(y_pred, y_test))

output = pd.DataFrame({'Id': X_test.index,
                      'Predicted Sale Price': y_pred})
print(output.head())
