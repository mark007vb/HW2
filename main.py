import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn import model_selection

# Load and preprocess data
data = pd.read_csv("flights.csv").dropna()
data = data[['origin', 'month', 'dep_delay']]
data['origin'] = data['origin'].astype('category').cat.codes
data.rename(columns={'dep_delay': 'target'}, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.4, random_state=8)

# DecisionTreeRegressor
parametrs = {'max_depth': range(2, 10, 1)}
grid = GridSearchCV(DecisionTreeRegressor(), parametrs, cv=5)
grid.fit(X_train, y_train)
dtr = DecisionTreeRegressor(max_depth=grid.best_params_['max_depth']).fit(X_train, y_train)

# PCA Transformation
pca = PCA(n_components=0.93).fit(X_train)
X_train_pca = pca.transform(X_train)

# DecisionTreeRegressor with PCA
grid.fit(X_train_pca, y_train)
dtr_pca = DecisionTreeRegressor(max_depth=grid.best_params_['max_depth']).fit(X_train_pca, y_train)

# Test models
models = {'DecisionTreeRegressor': dtr, 'DecisionTreeRegressorWithPCA': dtr_pca}

for name, model in models.items():
    if 'PCA' in name:
        X_tst = pca.transform(X_test)
    else:
        X_tst = X_test
    kfold = model_selection.KFold(n_splits=10, random_state=8, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_tst, y_test, cv=kfold, scoring='r2')
    print(f"{name}: {cv_results.mean()} ({cv_results.std()})")
