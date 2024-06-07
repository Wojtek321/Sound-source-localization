from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from scirpts.data_preparation import X_train, Y_train, X_test, Y_test, scaler_y
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump
from utils import PLOTS_PATH, MODELS_PATH
from pprint import pprint
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def optimize_parameters(model, params):
    best_params = []
    for i in range(0, 5):
        grid_search = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)
        grid_search.fit(X_train, np.ravel(Y_train))
        best_params.append(grid_search.best_params_)
    return best_params



# decision tree
dt = DecisionTreeRegressor()

dt_params = {
    'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    'min_samples_split': [2, 4,  6, 8, 10, 11],
    'max_leaf_nodes': [9, 13, 15, 19, 23, 25, 26, 30, 35],
}

dt_best_params = optimize_parameters(model=dt, params=dt_params)
pprint(dt_best_params)

max_depth = round(mean([dict['max_depth'] for dict in dt_best_params]))
min_samples_split = round(mean([dict['min_samples_split'] for dict in dt_best_params]))
max_leaf_nodes = round(mean([dict['max_leaf_nodes'] for dict in dt_best_params]))

decision_tree_regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes)
decision_tree_regressor.fit(X_train, Y_train)


# random forest
rf = RandomForestRegressor()

rf_params = {
    'n_estimators': [1, 3, 5, 8, 16, 32, 64, 100],
    'min_samples_split': [2, 4, 5, 7, 9, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf_best_params = optimize_parameters(model=rf, params=rf_params)
pprint(rf_best_params)

n_estimators = round(mean([dict['n_estimators'] for dict in rf_best_params]))
min_samples_split = round(mean([dict['min_samples_split'] for dict in rf_best_params]))
min_samples_leaf = round(mean([dict['min_samples_leaf'] for dict in rf_best_params]))

random_forest_regressor = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
random_forest_regressor.fit(X_train, Y_train)


# kneighbors
kn = KNeighborsRegressor()

kn_params = {
    'n_neighbors': [2, 3, 4, 5, 6, 8, 10],
    'weights': ['uniform', 'distance'],
}

kn_best_params = optimize_parameters(model=kn, params=kn_params)
pprint(kn_best_params)

n_neighbors = round(mean([dict['n_neighbors'] for dict in kn_best_params]))
weights_arr = [dict['weights'] for dict in kn_best_params]
weights = max(set(weights_arr), key=weights_arr.count)

k_neighbors_regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
k_neighbors_regressor.fit(X_train, Y_train)



MODELS = [decision_tree_regressor, random_forest_regressor, k_neighbors_regressor]

data = {
    'Angle': scaler_y.inverse_transform(Y_test.reshape(-1, 1)).flatten(),
    MODELS[0]: [],
    MODELS[1]: [],
    MODELS[2]: [],
}
R2 = []
MAE = []
MSE = []


for model in MODELS:
    Y_pred = model.predict(X_test)

    R2.append(r2_score(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    MAE.append(scaler_y.inverse_transform(np.reshape(mae, (-1, 1)))[0][0])
    mse = mean_squared_error(Y_test, Y_pred)
    MSE.append(scaler_y.inverse_transform(np.reshape(mse, (-1, 1)))[0][0])

    Y_pred = scaler_y.inverse_transform(Y_pred.reshape(-1, 1))
    Y_pred = Y_pred.flatten()
    Y_pred = [round(x, 1) for x in Y_pred]
    data[model] = Y_pred


df = pd.DataFrame(data)
df.to_excel(os.path.join('..', 'assets', 'Models_results.xlsx'), index=False)


X_axis = np.arange(len(MODELS))
plt.clf()
plt.figure(figsize=(10, 8))
plt.title('Comparison of model performance metrics')
plt.bar(X_axis-0.3, R2, 0.3, label='R Square', color='royalblue')
plt.bar(X_axis, MAE, 0.3, label='MAE', color='magenta')
plt.bar(X_axis+0.3, MSE, 0.3, label='MSE', color='deepskyblue')
plt.xticks(X_axis, ['DecisionTreeRegressor()', 'RandomForestRegressor()', 'KNeighborsRegressor()'])
plt.legend()
plt.savefig(os.path.join(PLOTS_PATH, 'Model_evaluation.png'))


dump(decision_tree_regressor, os.path.join(MODELS_PATH, 'decision_tree_regressor.joblib'))
dump(random_forest_regressor, os.path.join(MODELS_PATH, 'random_forest_regressor.joblib'))
dump(k_neighbors_regressor, os.path.join(MODELS_PATH, 'k_neighbors_regressor.joblib'))
