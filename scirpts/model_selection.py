from scirpts.data_preparation import X_train, X_test, Y_train, Y_test, scaler_y
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from utils import PLOTS_PATH
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_model(model):
    model.fit(X_train, np.ravel(Y_train))
    Y_pred = model.predict(X_test)

    R2.append(r2_score(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    MAE.append(scaler_y.inverse_transform(np.reshape(mae, (-1, 1)))[0][0])
    mse = mean_squared_error(Y_test, Y_pred)
    MSE.append(scaler_y.inverse_transform(np.reshape(mse, (-1, 1)))[0][0])


MODELS = [DecisionTreeRegressor(), RandomForestRegressor(), SVR(), KNeighborsRegressor()]
R2 = []
MAE = []
MSE = []

for model in MODELS:
    evaluate_model(model)


X_axis = np.arange(len(MODELS))
plt.clf()
plt.figure(figsize=(10, 8))
plt.title('Comparison of model performance metrics')
plt.bar(X_axis-0.3, R2, 0.3, label = 'R Square', color='royalblue')
plt.bar(X_axis, MAE, 0.3, label = 'MAE', color='magenta')
plt.bar(X_axis+0.3, MSE, 0.3, label = 'MSE', color='deepskyblue')
plt.xticks(X_axis, MODELS)
plt.legend()
plt.savefig(os.path.join(PLOTS_PATH, 'Model_selection.png'))
