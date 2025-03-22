import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('linear_regression.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

residuals = y - y_pred

residual_std = np.std(residuals)

outliers = np.abs(residuals) > 3 * residual_std

data['Outlier'] = outliers.astype(int)

data.to_csv('linear_regression_done.csv', index=False)