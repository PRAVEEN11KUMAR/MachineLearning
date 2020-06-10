import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]


regressor = linear_model.LinearRegression()
regressor.fit(diabetes_X, diabetes_y)

pred = regressor.predict(diabetes_X)

plt.style.use('ggplot')

plt.scatter(diabetes_X, diabetes_y, color='red')
plt.plot(diabetes_X, pred, color='blue', lw=3)

plt.xlabel('Predicted Values')
plt.ylabel('Real Values')
plt.title('Linear Regression Model')

plt.show()
