import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

plt.style.use('ggplot')


iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target

model = LogisticRegression()

model.fit(x, y)

x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

pred = model.predict(np.c_[xx.ravel(), yy.ravel()])

pred = pred.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, pred, cmap=plt.cm.Paired)


plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)

plt.title('Tris Classification with Logistic Regression')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
