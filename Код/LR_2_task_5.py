# імпорт бібліотек
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# завантаження датасету Iris
iris = load_iris()
X, y = iris.data, iris.target

# розбиття на навчальну і тестову вибірки (70% / 30%)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

# ініціалізація та навчання RidgeClassifier
# tol=1e-2 — допустима похибка, яка визначає, коли алгоритм повинен зупинити оптимізацію
# solver="sag" — стохастичний середньоградієнтний метод (Stochastic Average Gradient), який є ефективним при великій кількості даних
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)

# прогнозування на тестовій вибірці
ypred = clf.predict(Xtest)

# виведення показників якості класифікації
print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4))

# звіт про класифікацію
print('\nClassification Report:\n', metrics.classification_report(ytest, ypred))

# матриця плутанини
mat = confusion_matrix(ytest, ypred)

# візуалізація матриці плутанини
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Матриця плутанини RidgeClassifier')
plt.show()