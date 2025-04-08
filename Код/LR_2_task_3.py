# Завантаження бібліотек
import pandas as pd
import numpy as np 
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# shape
print(dataset.shape)

# зріз даних head
print(dataset.head(20))

# стастичні зведення 
print(dataset.describe())

# розподіл за атрибутом class
print(dataset.groupby('class').size())

# діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2,2), 
sharex=False, sharey=False)
plt.show()

# гістограма розподілу атрибутів датасета
dataset.hist()
plt.show()

# матриця діаграм розсіювання
scatter_matrix(dataset)
plt.show()

# поділ датасету на навчальну та контрольну вибірки
array = dataset.values
# Вибір перших 4-х стовпців
X = array[:,0:4]
# Вибір 5-го стовпця
y = array[:,4]
# поділ X та y на навчальну та контрольну вибірки
X_train, X_validation, y_train, Y_validation = train_test_split(X, y, 
test_size=0.20, random_state=1)

# завантаження алгоритмів моделі
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# оцінка моделі на кожній ітерації
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# порівняння алгоритмів
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# створюємо прогноз на контрольній вибірці
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

# оцінюємо прогноз
# оцінка точності
print("Точність на контрольній вибірці:", accuracy_score(Y_validation, predictions))
# матриця помилок
print("Матриця помилок:")
print(confusion_matrix(Y_validation, predictions))
# звіт про класифікацію
print("Звіт про класифікацію:")
print(classification_report(Y_validation, predictions))

X_new = np.array([[4.7, 3.0, 1.6, 0.2]])
print('Форма масиву X_new:', X_new.shape)

predictions = model.predict(X_new)
print('Прогноз {для X_new:}', predictions)