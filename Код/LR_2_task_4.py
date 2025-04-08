# завантаження бібліотек
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
from sklearn.svm import LinearSVC

# завантаження набору даних
data = pd.read_csv("income_data.txt", delimiter=",", header=None)

# назви колонок
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# обробка пропущених значень (якщо є)
data.replace(' ?', np.nan, inplace=True)
data.dropna(inplace=True)

# перетворення категоріальних змінних у числові
data = pd.get_dummies(data)

# поділ на X (ознаки) і y (цільову змінну)
X = data.drop('income_ >50K', axis=1, errors='ignore')
y = data['income_ >50K'] if 'income_ >50K' in data.columns else data['income']

# поділ на тренувальну та тестову вибірки
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

# список моделей
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('LinearSVC', LinearSVC(max_iter=10000))) # замість SVC використовую LinearSVC, оскільки SVC займає занадто багато часу при навчанні


# оцінка моделей
results = []
names = []

print("Оцінка моделей (крос-валідація):\n")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# візуалізація порівняння
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів класифікації')
plt.ylabel('Точність')
plt.show()

# вибір найкращої моделі (CART)
best_model = DecisionTreeClassifier()
best_model.fit(X_train, y_train)

# прогнозування на тестовій вибірці
predictions = best_model.predict(X_validation)

# оцінка якості класифікації
print("\nТочність на тестовій вибірці:", accuracy_score(y_validation, predictions))
print("\nМатриця помилок:")
print(confusion_matrix(y_validation, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(y_validation, predictions))