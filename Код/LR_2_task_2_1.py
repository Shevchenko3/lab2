# імпорт необхідних бібліотек
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# шлях до вхідного файлу з даними
input_file = 'income_data.txt'

# створення списків для зберігання вхідних даних і міток класу
X = []  # тут зберігатимуться ознаки
y = []  # тут зберігатимуться мітки класів: <=50K або >50K
count_class1 = 0  # лічильник для класу <=50K
count_class2 = 0  # лічильник для класу >50K
max_datapoints = 25000  # обмеження на кількість прикладів кожного класу

# зчитування даних з текстового файлу построково
with open(input_file, 'r') as f:
    for line in f.readlines():
        # якщо обидва класи вже мають по 25000 прикладів — припинити зчитування
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        # якщо у рядку є пропущені значення — пропускаємо його
        if '?' in line:
            continue
        # поділ рядка на окремі атрибути за роздільником ", "
        data = line.strip().split(', ')
        # якщо клас - <=50K і ще не досягнуто межі 25000 — додаємо в список
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        # якщо клас - >50K і ще не досягнуто межі 25000 — додаємо в список
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# перетворення списку X у масив numpy
X = np.array(X)

# створення списку для кодувальників міток
label_encoder = []
# підготовка масиву тієї ж форми, що й X, для збереження закодованих значень
X_encoded = np.empty(X.shape)

# перетворення кожного стовпця в числовий формат
for i in range(X.shape[1]):
    # якщо значення в першому рядку є числом - залишаємо як є
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        # в іншому випадку створюємо LabelEncoder і перетворюємо текст у числа
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)

# поділ даних на ознаки (усі колонки, крім останньої) та мітки (останній стовпець)
X_features = X_encoded[:, :-1].astype(int)  # ознаки
y_labels = X_encoded[:, -1].astype(int)     # мітки класів

# поділ на тренувальний та тестовий набори у співвідношенні 80/20
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=5)

#SVC з поліноміальним ядром
classifier = SVC(kernel='poly', degree=2)
# навчання класифікатора на тренувальних даних
classifier.fit(X_train, y_train)

# прогнозування результатів для тестового набору
y_pred = classifier.predict(X_test)

# обчислення метрик якості класифікації
accuracy = accuracy_score(y_test, y_pred)                        # акуратність
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # точність
recall = recall_score(y_test, y_pred, average='weighted')        # повнота
f1 = f1_score(y_test, y_pred, average='weighted')                # F-мiра

# виведення результатів на екран
print(f"Поліноміальне ядро")
print(f"Акуратність: {round(accuracy * 100, 2)}%")
print(f"Точність: {round(precision * 100, 2)}%")
print(f"Повнота: {round(recall * 100, 2)}%")
print(f"F1 міра: {round(f1 * 100, 2)}%")

# створення нового прикладу для передбачення
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# створення порожнього масиву для збереження закодованих значень
input_data_encoded = np.empty(len(input_data))
count = 0

# кодування кожного атрибуту нової точки
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        input_data_encoded[i] = label_encoder[count].transform([item])[0]
        count += 1
input_data_encoded = input_data_encoded.reshape(1, -1)

# передбачення класу нової точки
predicted_class = classifier.predict(input_data_encoded)
# декодування числового класу у текстову мітку ('<=50K' або '>50K')
predicted_label = label_encoder[-1].inverse_transform(predicted_class)[0]

# виведення результату
print(f"Прогноз для вхідної точки: {predicted_label}")