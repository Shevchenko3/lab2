from sklearn.datasets import load_iris
iris_dataset = load_iris()

# переглядаємо ключі даних
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

# опис набору даних
print(iris_dataset['DESCR'][:193] + "\n...")

# назви відповідей (класи ірисів)
print("Назви відповідей:{}".format(iris_dataset['target_names']))

# назви ознак
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))

# Переглядаємо тип та форму даних
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))


# Переглядаємо перші 5 рядків даних
print("Перші 5 прикладів:", iris_dataset['data'][:5])

# Мітки класів
print("Відповіді:\n{}".format(iris_dataset['target']))