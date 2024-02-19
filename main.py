import numpy as np

# Класс для построения отдельного дерева изоляции
class IsolationTree:
    def _init_(self, max_depth=10):
        # Инициализация параметров дерева
        self.max_depth = max_depth  # Максимальная глубина дерева
        self.threshold = None  # Порог для разделения данных
        self.feature_index = None  # Индекс признака для разделения
        self.left = None  # Левое поддерево
        self.right = None  # Правое поддерево
        self.depth = 0  # Текущая глубина узла

    # Обучение дерева на данных X
    def fit(self, X, depth=0):
        self.depth = depth
        # Условие остановки: достигнута максимальная глубина или в узле остался 1 элемент
        if depth == self.max_depth or len(X) <= 1:
            return self
        # Выбор случайного признака для разделения
        self.feature_index = np.random.randint(X.shape[1])
        # Выбор случайного порога для разделения данных
        min_val, max_val = np.min(X[:, self.feature_index]), np.max(X[:, self.feature_index])
        self.threshold = np.random.uniform(min_val, max_val)
        # Разделение данных на две группы
        left_mask = X[:, self.feature_index] < self.threshold
        right_mask = ~left_mask
        # Рекурсивное создание поддеревьев, если есть данные для разделения
        if np.any(left_mask) and np.any(right_mask):
            self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
            self.right = IsolationTree(self.max_depth).fit(X[right_mask], depth + 1)
        return self

    # Предсказание: вычисление средней глубины дохода до листа
    def predict(self, X):
        if self.threshold is None:
            return np.zeros(X.shape[0])
        mask = X[:, self.feature_index] < self.threshold
        if self.left is None or self.right is None:
            return np.ones(X.shape[0]) * self.depth
        else:
            left_scores = self.left.predict(X[mask])
            right_scores = self.right.predict(X[~mask])
            scores = np.zeros(X.shape[0])
            scores[mask] = left_scores
            scores[~mask] = right_scores
            return scores

# Класс для построения леса из деревьев изоляции
class IsolationForest:
    def _init_(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees  # Количество деревьев в лесу
        self.max_depth = max_depth  # Максимальная глубина деревьев
        self.trees = []  # Список деревьев

    # Обучение леса: создание и обучение n_trees деревьев
    def fit(self, X):
        self.trees = [IsolationTree(self.max_depth).fit(X) for _ in range(self.n_trees)]
        return self

    # Предсказание аномальности для данных X
    def predict(self, X):
        scores = np.mean([tree.predict(X) for tree in self.trees], axis=0)
        return scores

    # Вычисление оценки аномальности
    def anomaly_score(self, X):
        return 2 ** (-np.mean([tree.predict(X) for tree in self.trees], axis=0) / self.avg_path_length(len(X)))

    # Статический метод для вычисления средней длины пути в случайном дереве
    @staticmethod
    def avg_path_length(n):
        if n <= 1:
            return 1
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)




from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Генерируем искусственные данные с выбросами для тестирования
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X, outliers])

# Визуализация сгенерированных данных
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Регулярные данные')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Выбросы')
plt.title('Искусственные данные с выбросами')
plt.legend()
plt.show()

# Инициализация и обучение Isolation Forest
forest = IsolationForest(n_trees=100, max_depth=10)
forest.fit(X)

# Вычисление аномальности для каждой точки
scores = forest.anomaly_score(X)

# Визуализация результата работы Isolation Forest
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis')
plt.colorbar(label='Оценка аномальности')
plt.title('Результаты Isolation Forest')
plt.show()