import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class EnsemblePerceptronStep:
    def __init__(self, num_perceptrons):
        self.perceptrons = [Perceptron(random_state=0) for _ in range(num_perceptrons)]

    def predict(self, xs):
        prediction = np.mean(np.array([perceptron.predict(xs) for perceptron in self.perceptrons]))
        return 1 if prediction > 0.5 else 0

    def train(self, xs, ys):
        for perceptron in self.perceptrons:
            perceptron.fit(xs, ys)


class EnsemblePerceptronSigmoid:
    def __init__(self, num_perceptrons):
        # у нас один только слой - ансамбль
        # logistic = sigmoid
        # sgd = градиентный спуск
        self.perceptrons = MLPClassifier(hidden_layer_sizes=[num_perceptrons], activation='logistic', solver='sgd')

    def predict(self, xs):
        prediction = self.perceptrons.predict(xs)
        return prediction

    def train(self, xs, ys):
        self.perceptrons.fit(xs, ys)



#----------------------------------------Генерация-выборок---------------------------------------#

SAMPLE_SIZE = 500

colors_type1 = np.array([1] * SAMPLE_SIZE)  # 1 для оранжевых точек
colors_type2 = np.array([0] * SAMPLE_SIZE)  # 0 для синих точек

xs = []
ys = []

def add_noise(x, y, noise_level=0.1):
    x_noisy = x + np.random.normal(0, noise_level, size=len(x))
    y_noisy = y + np.random.normal(0, noise_level, size=len(y))
    return x_noisy, y_noisy

def ring_cluster():
    # Тип 1: Кольцевое распределение (Оранжевые точки)
    theta_outer = np.linspace(0, 2*np.pi, SAMPLE_SIZE)
    radius_outer = np.random.uniform(5, 10, size=SAMPLE_SIZE)
    x_outer = radius_outer * np.cos(theta_outer)
    y_outer = radius_outer * np.sin(theta_outer)
    x_outer, y_outer = add_noise(x_outer, y_outer)
    samples_type1 = np.column_stack([x_outer, y_outer])

    # Тип 2: Кластер (Синие точки)
    cluster_center = np.array([0, 0])
    cluster_radius = 2
    theta_inner = np.linspace(0, 2*np.pi, SAMPLE_SIZE)
    radius_inner = np.random.uniform(0, cluster_radius, size=SAMPLE_SIZE)
    x_inner = cluster_center[0] + radius_inner * np.cos(theta_inner)
    y_inner = cluster_center[1] + radius_inner * np.sin(theta_inner)
    x_inner, y_inner = add_noise(x_inner, y_inner)
    samples_type2 = np.column_stack([x_inner, y_inner])

    xs.append(np.concatenate([samples_type1, samples_type2]))
    ys.append(np.concatenate([colors_type1, colors_type2]))

    plt.scatter(samples_type1[:, 0], samples_type1[:, 1], color='orange', label='Type 1 (Ring)')
    plt.scatter(samples_type2[:, 0], samples_type2[:, 1], color='blue', label='Type 2 (Cluster)')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Samples')
    plt.legend()
    plt.draw()


def squares():
    # Тип 1: Квадраты с оранжевыми точками
    side_length = 2
    square1 = np.random.uniform(0, side_length, size=(SAMPLE_SIZE//2, 2)) + np.array([0, 1 * side_length])
    square2 = np.random.uniform(0, side_length, size=(SAMPLE_SIZE//2, 2)) + np.array([1 * side_length, 2 * side_length])
    square1[0], square1[1] = add_noise(square1[0], square1[1])
    square2[0], square2[1] = add_noise(square2[0], square2[1])
    samples_type1 = np.vstack([square1, square2])

    # Тип 2: Квадраты с синими точками
    square5 = np.random.uniform(0, side_length, size=(SAMPLE_SIZE//2, 2)) + np.array([1 * side_length])
    square6 = np.random.uniform(0, side_length, size=(SAMPLE_SIZE//2, 2)) + np.array([0, 2 * side_length])
    square5[0], square5[1] = add_noise(square5[0], square5[1])
    square6[0], square6[1] = add_noise(square6[0], square6[1])

    samples_type2 = np.vstack([square5, square6])

    xs.append(np.concatenate([samples_type1, samples_type2]))
    ys.append(np.concatenate([colors_type1, colors_type2]))

    # Отображение выборок
    plt.scatter(samples_type1[:, 0], samples_type1[:, 1], color='orange', label='Type 1 (Orange Squares)')
    plt.scatter(samples_type2[:, 0], samples_type2[:, 1], color='blue', label='Type 2 (Blue Squares)')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Samples')
    plt.legend()
    plt.draw()


def piles():
    # Тип 1: Кучка оранжевых точек
    cluster1 = np.random.normal([0, 0], 1, size=(500, 2))
    cluster1[0], cluster1[1] = add_noise(cluster1[0], cluster1[1])

    # Тип 2: Кучка синих точек
    cluster2 = np.random.normal([4, 4], 1, size=(500, 2))
    cluster2[0], cluster2[1] = add_noise(cluster2[0], cluster2[1])

    samples_type3 = np.vstack([cluster1, cluster2])

    xs.append(samples_type3)
    ys.append([1] * len(cluster1) + [0] * len(cluster2))

    # Отображение выборок
    plt.scatter(samples_type3[:, 0], samples_type3[:, 1], c=['orange' if i < SAMPLE_SIZE else 'blue' for i in range(SAMPLE_SIZE*2)])

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Samples')
    plt.draw()


def spirals():
    # Тип 1: Внешняя синяя спираль
    theta_outer = np.linspace(0, 6 * np.pi, SAMPLE_SIZE)
    radius_outer = np.linspace(0, 2.8, SAMPLE_SIZE)
    x_outer = radius_outer * np.cos(theta_outer)
    y_outer = radius_outer * np.sin(theta_outer)
    x_outer, y_outer = add_noise(x_outer, y_outer)
    samples_type1 = np.column_stack([x_outer, y_outer])

    # Тип 2: Внутренняя оранжевая спираль
    theta_inner = np.linspace(0, 6 * np.pi, SAMPLE_SIZE)
    radius_inner = np.linspace(0, 2, SAMPLE_SIZE)
    x_inner = radius_inner * np.cos(theta_inner)
    y_inner = radius_inner * np.sin(theta_inner)
    x_inner, y_inner = add_noise(x_inner, y_inner)
    samples_type2 = np.column_stack([x_inner, y_inner])

    xs.append(np.concatenate([samples_type1, samples_type2]))
    ys.append(np.concatenate([colors_type1, colors_type2]))

    # Отображение выборок
    plt.scatter(x_outer, y_outer, c='blue', label='Type 1 (Outer Spiral)')
    plt.scatter(x_inner, y_inner, c='orange', label='Type 2 (Inner Spiral)')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Samples')
    plt.legend()
    plt.draw()

#----------------------------------------Генерация-выборок---------------------------------------#


print("Выборка: 1 - кольцо и кластер  2 - квадраты  3 - кучки  4 - спирали")
xs_choice = str(input())

if xs_choice == '1':
    ring_cluster()
elif xs_choice == '2':
    squares()
elif xs_choice == '3':
    piles()
elif xs_choice == '4':
    spirals()
else:
    print("input error")
    quit()


print("Функция активации: 1 - ступенчатая  2 - сигмоидальная")
fucn_choice = str(input())

# xs_size = 2 у нас две характеристика - координаты (x, y)
if fucn_choice == '1':
    ensemble = EnsemblePerceptronStep(num_perceptrons=10)

elif fucn_choice == '2':
    ensemble = EnsemblePerceptronSigmoid(num_perceptrons=10)
else:
    print("input error")
    quit()
    
xs = xs[0]
ys = ys[0]

ensemble.train(xs, ys)
# print(xs[0])
# print(len(xs))
# print()
# print(ys)
# print(len(ys))
# quit()


plt.show()

prediction = ensemble.predict([xs[0]])
print("Predicted: " + str(prediction) + " is correct: " + str(prediction == ys[0]))


predictions = [ensemble.predict([x]) for x in xs]
conf_matrix = confusion_matrix(ys, predictions, normalize='true')

confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

confusion_matrix_display.plot()
plt.show()