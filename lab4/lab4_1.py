import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class EnsemblePerceptronStep:
    def __init__(self, num_perceptrons):
        self.perceptrons = [Perceptron(random_state=0) for _ in range(num_perceptrons)]

    def predict(self, xs):
        predictions = []
        for x in xs:
            prediction = round(np.mean(np.array([perceptron.predict([x]) for perceptron in self.perceptrons])))
            predictions.append(prediction)
        return predictions

    def train(self, xs, ys):
        for perceptron in self.perceptrons:
            perceptron.fit(xs, ys)


class EnsemblePerceptronSigmoid:
    def __init__(self, num_perceptrons):
        self.perceptrons = MLPClassifier(hidden_layer_sizes=[num_perceptrons], activation='logistic', solver="adam")

    def predict(self, xs):
        predictions = self.perceptrons.predict(xs)
        return predictions


    def train(self, xs, ys):
        self.perceptrons.fit(xs, ys)



print("Функция активации: 1 - ступенчатая  2 - сигмоидальная")
fucn_choice = str(input())

if fucn_choice == '1':
    ensemble = EnsemblePerceptronStep(num_perceptrons=1)

elif fucn_choice == '2':
    ensemble = EnsemblePerceptronSigmoid(num_perceptrons=7)
else:
    print("input error")
    quit()


(digits, labels) = load_digits(return_X_y=True)

ensemble.train(digits, labels)

prediction = ensemble.predict([digits[0]])
print("Predicted: " + str(prediction) + " is correct: " + str(prediction == labels[0]))

predictions = ensemble.predict(digits)

disp = ConfusionMatrixDisplay.from_predictions(labels, predictions)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()