import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim
from sklearn.model_selection import KFold


# MLP = multilayer perceptron
class MLP(nn.Module):
    def __init__(self, num_layers, num_perceptrons, activation_func):
        super(MLP, self).__init__()
        act_f = None
        if activation_func == 'sigmoid':
            act_f = nn.Sigmoid()
        elif activation_func == 'tanh':
            act_f = nn.Tanh()
        elif activation_func == 'relu':
            act_f = nn.ReLU()

        self.layers = nn.Sequential()

        self.layers.add_module('input', nn.Linear(2, num_perceptrons))
        self.layers.add_module('input_act', act_f)

        for i in range(1, num_layers):
            self.layers.add_module('hidden_layer' + str(i), nn.Linear(num_perceptrons, num_perceptrons))
            self.layers.add_module('act' + str(i), act_f)

        self.layers.add_module('output', nn.Linear(num_perceptrons, 1))
        self.layers.add_module('output_act', nn.Sigmoid())

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.layers(x)
    
    def train_(self, xs, ys, epochs = 100): 
        xx = torch.from_numpy(xs).type(torch.FloatTensor)
        yy = torch.tensor(ys).type(torch.FloatTensor)   
        for epoch in range(0, epochs):
            self.train()
            self.optimizer.zero_grad()

            predictions = self(xx).squeeze()

            loss = self.loss_func(predictions, yy)

            loss.backward()

            self.optimizer.step()

            print(loss.item())
        
    def train_kfold(self, xs, ys, epochs = 100, splits = 2, batch = 10):
        xx = torch.from_numpy(xs).type(torch.FloatTensor)
        yy = torch.tensor(ys).type(torch.FloatTensor)

        dataset = torch.utils.data.TensorDataset(xx,yy)

        results = {}

        # shuffle = перетасовка
        kfold = KFold(n_splits=splits, shuffle=True)

        print('--------------------------------')

        for fold, (train_idxs, test_idxs) in enumerate(kfold.split(dataset)):

            # print(f'FOLD {fold}')
            # print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idxs)
            
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=batch, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch, sampler=test_subsampler)
            
            # Run the training loop for defined number of epochs
            for epoch in range(0, epochs):

                # print(f'Starting epoch {epoch+1}')

                current_loss = 0.0

                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):
                    
                    # Get inputs
                    inputs, targets = data
                    
                    # Zero the gradients
                    self.optimizer.zero_grad()
                    
                    # Perform forward pass
                    predictions = self(inputs).squeeze()
                    
                    # Compute loss
                    loss = self.loss_func(predictions, targets)
                    
                    # Perform backward pass
                    loss.backward()
                    
                    # Perform optimization
                    self.optimizer.step()
                    
            # print('Training process has finished. Saving trained model.')
            # print('Starting testing')
            
            # Saving the model
            save_path = f'models/model-fold-{fold+1}.pth'
            torch.save(self.state_dict(), save_path)

            # Evaluationfor this fold
            correct, total = 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):

                    # Get inputs
                    inputs, targets = data

                    # Generate outputs
                    outputs = self(inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                # Print accuracy
                # print('Accuracy for fold %d: %d %%' % (fold+1, 100.0 * correct / total))
                # print('--------------------------------')
                results[fold] = 100.0 * (correct / total)
            
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS')
        print('--------------------------------')
        max = 0
        max_accuracity_id = 0
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key + 1}: {value} %')
            sum += value
            if (value >= max):
                max = value
                max_accuracity_id = key

        print(f'Average: {sum/len(results.items())} %')

        PATH = "models/model-fold-" + str(max_accuracity_id + 1) + ".pth"

        # Load the most accurate model
        self.load_state_dict(torch.load(PATH))
        self.eval()

              

    def predict(self, xs):
        self.eval()
        with torch.no_grad():
            xx = torch.from_numpy(xs).type(torch.FloatTensor)
            predictions = self(xx)
        result = []
        for prediction in predictions:
            if prediction > 0.5:
                result.append(1)
            else :
                result.append(0)
        return result



#----------------------------------------Генерация-выборок---------------------------------------#

SAMPLE_SIZE = 500

torch.manual_seed(42)
np.random.seed(42)

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

matplotlib.use('TkAgg')

print("Выборка: 1 - кольцо и кластер  2 - квадраты  3 - кучки  4 - спирали")
# xs_choice = str(input())
xs_choice = '1'

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

# слоев от 2 до 4
# нейронов от 1 до 5

mlp = MLP(2, 3,'sigmoid')

xs = xs[0]
ys = ys[0]

mlp.train_kfold(xs, ys, epochs = 1000, splits=3)

plt.show()

predictions = mlp.predict(xs)
# print(predictions)
conf_matrix = confusion_matrix(ys, predictions, normalize='true')

confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

confusion_matrix_display.plot()
plt.show()

# 1 - круг и кластер
# sigmoid 2 3 epochs = 1000
# tanh 2 3 epochs = 1000
# relu 2 5 epochs = 400

# 2 - квадраты
# sigmoid 3 20 3500
# tanh 4 5 2000
# relu 3 5 3500

# 3 - кучки
# 2 2 epochs = 400

# 4 - спирали
# sigmoid ---
# tanh 5 50 7000
# relu 5 55 5000
