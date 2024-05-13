import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from sklearn.model_selection import KFold
import math
import sklearn.metrics as metrics
from torch import Tensor


class myTanh(nn.Tanh):
    def __init__(self):
        super(myTanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return 10 * torch.tanh(input)

# MLP = multilayer perceptron
class MLP(nn.Module):
    def __init__(self, num_layers, num_perceptrons, function_choice):
        super(MLP, self).__init__()
        if function_choice == 'a':
            act_f = myTanh()
        if function_choice == 'b':
            act_f = nn.Tanh()    

        self.layers = nn.Sequential()

        self.layers.add_module('input', nn.Linear(1, num_perceptrons))
        self.layers.add_module('input_act', act_f)

        for i in range(1, num_layers - 1):
            self.layers.add_module('hidden_layer' + str(i), nn.Linear(num_perceptrons, num_perceptrons))
            self.layers.add_module('act' + str(i), act_f)

        self.layers.add_module('output', nn.Linear(num_perceptrons, 1))
        self.layers.add_module('output_act', act_f)
        # add linear layer 1 1

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.layers(x)
    
    def train_(self, xs, ys, epochs = 1000): 
        xx = torch.tensor(xs).type(torch.FloatTensor)
        yy = torch.tensor(ys).type(torch.FloatTensor)   

        for epoch in range(0, epochs):
            self.train()
            self.optimizer.zero_grad()

            predictions = self(xx).squeeze()

            loss = self.loss_func(predictions, yy)

            loss.backward()

            self.optimizer.step()
        
            if (epoch % 50 == 0) :
                print("epoch :" + str(epoch) + "/" + str(epochs)) 
                print("MSE :" + str(loss.item()))

              

    def predict(self, xs):
        self.eval()
        with torch.no_grad():
            xx = torch.tensor(xs).type(torch.FloatTensor)
            predictions = self(xx)
        result = []
        for prediction in predictions:
            result.append(prediction.item())
        return result

# --------------------- генерация выборки ------------------#

class NegativeSize(Exception):
    pass

class NegataiveEpsilon(Exception):
    pass

class NonExistingChoice(Exception):
    pass

class NegataiveDegree(Exception):
    pass

N = 20
epsilon = 1e-2
print('N =', N)
print('epsilon =', epsilon, '\n')
if N < 0:
    print('size N cannot be negative')
    raise NegativeSize()
if epsilon < 0:
    print('epsilon cannot be negative')
    raise NegataiveEpsilon()

# generating sampling {x} where x is from [-1;1]
xs = 2 * np.random.random_sample(size = N) - 1

# a) epsilon distributed uniformly
# b) epsilon distributed normally
print("enter epsilon distribution: ")
print("a) epsilon distributed uniformly")
print("b) epsilon distributed normally")
print("enter epsilon distribution: ")
epsilon_distribution_choice = input()

if epsilon_distribution_choice == 'a' :
    epsilons = np.random.uniform(low = -epsilon, high = epsilon, size = N)
elif epsilon_distribution_choice == 'b' :
    epsilons = np.random.normal(loc = epsilon, scale = epsilon/3, size = N)
else :
    raise NonExistingChoice()

# a) f = a*x^3 + b*x^2 + c*x + d where a,b,c,d is from [-3;3]
# b) f = x * sin(2 * Pi * x)
print("enter f function: ")
print("a) f = a*x^3 + b*x^2 + c*x + d where a,b,c,d is from [-3;3]")
print("b) f = x * sin(2 * Pi * x)")
fucntion_f_choice = input()
t = np.arange(-1,1,0.001)
if fucntion_f_choice == 'a' :
    plt.title('f = a*x^3 + b*x^2 + c*x + d ')
    a,b,c,d = 3 * np.random.random_sample(size = 4) - 3
    print('a = ', a, '; b = ', b, '; c = ', c, '; d = ', d, '\n', sep = '')
    ys = list(map(lambda x : a * x**3 + b * x**2 + c * x + d, xs))
    plt.plot(t, list(map(lambda x : a * x**3 + b * x**2 + c * x + d, t)))
elif fucntion_f_choice == 'b' :
    plt.title('f = x * sin(2 * Pi * x)\n')
    ys = list(map(lambda x : x * math.sin(2 * math.pi * x), xs))
    plt.plot(t, list(map(lambda x : x * math.sin(2 * math.pi * x), t)))
else :
    raise NonExistingChoice()

ys_plus_epsilons = list(map(lambda y, epsilon : y + epsilon, ys, epsilons))

tmp = []
for i in range(N):
    tmp.append([xs[i]])
xs = tmp

tmp = []
for i in range(len(t)):
    tmp.append([t[i]])
t = tmp

# --------------------- генерация выборки ------------------#

# matplotlib.use('TkAgg')

mlp = MLP(10, 10, fucntion_f_choice)

mlp.train_(xs, ys_plus_epsilons, epochs=1000)

predictions = mlp.predict(xs)

print("mean square error = ")
print(metrics.mean_squared_error(ys_plus_epsilons, predictions))

plt.title('MLP regression')
plt.scatter(xs, ys_plus_epsilons)
plt.scatter(xs, predictions)
plt.plot(t, mlp.predict(t))

plt.show()
