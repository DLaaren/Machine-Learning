import numpy as np
import matplotlib.pyplot as plt
import math

class NegativeSize(Exception):
    pass

class NegataiveEpsilon(Exception):
    pass

class NonExistingChoice(Exception):
    pass

class NegataiveDegree(Exception):
    pass

N = 20
epsilon = 1e-1
print('N =', N)
print('epsilon =', epsilon, '\n')
if N < 0:
    print('size N cannot be negative')
    raise NegativeSize()
if epsilon < 0:
    print('epsilon cannot be negative')
    raise NegataiveEpsilon()

# generating sampling {x} where x is from [-1;1]
xs = (2 * np.random.random_sample(size = N) - 1)

# a) epsilon distributed uniformly
# b) epsilon distributed normally
epsilon_distribution_choice = input("enter epsilon distribution: ")
if epsilon_distribution_choice == 'a' :
    print('epsilon distributed uniformly\n')
    epsilons = np.random.uniform(low = -epsilon, high = epsilon, size = N)
elif epsilon_distribution_choice == 'b' :
    print('epsilon distributed normally\n')
    epsilons = np.random.normal(loc = epsilon, scale = epsilon/3, size = N)
else :
    raise NonExistingChoice()

# a) f = a*x^3 + b*x^2 + c*x + d where a,b,c,d is from [-3;3]
# b) f = x * sin(2 * Pi * x)
fucntion_f_choice = input("enter f function: ")
t = np.arange(-1,1,0.001)
if fucntion_f_choice == 'a' :
    plt.title('f = a*x^3 + b*x^2 + c*x + d where a,b,c,d is from [-3;3]')
    print('f = a*x^3 + b*x^2 + c*x + d where a,b,c,d is from [-3;3]')
    a,b,c,d = 3 * np.random.random_sample(size = 4) - 3
    print('a = ', a, '; b = ', b, '; c = ', c, '; d = ', d, '\n', sep = '')
    ys = list(map(lambda x : a * x**3 + b * x**2 + c * x + d, xs))
    plt.plot(t, list(map(lambda x : a * x**3 + b * x**2 + c * x + d, t)))
elif fucntion_f_choice == 'b' :
    plt.title('f = x * sin(2 * Pi * x)\n')
    print('f = x * sin(2 * Pi * x)\n')
    ys = list(map(lambda x : x * math.sin(2 * math.pi * x), xs))
    plt.plot(t, list(map(lambda x : x * math.sin(2 * math.pi * x), t)))
else :
    raise NonExistingChoice()

ys_plus_epsilons = list(map(lambda y, epsilon : y + epsilon, ys, epsilons))

degree = int(input('enter degree for poly regression: '))
if degree < 0:
    print('degree cannot be negative')
    raise NegataiveDegree()

plt.scatter(xs, ys_plus_epsilons)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.title('polynomial regression')
polynomial_regression = np.poly1d(np.polyfit(xs,ys_plus_epsilons, degree))
polyline = np.linspace(-1, 1, 1000)
plt.scatter(xs, ys_plus_epsilons)
plt.plot(polyline, polynomial_regression(polyline))
if fucntion_f_choice == 'a' :
    plt.plot(t, list(map(lambda x : a * x**3 + b * x**2 + c * x + d, t)))
elif fucntion_f_choice == 'b' :
    plt.plot(t, list(map(lambda x : x * math.sin(2 * math.pi * x), t)))
else :
    raise NonExistingChoice()
plt.ylim(-2,2)
plt.show()
