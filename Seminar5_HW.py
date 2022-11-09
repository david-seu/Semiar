import matplotlib.pyplot as plt
import numpy as np
import random


def non_convex_function(x):
    return x**4 + x**3 - 2*x**2 - 2*x


def non_convex_function_first_derivative(x):
    return 4*x**3 + 3*x**2 - 4*x - 2


def non_convex_function_image(domain_maximum, domain_minimum):
    image = []
    for x in range(domain_minimum, domain_maximum + 1):
        image.append(non_convex_function(x))
    return image


def convex_function(x):
    return x**2 - 4*x + 3


def convex_function_first_derivative(x):
    return 2*x - 4


def convex_function_image(domain_maximum, domain_minimum):
    image = []
    for x in range(domain_minimum, domain_maximum+1):
        image.append(convex_function(x))
    return image


def gradient_descent(start, gradient, learning_rate, max_iteration, tolerance=0.1):
    steps = [start]
    x = start
    for i in range(max_iteration):
        diff = learning_rate*gradient(x)
        if np.abs(diff) < tolerance:
            break
        x -= x - diff
        steps.append(x)
    return steps, x


def image_history(history, function):
    image_history = []
    for x in history:
        image_history.append(function(x))
    return image_history


if __name__ == '__main__':

    domain_of_function = [x for x in range(-500, 500)]
    image_convex_function = convex_function_image(domain_of_function[-1], domain_of_function[0])
    image_non_convex_function = non_convex_function_image(domain_of_function[-1], domain_of_function[0])
    intervals = ((-500, -400), (400, 500))
    figure, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2)
    figure.set_figwidth(20)
    figure.set_figheight(10)

    plt1.plot(domain_of_function, image_convex_function, color='black', linewidth='2')
    interval = random.choice(intervals)
    history, result = gradient_descent(random.choice(interval), convex_function_first_derivative, 0.1, 100)
    plt1.plot(history, image_history(history, convex_function), color='green', linestyle='dashed', marker='o', markerfacecolor='brown', linewidth=2)

    plt2.plot(domain_of_function, image_convex_function, color='black', linewidth='2')
    interval = random.choice(intervals)
    history, result = gradient_descent(random.choice(interval), convex_function_first_derivative, 5, 100)
    #plt2.plot(history, image_history(history, convex_function), color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', linewidth=2)

    plt3.plot(domain_of_function, image_convex_function, color='black', linewidth='2')
    interval = random.choice(intervals)
    history, result = gradient_descent(random.choice(interval), convex_function_first_derivative, 10, 100)
    #plt3.plot(history, image_history(history, convex_function), color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', linewidth=2)

    plt4.plot(domain_of_function, image_non_convex_function, color='black', linewidth='2')
    interval = random.choice(intervals)
    history, result = gradient_descent(random.choice(interval), non_convex_function_first_derivative, 0.000001, 100)
    #plt4.plot(history, image_history(history, non_convex_function), color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', linewidth=2)
    plt.show()
