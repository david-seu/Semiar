import matplotlib.pyplot as plt
import numpy as np
import random


def generates_function_domain(min, max):
    domain = []
    number = min
    while number <= max:
        domain.append(number)
        number += 0.1
    return domain


def generates_function_image(domain, function):
    image = []
    for element in domain:
        image.append(function(element))
    return image


def non_convex_function(x):
    return (x-1)*x*(x+1)


def non_convex_function_gradient(x):
    return 3*x**2-1


def convex_function(x):
    return x**2 - 4*x + 3


def convex_function_gradient(x):
    return 2*x - 4


def gradient_descent(start, gradient, learning_rate, max_iteration, tolerance=1):
    steps = [start]
    x = start
    for i in range(max_iteration):
        diff = learning_rate*gradient(x)
        if np.abs(diff) < tolerance:
            break
        x = x - diff
        steps.append(x)
    return steps, x


if __name__ == '__main__':
    domain_of_function = generates_function_domain(-100, 100)
    image_convex_function = generates_function_image(domain_of_function, convex_function)
    image_non_convex_function = generates_function_image(domain_of_function, non_convex_function)
    figure, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2)
    figure.set_figwidth(20)
    figure.set_figheight(10)

    plt1.set_title('Convex function, learning rate = 0.1')
    plt1.set_xlabel('x in [-100,100]')
    plt1.set_ylabel('f(x)=x^2-4*x+3 ')
    plt1.plot(domain_of_function, image_convex_function, color='black', linewidth='2')
    history, result = gradient_descent(100, convex_function_gradient, 0.1, 100)
    plt1.plot(history, generates_function_image(history, convex_function), color='green', linestyle='dashed', marker='o', markerfacecolor='brown', linewidth=2)

    plt2.set_title('Convex function, learning rate = 0.5')
    plt2.set_xlabel('x in [-100,100]')
    plt2.set_ylabel('f(x)=x^2-4*x+3')
    plt2.plot(domain_of_function, image_convex_function, color='black', linewidth='2')
    history, result = gradient_descent(100, convex_function_gradient, 0.5, 100)
    plt2.plot(history, generates_function_image(history, convex_function), color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', linewidth=2)

    plt3.set_title('Convex function, learning rate = 0.8')
    plt3.set_xlabel('x in [-100,100]')
    plt3.set_ylabel('f(x)=x^2-4*x+3')
    plt3.plot(domain_of_function, image_convex_function, color='black', linewidth='2')
    history, result = gradient_descent(100, convex_function_gradient, 0.8, 100)
    plt3.plot(history, generates_function_image(history, convex_function), color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', linewidth=2)

    plt4.set_title('Non-convex function, learning rate = 0.002')
    plt4.set_xlabel('x in [-100,100]')
    plt4.set_ylabel('f(x)=(x-1)*x*(x+1)')
    plt4.plot(domain_of_function, image_non_convex_function, color='black', linewidth='2')
    history, result = gradient_descent(100, non_convex_function_gradient, 0.002, 100)
    plt4.plot(history, generates_function_image(history, non_convex_function), color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', linewidth=2)
    plt.show()
