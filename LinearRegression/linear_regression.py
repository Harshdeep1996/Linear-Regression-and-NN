import sys
from numpy import *

# TODO: read what is line search? and how does it reduce the number of 
# iterations for gradient descent?


def compute_error(b, m, data):
    error = 0

    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]

        # This is the error,  because we are find the gap between each y
        # data point and y coordinate on the line and minimizing it.
        # Sum of squared distances/errors
        error += pow((y - (m * x + b)),2)

    # That is the total error, which has been averaged for all the points
    return error / float(len(data))


def step_gradient_descent(initial_b, initial_m, data, learning_rate):
    number_points = float(len(data))
    b_gradient = 0
    m_gradient = 0

    for j in range(0, len(data)):
        x = data[j, 0]
        y = data[j, 1]

        # Partial derivative of the cost function with 
        # respect to b and m and update it.
        b_gradient += -(2/number_points) * (y - (initial_m * x + initial_b))
        m_gradient += -(2/number_points) * x * (y - (initial_m * x + initial_b))

    # Learning rate used here, which tells how fast are we updating our
    # values
    initial_b = initial_b - (learning_rate * b_gradient)
    initial_m = initial_m - (learning_rate * m_gradient)

    return [initial_b, initial_m]


def gradient_descent_runner(initial_b, initial_m, data, learning_rate):
    b = initial_b
    m = initial_m
    iterations = 1000

    for i in range(iterations):
        b, m = step_gradient_descent(b, m, data, learning_rate)
    return [b, m]


def run():
    # Get data from a csv file with delimiter as ,(standard csv format)
    data = genfromtxt('data.csv', delimiter=',')

    # Define the hyperparameters
    # Learning Rate:  how fast our model is going to converge?
    #   Too small, then model will be too slow
    #   Too large,  then model will not converge
    learning_rate = 0.0001

    # As it is linear regression, and we have to find the best fit line
    # which decreases the sum of squared error, we can represent it by : 
    #                       y = mx + b
    initial_b = 0
    initial_m = 0

    error = compute_error(initial_b, initial_m, data)
    print('This is the initial error, : {}\n\n'.format(error))
    parameters = gradient_descent_runner(
        initial_b, initial_m, data, learning_rate
    )
    sys.stdout.write(str(parameters) + '\n\n')
    # print(
    #     'These are the values of b: {} and '
    #     'm: {} after running gradient descent, : {}\n\n'.format(
    #         parameters[0], parameters[1]
    #     )
    # )
    error = compute_error(parameters[0], parameters[1], data)
    print('This is the after error, : {}\n\n'.format(error))


if __name__ == '__main__':
    run()