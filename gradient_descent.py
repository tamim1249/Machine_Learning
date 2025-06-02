import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.001
    cost_list = []

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        cost_list.append(cost)
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print(f"m {m_curr:.4f}, b {b_curr:.4f}, cost {cost:.4f}, iteration {i}")

    # Plotting after loop
    plt.plot(range(iterations), cost_list)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost vs Iteration in Gradient Descent")
    plt.grid(True)
    plt.show()

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
gradient_descent(x, y)
#02/06/2025
