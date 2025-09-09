import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

W1 = np.random.uniform(-1,1,(input_neurons, hidden_neurons))
W2 = np.random.uniform(-1,1,(hidden_neurons, output_neurons))

lr = 0.5
for cnt in range(10000):
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)

    error = y - final_output

    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    W2 += hidden_output.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

print("Predicted Output after training:")
print(final_output)
