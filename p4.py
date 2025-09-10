import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDerivative(x):
    return x*(1-x)


x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[0],[0],[1]])

input_nodes=2
hidden_nodes=6
output_nodes=1

np.random.seed(1)
W1=np.random.uniform(-1,1,(input_nodes,hidden_nodes))
W2=np.random.uniform(-1,1,(hidden_nodes,output_nodes))

lr=0.1
for cnt in range(10000):
    hidden_in=np.dot(x,W1)
    hidden_out=sigmoid(hidden_in)

    final_in=np.dot(hidden_out,W2)
    final_out=sigmoid(final_in)

    error=y-final_out

    d_out=error*sigmoidDerivative(final_out)
    d_hidden=d_out.dot(W2.T)*sigmoidDerivative(hidden_out)

    W2+=hidden_out.T.dot(d_out)*lr
    W1+=x.T.dot(d_hidden)*lr

print("final o/p: ", np.round(final_out,3))
