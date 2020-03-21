import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn
from torch.autograd import Variable
from pyDOE import *

# This program uses neural network to approximate the solution to the inhomogeneous Helmholtz equation
# \nabla^2*u(x) - lambda*u(x) = f(x) with range [upper_x, lower_x].

# -----------------------------define functions-----------------------------
def f(x):
    y = -(np.pi ** 2 + 1) * np.sin(np.pi * x)
    return y


def u(x):
    y = np.sin(np.pi * x)
    return y


# --------------------------hyperparameters--------------------------------
# the interval on which f(x) and u(x) are defined
upper_x = 1
lower_x = -1

# the coefficient before u(x) in the equation
lambda_coeff = 1

# the collection of number of training points to choose from
n_range = [2, 5, 20, 50, 100, 200, 500]

lr = 0.001
hidden_nodes = 50


# ----------------------------neural network---------------------------------
class ODESolver(torch.nn.Module):

    def __init__(self, x, y, hidden, lr, lambda_coeff=1):
        super(ODESolver, self).__init__()
        # parameters
        self.N = len(y)
        self.lr = lr
        self.inputSize = x.shape[1]
        self.outputSize = y.shape[1]
        self.hiddenSize = hidden
        self.lambda_coeff = lambda_coeff

        # data preparation
        self.x = Variable(torch.Tensor(x.astype(float)), requires_grad=True)
        self.y = Variable(torch.Tensor(y.astype(float)))

        self.l1 = torch.nn.Linear(self.inputSize, self.hiddenSize)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=1.0)
        self.l2 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)
        self.l3 = torch.nn.Linear(self.hiddenSize, self.outputSize)
        torch.nn.init.xavier_uniform_(self.l3.weight, gain=1.0)
        self.Tanh = torch.nn.Tanh()

    def forward(self, X):
        output = self.l1(X)
        output = self.Tanh(output)
        output = self.l2(output)
        output = self.Tanh(output)
        output = self.l3(output)
        return output

    def composite_mse_loss(self):
        boundaryX = torch.Tensor([[-1.], [1.]])
        boundaryY = torch.Tensor([[0.], [0.]])
        nu = self(boundaryX) - boundaryY

        ypred = self(self.x)
        u_x, = torch.autograd.grad(outputs=ypred, inputs=self.x,
                                   grad_outputs=torch.ones_like(self.x), create_graph=True)
        u_xx, = torch.autograd.grad(outputs=u_x, inputs=self.x,
                                    grad_outputs=torch.ones_like(self.x), create_graph=True)
        nf = u_xx - self.lambda_coeff * ypred - torch.from_numpy(f(x))

        out = 1 / 2 * torch.mm(nu.T, nu) + 1 / self.N * torch.mm(nf.T, nf)
        return out

    def train(self, tolerance=-1):

        if tolerance == -1:
            tolerance = self.lr / 2
        initial_loss = self.composite_mse_loss().detach().cpu().clone().numpy().astype(float).item()
        loss_hist = np.array([initial_loss * 20, initial_loss * 10])
        epoch = 0
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        while (epoch < 10000) & (np.abs(loss_hist[1] - loss_hist[0]) > tolerance):

            epoch += 1
            ypred = self(self.x)

            # Check each 100 epoch for loss reduction
            loss = self.composite_mse_loss()
            if epoch % 100 == 0:
                loss_hist[0] = loss_hist[1]
                loss_hist[1] = self.composite_mse_loss().detach().cpu().clone().numpy().astype(float).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss

            if epoch >= 10000:
                warnings.warn('Exceeded epoch limit, may not reach optimal point')

        accuracy = torch.norm(ypred - self.y) / self.N
        print(f'N = {self.N}, L2 Error: {accuracy:.4f}')
        return accuracy

    def predict(self, x):
        x = torch.Tensor(x)
        y = self(x).cpu().data.numpy().astype(float)
        return y


def generate_data(n=2, upper=1, lower=-1):
    x = (upper - lower) * lhs(1, n) + lower
    return x, f(x)


# ------------------------------main program------------------------------
np.random.seed(2000)
warnings.simplefilter('always', UserWarning)

# generate analytical data pairs
xtest = (upper_x - lower_x) * lhs(1, 2000) + lower_x
ytest = u(xtest)

# begin training, plotting and comparing models using different training point numbers
acc_hist = []
plt.figure(1)
for i in n_range:
    x, y = generate_data(i, upper_x, lower_x)
    model = ODESolver(x, y, hidden=hidden_nodes, lr=lr, lambda_coeff=lambda_coeff)
    accuracy = model.train()
    acc_hist.append(accuracy)
    plt.scatter(xtest, model.predict(xtest), s=2, label=f'N = {i}')

plt.scatter(xtest, ytest, s=5, label='Analytical Solution', alpha=0.6)
plt.title('All models visualized')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.figure(2)
plt.plot(n_range, acc_hist, linewidth=2)
plt.title('L2 norm error with respect to training points')
plt.xlabel('Number of Training Points')
plt.ylabel('L2 Norm Error')
plt.show()
