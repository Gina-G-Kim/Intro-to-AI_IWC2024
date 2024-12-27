import torch

x_train = torch.FloatTensor([[1,1], [2,2], [3,3]])
y_train = torch.FloatTensor([[10], [20], [30]])

W = torch.randn([2,1], requires_grad=True)
b = torch.randn([1], requires_grad=True)

optimizer = torch.optim.SGD([W,b], lr = 0.01)

# 1) Model Setup
def H(x):
    model = torch.matmul(x,W)+b # H(x) = Wx + b
    return model

# 2) Training
for step in range (2000):
    # Cost func. Minimization -> then, we have to make cost 
    cost = torch.mean((H(x_train) - y_train)**2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if(step%100 == 0):
        print(cost)

# 3) Testing
x_test = torch.FloatTensor([[4,4]])
print(H(x_test))