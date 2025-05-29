import torch
import torch.nn as nn

#f(x)=w*x+b
x = torch.tensor([1,2,3],device="cuda")
y = torch.tensor([4,7,11],device="cuda")
w= torch.tensor(0.0,requires_grad=True,device="cuda")

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.x = x
        self.y = y
        self.w = w
    def forward(self,x):    
        return self.w * x
    def loss(self,y,y_pred):
        return ((y-y_pred)**2).mean()
    
epoch =100
lr = 0.001
X_test = torch.tensor(6.5,requires_grad=True)
model = LinearRegression()

for e in range(epoch):
    y_pred = model.forward(x=x)
    l = model.loss(y,y_pred)
    l.backward()
    with torch.no_grad():
        model.w -= lr * model.w.grad
    model.w.grad.zero_()
    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.3f}')

print(f'Prediction after training: f({X_test}) = {model.forward(X_test).item():.3f}')