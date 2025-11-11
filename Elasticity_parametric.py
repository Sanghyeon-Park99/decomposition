import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

"""#Build Network"""

from math import tanh
delta = 1

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return  x* self.sigmoid(x)

activation = Swish()
activation_ = torch.tanh

M = 64
selu = Swish()

T = True
F = False

class Net_2D(nn.Module):
    def __init__(self):
        super(Net_2D, self).__init__()
        torch.manual_seed(1234) # Fix Initial_Parameter
        self.hidden_layer1 = nn.Linear(4,M, bias = T)
        torch.manual_seed(1234)
        self.hidden_layer2 = nn.Linear(M,M, bias = T)
        torch.manual_seed(1234)
        self.hidden_layer3 = nn.Linear(M,M, bias = T)
        torch.manual_seed(1234)
        self.hidden_layer4 = nn.Linear(M,M, bias = T)
        torch.manual_seed(1234)
        # self.hidden_layer5 = nn.Linear(M,M, bias = T)
        # torch.manual_seed(1234)
        # self.hidden_layer6 = nn.Linear(M,M, bias = T)
        # torch.manual_seed(1234)
        self.output_layer = nn.Linear(M,2, bias = T)

    def forward(self, x,y,nu,E):
        inputs = torch.cat([x,y,nu,E],axis=1)
        layer1_out = activation(self.hidden_layer1(inputs))
        layer2_out = activation(self.hidden_layer2(layer1_out))
        layer3_out = activation(self.hidden_layer3(layer2_out))
        layer4_out = activation(self.hidden_layer4(layer3_out))
        # layer5_out = activation(self.hidden_layer5(layer4_out))
        # layer6_out = activation(self.hidden_layer6(layer5_out))
        output = self.output_layer(layer4_out)

        return output

class Net_2D_f(nn.Module):
    def __init__(self):
        super(Net_2D_f, self).__init__()
        torch.manual_seed(1234) # Fix Initial_Parameter
        self.hidden_layer1 = nn.Linear(4,M, bias = T)
        torch.manual_seed(1234)
        self.hidden_layer2 = nn.Linear(M,M, bias = T)
        torch.manual_seed(1234)
        self.hidden_layer3 = nn.Linear(M,M, bias = T)
        torch.manual_seed(1234)
        self.hidden_layer4 = nn.Linear(M,M, bias = T)
        torch.manual_seed(1234)
        # self.hidden_layer5 = nn.Linear(M,M, bias = T)
        # torch.manual_seed(1234)
        # self.hidden_layer6 = nn.Linear(M,M, bias = T)
        # torch.manual_seed(1234)
        self.output_layer = nn.Linear(M,2, bias = T)

    def forward(self, x,y,nu,E):
        inputs = torch.cat([x,y,nu,E],axis=1)
        layer1_out = activation(self.hidden_layer1(inputs))
        layer2_out = activation(self.hidden_layer2(layer1_out))
        layer3_out = activation(self.hidden_layer3(layer2_out))
        layer4_out = activation(self.hidden_layer4(layer3_out))
        # layer5_out = activation(self.hidden_layer5(layer4_out))
        # layer6_out = activation(self.hidden_layer6(layer5_out))
        output = self.output_layer(layer4_out)

        return output

"""#loss,optimize"""

import torch
import torch.nn.init as init

### (2) Model
mse_cost_function = torch.nn.MSELoss() #MSE Function can be shared

net_2d_1 = Net_2D()
net_2d_1 = net_2d_1.to(device)

net_2d_2 = Net_2D()
net_2d_2= net_2d_2.to(device)

net_2d_f = Net_2D_f()
net_2d_f= net_2d_f.to(device)

net_2d_f0 = Net_2D_f()
net_2d_f0= net_2d_f0.to(device)

optimizer1 = torch.optim.Adam(net_2d_1.parameters(),lr=0.001)
optimizer2 = torch.optim.Adam(net_2d_2.parameters(),lr=0.001)
optimizerf = torch.optim.Adam(net_2d_f.parameters(),lr=0.001)
optimizerf0 = torch.optim.Adam(net_2d_f0.parameters(),lr=0.001)

"""#set PDE,residual
![{D25C7BD6-E433-4B27-BC34-3C39EFBF4C0B}.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAATAAAAA9CAYAAAAqEjWZAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAA52SURBVHhe7Z0J1E1VFMePWeYmaqWJBkNKLE2aKVSSZVWkQRlWhVpLg4oUDWphKazQJMqQplUZG0grkpmSyJRSGigsKuJ0/vvb93rvefe+ez7P++559m+ty7n77nff/e65d5999j7nvGLaoARBEBykOP8vCILgHGLABEFwFjFggiA4ixgwQRCcRQyYIAjOIgZMEARnEQMmCIKziAETBMFZxIAJguAsYsAEQXAWMWCCIDiLGDBBEJxFDJggCM4iBkwQBGcRAyYIgrOIARMEwVnEgAmC4CxiwARBcBYxYIIgOIsYsDzijz/+UPfddx/vFQ3jxo1T06dP572DxyuvvKJmz57Ne5mx1RfiW5dJ4Ec9BPf56aefdJ06dfQnn3zCktwzYMAA3aRJE71nzx6WHDy2bdumzz77bP3hhx+yJBxb/XzBNGp66dKleuPGjSyJRpzrMpFY/SrRtGnT1MiRI3kvuxx//PFqyJAhvJdf/P333+rCCy9UrVq1Un369GFpbpkwYYLq0aOHWrx4sapWrRpLDy7ffvutuvTSS8lLMC8AS4Ox1XeV5cuXq+HDh6uxY8eqv/76S5UvX17t3LlTnXzyyeqFF15QzZo1Y830uFCXPmTGYsL69et1qVKlYFCzvj3yyCP8LflH+/btddOmTXPSWqYDLXzFihX1jBkzWJI7zEuqTeOkN2/ezJJwbPVd4/333/ffIdOo6W+++UbPnz/fl5UrV07/+OOPrL0/LtUliN3vQnbp0kW99NJLVIYlPvfcc6lcWH7++Wc1c+ZMtW7dOnXkkUeyNH9AiwXPC3/fsccey9LcgcencePGqlGjRur5559naW655JJLVN26dcm7iIKtvissWLCA6mLXrl3qmGOOUcZ40TP/9NNPq169erGWUu+++65q3bo17+3DxbqMXQzMvIh+a3HKKafo//77j48UjjvvvDNvva/du3fr2rVr69tuu40lueeNN97QxYsX16tXr2ZJ7kHsBNewcOFCloRjq+8KnTp1ovcGG557j1mzZulixYr5Hhh6OulwsS5jGcRPrIjRo0ez1J4NGzZo0wJRIDMfMZ4q3SPT8rIkt6BxOeGEE3TLli1ZUjTs3buXDPlVV13FknBs9V3BeC/+e2O8LpYWsHz5cj1ixAhK9qTD1bqMpQHLlhd211135XXs65xzztEXXHAB7+WeKVOmUB199NFHLCk6nnvuOWq5w+I7idjqu8B5553nG7CBAweyNBqu1mUsx4GddNJJynSLqGzcWcqm2GJaGjVx4kTKpuQjyNrMmzdPXXfddSzJPaNGjVIVKlRQTZo0YUnRgQyWab3pmqJgq5/vuFqXsQvieyAoffrpp6vdu3cr44Wp7777TpUoUYKPZqZr166qSpUq6qmnnmJJfDAepdqyZQsFTcOoWLGiKleuHO8l079/f2W8SzVr1ix18cUXs3R/ENDdvn272rFjR9J21lln0f3xwDVt27ZtP71atWqpqlWrstY+oI/PGy9QzZgxg6XRQGp/6dKldP4zzjhDma4LHykAw0IQkEbqH9eJgHQm8MAfccQR9Mx89dVXLA3GVj+uoB42b95MZdPtUosWLaIyhtPcfffdVAaoqzJlyvBeMk7XJQxYXOnYsaPvEtvEwuB6xin2ZSpET58+Xbdr147SxMYQ+39X2IbA64MPPshnSaZ58+bkZhvjxJL0dOvWLe25v/jiC9Yo4Jlnnkmr9/bbb7NGMnPnzqXjPXv2ZElmzENOwWXzImnzIOtq1arROczLox977DEaBoJhAN6xE088UZcsWVJ3796d7mEmWrRoQfqZ7omHrX4cwTCJxPoK2iZNmsSf2B+X6zLWBmzt2rX0R+DG2MTCTMsTm9gX/gbjEvsPku127bXX8pmSqVSpkq5Xrx7vBYOR+U888YSuW7du0nlTDRgeYuOtUlwtUS/IgA0aNIiOv/POOywJB5kt00Lr+vXrU1YM4EHu16+f/13NmjUj4/7oo4/SsV9++UWbVpiOjRw5kj4TBj4H3U8//ZQl4djqxxHjyeuXX36Ztho1avj3EvEwT44NCa0gXK7LWBswYOuFIcty1FFHxcL7wqBA4wr71+9tnlFO3ZC4qFChAm0YTIjpFThHKps2bSL9IOOWDjw0id+VasA8PvjggyS9IAOG1hfHTZeFJcH8888/2nQfdKNGjbTpcrC0gBUrViR9HwZfeqDR8uR33HEHS4MZOnQo6UZ5QYCtfirwQjp37qzbtm2btW3IkCF8dnsSg/g333wzSzPjcl3GfjI3BuCZF57KxpNQxjWlchCIDZmHqsgHrZoWRzVt2pRiXQCxpDfffJMmXCOuh9jBW2+9pSpXrkzHgeliUrwKG+JRiGeceeaZfHQfiA+CxM9mIij+kUrp0qW5FI7NNQwcOFAtW7ZMGU9gP/0ffviBSwWg7sDGjRspgeOB+EwmEDcB3rVlwlY/FdOFV2XLls3qFrWesonTdcmGLNbAYuNSsYV5Ydn0vjDBFK71a6+9RgNGbcF4Gu+a4Tpv3bqVjyTTo0cPX88YK5aGg8F+0Ed8Kyr4O7zvwRbkgU2bNi1JL8gDa9iwIR2Pcq8vuugifcMNN/BeMoiXeN+FmN9vv/3GR7R+6KGH9Kmnnkr3KMo0qalTp9J5MI4wCrb6caewHpjLdVkM/xjFyGCiZzaWJYFXBQtdp04dlgSzdu1aykggWxKWkezevTtl7jB14kAxN1w9++yzVMYwjptuuonKUZgzZw5NyfCYPHkyZYjS0a9fP2UqnspIY8P7ygSGh9x4442qd+/e5JVGwRh+1aFDB95TyhiwpGv0wNSk5s2b855SxoCpNm3a8N4+UG+my0DepOchBwGvGd6KeahZsg+k7b3MV7169ah1LywYVoKpZ6irKENvbPXjzvnnn6/mzp1LZWPA1Ouvv07lTLhcl9ZdSLiD69evz8r2559/8lnDqVGjhrr11lupHDQuDHMeYVyztR5WoiFBd86GYcOGcUmp6tWrqxYtWvDe/sA4exjvkUvheG1OuocoV3jXgIc5E2hs0l0rhnh4Lxy47LLLuFQ4vEZt79699H8mbPXzFafrEh6YC6xZsyY0I4n07MMPP8x7B44xiNRFg1uMYG1U4B4nBu67dOnCR9KDaROebqtWrVgajhdov/fee1mSmWx3IRs0aEDHTSPEEntwDYnf9d577/GRwoGMFc6DxE8UbPXjTmG7kC7XZeyD+B7wwm655RYqp3ph8L7Gjx+f1dVIsbLD0KFD1eOPPx44mDQdmzZt8gP3IF03zQNBT7juHtdccw2XwvECnFu3bqX/s0lUbyQb1/DZZ59xqcCbDBqQi1kVUfCuBYMyo2Crnwq6U4MHD1Z9+/bN2jZlyhQ+e+5wuS6dMWAgKCOJWFUcMo/ANApcKgCjk4NAVtIDBrN9+/a8Fw6mWoEDeeCC2LBhA5fCsbkGNDb4++rXr09ToDwwi8ADcRjvRUoE9wiLUa5atYolwSCzC7BwXxRs9VNBmGHq1Km0EGe2ts8//5zPnjucrssCR8wdOnTo4LupyEiiq3f00UfHZtQ9qFq1qn+N69atY2kyO3fu1OZB8PUmTJjARzKDgYGHHXYYDSaMijdZ19vMi8JHkrn99tuT9IK6kP379w897oEBjKVLl/bPh0G1AFle/A2eHN+bCrrjtWrVijRgFxhvmc6FDFYUbPXjTmG7kC7XpVMeGEDmLdELw1zHTp06xWqxwquvvppLiuaJpYJuGrrDGCsGMKcRWcWowEVHhgbdT8wxi0LqwpDpWtuvv/6aso6J/Pvvv1xKBhkvsHDhQvo/iO+//54CvKBUqVJ+QuPVV1+leXIephHi0j4QEkCSY8yYMSwJZ/78+f69iYKtfr7idF2yIXOKRC8Mo9bj5H0B07Xwp+5gjSVM00HSAZ7Tl19+6U8twrUPHz6cP2WHN9Vizpw5LMkMlhfy7tsVV1xB00twTb/++qseNWoUeY5Y1M7TwWa6A5Qc+fjjj/ksBZgHllrjK6+8kiXpwb04/PDDad4m5saByZMn08J6GHleuXJl+p7TTjvNX2gPa1e1adOG5tUhYREVeOJRx9IBW/248uSTT5LHUr16db/e8HdBhi3TenEu16WTBgxzsbyMZDYzj9kE0yzgYmNgLa4TlQyDhTKyqA888ABNCSoss2fPpnPZTD2BscKvzcCNx2exefcRU0Mwr82bppS6YSBiKjCCmDSP84YBow2DXqJECXqQ0d3o27cvfW7JkiX04pQvX56+B1OooIcF7TCPNCp4YfD5qM+DrX6cwb3DBuOCFx+Tp/E/9suWLUsLGWbC1bp00oABTNiGxxA37ysdGJE8b948qmB4O9kCMQWMjC4MpgtJ8TnTbdS///47SwuWqcYKBxi2gp/iwmRhxOvSPdiI2+FBizIZGp/HQwxvAC15KjiOhgnH8Z22DBs2jK5l5cqVLAnHVj/fcbUurUfixwlzoyINvstXBg0apO6//376GS1kfnIN4mPHHXecuvzyy2l2QFFSu3Ztir1EzeLZ6uc7ztYlmTHBSZD9QSbznnvuYUnuwdLFWEXjQLrDBwqWQcajHJRZTcVW/1DBxbp02gMTCsbldOvWjQbFVqpUiaW5A/PnsNJm27Zti+xHdVu2bEk/3oqpZFGw1T9UcLEuxQPLA/AT8IgJFhUzZ86kgG3QmLeDyfjx42nqVtiCfYnY6h9quFSXQAxYHoAgfM2aNYv0F2XQ/cDwECQBcsWqVasoE5Y6xCMIW/1DFRfq0kMMWJ6AzCHG32DcTVHRtWvXnK2thbFLWLH2xRdfZEk4tvqHOnGuy0QkBpZHYJI7Zidg7a+iAmuxIY6SOBvhYICJ9viFm+uvv54l4djqC/Gty0TEgAmC4CyH7iAqQRCcRwyYIAjOIgZMEARnEQMmCIKziAETBMFZxIAJguAsYsAEQXAWMWCCIDiKUv8DXPYol++gGM0AAAAASUVORK5CYII=)
"""

N = 1
# lamb = 10000
# C = 0.01
mu = 1

#/ 10,0.1 / 100,0.1 / 1000,0.05 / 10000,0.05 / 100000,0.01 / 1000000,0.001

def f(x, y, x_bd, y_bd, nu, nu_, E, E_, net1, net2, net3, net4, epoch):

    mu = E/(2*(1+0.5*nu))
    lamb = E*nu/((1-1*nu)*(1+0.5*nu))
    # lam_ = E*nu_/(1-nu_)

    f1 = -((lamb+mu)/lamb)*torch.cos(x+y) + mu*(8*torch.cos(2*x)*torch.sin(2*y) + (2/lamb)*torch.sin(x)*torch.sin(y) - 4*torch.sin(2*y))
    f2 = -((lamb+mu)/lamb)*torch.cos(x+y) + mu*(-8*torch.cos(2*y)*torch.sin(2*x) + (2/lamb)*torch.sin(x)*torch.sin(y) + 4*torch.sin(2*x))

    C = 1/((lamb)**(0.5))


    w1 = net1(x,y,nu,E)[:, 0:1]
    v1 = net1(x,y,nu,E)[:, 1:2] #u_tilda

    w2 = net2(x,y,nu,E)[:, 0:1]
    v2 = net2(x,y,nu,E)[:, 1:2] #u_hat

    net_f1 = net3(x,y,nu,E)[:, 0:1]
    net_f2 = net3(x,y,nu,E)[:, 1:2] #f_hat1

    net_f1_0 = net4(x,y,nu,E)[:, 0:1]
    net_f2_0 = net4(x,y,nu,E)[:, 1:2] #f_hat2

    w1_bdy = net1(x_bd,y_bd,nu_,E_)[:, 0:1]
    v1_bdy = net1(x_bd,y_bd,nu_,E_)[:, 1:2]

    w2_bdy = net2(x_bd,y_bd,nu_,E_)[:, 0:1]
    v2_bdy = net2(x_bd,y_bd,nu_,E_)[:, 1:2]

    w1_x = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
    w1_y = torch.autograd.grad(w1.sum(), y, create_graph=True)[0]
    w1_xx = torch.autograd.grad(w1_x.sum(), x, create_graph=True)[0]
    w1_yy = torch.autograd.grad(w1_y.sum(), y, create_graph=True)[0]
    w1_xy = torch.autograd.grad(w1_x.sum(), y, create_graph=True)[0]

    v1_x = torch.autograd.grad(v1.sum(), x, create_graph=True)[0]
    v1_y = torch.autograd.grad(v1.sum(), y, create_graph=True)[0]
    v1_xx = torch.autograd.grad(v1_x.sum(), x, create_graph=True)[0]
    v1_yy = torch.autograd.grad(v1_y.sum(), y, create_graph=True)[0]
    v1_xy = torch.autograd.grad(v1_x.sum(), y, create_graph=True)[0]

    w2_x = torch.autograd.grad(w2.sum(), x, create_graph=True)[0]
    w2_y = torch.autograd.grad(w2.sum(), y, create_graph=True)[0]
    w2_xx = torch.autograd.grad(w2_x.sum(), x, create_graph=True)[0]
    w2_yy = torch.autograd.grad(w2_y.sum(), y, create_graph=True)[0]
    w2_xy = torch.autograd.grad(w2_x.sum(), y, create_graph=True)[0]

    v2_x = torch.autograd.grad(v2.sum(), x, create_graph=True)[0]
    v2_y = torch.autograd.grad(v2.sum(), y, create_graph=True)[0]
    v2_xx = torch.autograd.grad(v2_x.sum(), x, create_graph=True)[0]
    v2_yy = torch.autograd.grad(v2_y.sum(), y, create_graph=True)[0]
    v2_xy = torch.autograd.grad(v2_x.sum(), y, create_graph=True)[0]

    u1_dual_r1 = ((-2*mu*(w1_xx + 0.5*(v1_xy+w1_yy)) - net_f1_0)**2).mean()
    u1_dual_r2 = ((-2*mu*(v1_yy + 0.5*(v1_xx+w1_xy)) - net_f2_0)**2).mean()
    u1_r1 = ((-(lamb)*(w1_xx+v1_xy) - net_f1)**2).mean()
    u1_r2 = ((-(lamb)*(w1_xy+v1_yy) - net_f2)**2).mean()

    u2_dual_r1 = ((-(1)*(w2_xx+v2_xy))**2).mean()
    u2_dual_r2 = ((-(1)*(w2_xy+v2_yy))**2).mean()
    u2_r1 = ((-2*mu*(w2_xx + 0.5*(v2_xy+w2_yy)) - f1 + net_f1 + net_f1_0)**2).mean()
    u2_r2 = ((-2*mu*(v2_yy + 0.5*(v2_xx+w2_xy)) - f2 + net_f2 + net_f2_0)**2).mean()

    bdy1_loss = ((w1_bdy + w2_bdy)**2).mean()
    bdy2_loss = ((v1_bdy + v2_bdy)**2).mean()

    # if epoch % 50 == 1:
    #     print("u1_dual_r1: ",u1_dual_r1)
    #     print("u1_dual_r2: ",u1_dual_r2)
    #     print("u1_r1: ",u1_r1)
    #     print("u1_r2: ",u1_r2)

    #     print("u2_dual_r1: ",u2_dual_r1)
    #     print("u2_dual_r2: ",u2_dual_r2)
    #     print("u2_r1: ",u2_r1)
    #     print("u2_r2: ",u2_r2)
    #     print("Bdy1 :",bdy1_loss)
    #     print("Bdy2 :",bdy2_loss)

    return u1_dual_r1 + u1_dual_r2 + C*u1_r1 + C*u1_r2 + u2_dual_r1 + u2_dual_r2 + u2_r1 + u2_r2 + 20*bdy1_loss + 20*bdy2_loss

def f_1(x, y, x_bd, y_bd, nu, nu_, E, E_, net1, epoch):

    mu = E/(2*(1+0.5*nu))
    lamb = E*nu/((1-nu)*(1+0.5*nu))
    # lam_ = E*nu_/(1-nu_)

    f1 = -((lamb+mu)/lamb)*torch.cos(x+y) + mu*(8*torch.cos(2*x)*torch.sin(2*y) + (2/lamb)*torch.sin(x)*torch.sin(y) - 4*torch.sin(2*y))
    f2 = -((lamb+mu)/lamb)*torch.cos(x+y) + mu*(-8*torch.cos(2*y)*torch.sin(2*x) + (2/lamb)*torch.sin(x)*torch.sin(y) + 4*torch.sin(2*x))

    C = 1/((lamb)**(0.5))

    w1 = net1(x,y,nu,E)[:, 0:1]
    v1 = net1(x,y,nu,E)[:, 1:2] #u_tilda

    w1_bdy = net1(x_bd,y_bd,nu_,E_)[:, 0:1]
    v1_bdy = net1(x_bd,y_bd,nu_,E_)[:, 1:2]

    w1_x = torch.autograd.grad(w1.sum(), x, create_graph=True)[0]
    w1_y = torch.autograd.grad(w1.sum(), y, create_graph=True)[0]
    w1_xx = torch.autograd.grad(w1_x.sum(), x, create_graph=True)[0]
    w1_yy = torch.autograd.grad(w1_y.sum(), y, create_graph=True)[0]
    w1_xy = torch.autograd.grad(w1_x.sum(), y, create_graph=True)[0]

    v1_x = torch.autograd.grad(v1.sum(), x, create_graph=True)[0]
    v1_y = torch.autograd.grad(v1.sum(), y, create_graph=True)[0]
    v1_xx = torch.autograd.grad(v1_x.sum(), x, create_graph=True)[0]
    v1_yy = torch.autograd.grad(v1_y.sum(), y, create_graph=True)[0]
    v1_xy = torch.autograd.grad(v1_x.sum(), y, create_graph=True)[0]

    u_r1 = ((-2*mu*(w1_xx + 0.5*(v1_xy+w1_yy))-(lamb)*(w1_xx+v1_xy) - f1)**2).mean()
    u_r2 = ((-2*mu*(v1_yy + 0.5*(v1_xx+w1_xy))-(lamb)*(w1_xy+v1_yy) - f2)**2).mean()

    bdy1_loss = ((w1_bdy)**2).mean()
    bdy2_loss = ((v1_bdy)**2).mean()


    return C*u_r1 + C*u_r2 + 50*bdy1_loss + 50*bdy2_loss

"""#Set data point"""

x_collocation = np.random.uniform(low = 0 , high = np.pi , size=(1000,1))
y_collocation = np.random.uniform(low = 0 , high = np.pi , size=(1000,1))

pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)

x = np.linspace(0, np.pi, 75)
y = np.linspace(0, np.pi, 75)

bottom_edge = np.column_stack((x, np.zeros_like(x)))  # y = 0, x는 [0, pi]
top_edge = np.column_stack((x, np.full_like(x, np.pi)))  # y = pi, x는 [0, pi]
left_edge = np.column_stack((np.zeros_like(y), y))  # x = 0, y는 [0, pi]
right_edge = np.column_stack((np.full_like(y, np.pi), y))  # x = pi, y는 [0, pi]


boundary_points = np.vstack((bottom_edge, top_edge, left_edge, right_edge))

print("Total boundary points:", boundary_points.shape[0])
# print(boundary_points)

x_coords = boundary_points[:, 0].reshape(-1, 1)  # x 좌표 (400, 1)
y_coords = boundary_points[:, 1].reshape(-1, 1)  # y 좌표 (400, 1)

bdy_x_collocation = Variable(torch.from_numpy(x_coords).float(), requires_grad=True).to(device)
bdy_y_collocation = Variable(torch.from_numpy(y_coords).float(), requires_grad=True).to(device)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

plt.scatter(x_collocation, y_collocation, color='blue', marker='o', label='Points',s=0.1)

plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Points',s=0.1)

plt.title("Point Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 
plt.legend()

#
plt.grid(True)
plt.show()

"""# Learn F"""

f_x_col = Variable(torch.from_numpy(x_collocation).float(), requires_grad=False).to(device)
f_y_col = Variable(torch.from_numpy(y_collocation).float(), requires_grad=False).to(device)

# len(f_x_col)

# label_f1 = -((torch.pi**2)*(lamb+1)/lamb)*torch.cos(torch.pi*(f_x_col+f_y_col)) - (12*f_x_col**2 - 12*f_x_col + 2)*(5*f_y_col**4 - 8*f_y_col**3 + 3*f_y_col**2) - (f_x_col**4 - 2*f_x_col**3 + f_x_col**2)*(60*f_y_col**2 - 48*f_y_col + 6) + (2/lamb)*(torch.pi**2)*torch.sin(torch.pi*f_x_col)*torch.sin(torch.pi*f_y_col)
# label_f2 = -((torch.pi**2)*(lamb+1)/lamb)*torch.cos(torch.pi*(f_x_col+f_y_col)) + (24*f_x_col - 12)*(20*f_y_col**3 - 24*f_y_col**2 + 6*f_y_col) + (4*f_x_col**3 - 6*f_x_col**2 + 2*f_x_col)*(20*f_y_col**3 - 24*f_y_col**2 + 6*f_y_col)+ (2/lamb)*(torch.pi**2)*torch.sin(torch.pi*f_x_col)*torch.sin(torch.pi*f_y_col)

previous_validation_loss = 99999999.0
gap3 = 1
epoch = 0


while epoch < 3000:
    optimizerf.zero_grad()

##------------------------------------Compute Loss-----------------------------------------##
    loss_f = 0
    for j in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9998]:
            nu = np.ones((1000,1))*i
            E = np.ones((1000,1))*j
            lamb = j*nu/((1-1*np.ones((1000,1))*i)*(1+ 0.5*nu))
            mu = E/(2*(1+0.5*nu))
            nu = Variable(torch.from_numpy(nu).float(), requires_grad=False).to(device)
            E = Variable(torch.from_numpy(E).float(), requires_grad=False).to(device)
            lamb = Variable(torch.from_numpy(lamb).float(), requires_grad=False).to(device)
            mu = Variable(torch.from_numpy(mu).float(), requires_grad=False).to(device)

            label_f1 = -((lamb+mu)/lamb)*torch.cos(f_x_col+f_y_col) + mu*(8*torch.cos(2*f_x_col)*torch.sin(2*f_y_col) + (2/lamb)*torch.sin(f_x_col)*torch.sin(f_y_col) - 4*torch.sin(2*f_y_col))
            label_f2 = -((lamb+mu)/lamb)*torch.cos(f_x_col+f_y_col) + mu*(-8*torch.cos(2*f_y_col)*torch.sin(2*f_x_col) + (2/lamb)*torch.sin(f_x_col)*torch.sin(f_y_col) + 4*torch.sin(2*f_x_col))

            pred_f_1 = net_2d_f(pt_x_collocation, pt_y_collocation, nu,E)[:, 0:1]
            pred_f_2 = net_2d_f(pt_x_collocation, pt_y_collocation, nu,E)[:, 1:2]
            msef_1 = mse_cost_function(pred_f_1, (0.3)*label_f1)
            msef_2 = mse_cost_function(pred_f_2, (0.3)*label_f2)

            loss_f += 1*(msef_1 + msef_2)

##------------------------------------Optimize Loss-----------------------------------------##

    loss_f.sum()
    loss_f.backward()
    optimizerf.step()

    epoch += 1

    if epoch % 100 == 1:
        print("MSE Error: ", loss_f.item())


while epoch < 3000:
    optimizerf0.zero_grad()

##------------------------------------Compute Loss-----------------------------------------##
    loss_f = 0
    for j in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9998]:
            nu = np.ones((1000,1))*i
            E = np.ones((1000,1))*j
            lamb = j*nu/((1-1*np.ones((1000,1))*i)*(1+ 0.5*nu))
            mu = E/(2*(1+0.5*nu))
            nu = Variable(torch.from_numpy(nu).float(), requires_grad=False).to(device)
            E = Variable(torch.from_numpy(E).float(), requires_grad=False).to(device)
            lamb = Variable(torch.from_numpy(lamb).float(), requires_grad=False).to(device)
            mu = Variable(torch.from_numpy(mu).float(), requires_grad=False).to(device)

            label_f1 = -((lamb+mu)/lamb)*torch.cos(f_x_col+f_y_col) + mu*(8*torch.cos(2*f_x_col)*torch.sin(2*f_y_col) + (2/lamb)*torch.sin(f_x_col)*torch.sin(f_y_col) - 4*torch.sin(2*f_y_col))
            label_f2 = -((lamb+mu)/lamb)*torch.cos(f_x_col+f_y_col) + mu*(-8*torch.cos(2*f_y_col)*torch.sin(2*f_x_col) + (2/lamb)*torch.sin(f_x_col)*torch.sin(f_y_col) + 4*torch.sin(2*f_x_col))

            pred_f_1 = net_2d_f0(pt_x_collocation, pt_y_collocation, nu,E)[:, 0:1]
            pred_f_2 = net_2d_f0(pt_x_collocation, pt_y_collocation, nu,E)[:, 1:2]
            msef_1 = mse_cost_function(pred_f_1, (0.3)*label_f1)
            msef_2 = mse_cost_function(pred_f_2, (0.3)*label_f2)

            loss_f += 1*(msef_1 + msef_2)

##------------------------------------Optimize Loss-----------------------------------------##

    loss_f.sum()
    loss_f.backward()
    optimizerf0.step()

    epoch += 1

    if epoch % 100 == 1:
        print("MSE Error: ", loss_f.item())

"""#Observe Points"""

x_ = np.linspace(0, np.pi, 16)
y_ = np.linspace(0, np.pi, 16)

ms_x_ , ms_y_= np.meshgrid(x_,y_)

x_ = np.ravel(ms_x_).reshape(-1,1)
y_ = np.ravel(ms_y_).reshape(-1,1)

pt_x_ = Variable(torch.from_numpy(x_).float(), requires_grad=False).to(device)
pt_y_ = Variable(torch.from_numpy(y_).float(), requires_grad=False).to(device)

Iterlist = []
Error_decom = []
Error_stan = []

"""#Train"""

previous_validation_loss = 99999999.0
gap3 = 1
epoch = 0

while epoch < 10000:
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizerf.zero_grad()
    optimizerf0.zero_grad()
##------------------------------------Compute Loss-----------------------------------------##

    loss = 0
    for j in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
        for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9998]:

            nu = np.ones((1000,1))*i
            nu = Variable(torch.from_numpy(nu).float(), requires_grad=False).to(device)

            nu_ = np.ones((300,1))*i
            nu_ = Variable(torch.from_numpy(nu_).float(), requires_grad=False).to(device)

            E = np.ones((1000,1))*j
            E = Variable(torch.from_numpy(E).float(), requires_grad=False).to(device)

            E_ = np.ones((300,1))*j
            E_ = Variable(torch.from_numpy(E_).float(), requires_grad=False).to(device)

            loss += 0.001*f(pt_x_collocation, pt_y_collocation, bdy_x_collocation, bdy_y_collocation, nu, nu_,E,E_, net_2d_1, net_2d_2, net_2d_f, net_2d_f0, epoch)
            # loss += 0.001*f_1(pt_x_collocation, pt_y_collocation, bdy_x_collocation, bdy_y_collocation, nu, nu_,E,E_, net_2d_1, epoch)

##------------------------------------Optimize Loss-----------------------------------------##

    loss = loss.sum()
    loss.backward()
    optimizer1.step()
    optimizer2.step()
    optimizerf.step()
    optimizerf0.step()

    epoch += 1

##----------------------------Result-----------------------------##

    if (epoch) % 100 == 1:

        gap2 = 0
        gap1 = 0

        for j in [1, 2, 3, 4]:
            for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9998]:

                nu1 = np.ones((256,1))*i
                nu1 = Variable(torch.from_numpy(nu1).float(), requires_grad=False).to(device)
                E1 = np.ones((256,1))*j
                E1 = Variable(torch.from_numpy(E1).float(), requires_grad=False).to(device)

                lamb = E1*nu1/((1-nu1)*(1+0.5*nu1))

                gap2 += (1/44)*torch.sqrt((((((net_2d_1(pt_x_,pt_y_,nu1,E1)+net_2d_2(pt_x_,pt_y_,nu1,E1))[:,1:2]) - (1-torch.cos(2*pt_y_))*torch.sin(2*pt_x_) - torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean())/((((1-torch.cos(2*pt_y_))*torch.sin(2*pt_x_) + torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean()))
                gap1 += (1/44)*torch.sqrt((((((net_2d_1(pt_x_,pt_y_,nu1,E1)+net_2d_2(pt_x_,pt_y_,nu1,E1))[:,0:1]) - (torch.cos(2*pt_x_)-1)*torch.sin(2*pt_y_) - torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean())/((((torch.cos(2*pt_x_)-1)*torch.sin(2*pt_y_) + torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean()))

                # gap2 += (1/147)*torch.sqrt((((((net_2d_1(pt_x_,pt_y_,nu1,E1))[:,1:2]) - (1-torch.cos(2*pt_y_))*torch.sin(2*pt_x_) - torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean())/((((1-torch.cos(2*pt_y_))*torch.sin(2*pt_x_) + torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean()))
                # gap1 += (1/147)*torch.sqrt((((((net_2d_1(pt_x_,pt_y_,nu1,E1))[:,0:1]) - (torch.cos(2*pt_x_)-1)*torch.sin(2*pt_y_) - torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean())/((((torch.cos(2*pt_x_)-1)*torch.sin(2*pt_y_) + torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean()))


            # gap2 += (1/18)*torch.sqrt((((((net_2d_1(pt_x_,pt_y_,nu1))[:,1:2]) - (1-torch.cos(2*pt_y_))*torch.sin(2*pt_x_) - torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean())/((((1-torch.cos(2*pt_y_))*torch.sin(2*pt_x_) + torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean()))
            # gap1 += (1/18)*torch.sqrt((((((net_2d_1(pt_x_,pt_y_,nu1))[:,0:1]) - (torch.cos(2*pt_x_)-1)*torch.sin(2*pt_y_) - torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean())/((((torch.cos(2*pt_x_)-1)*torch.sin(2*pt_y_) + torch.sin(pt_x_)*torch.sin(pt_y_)/lamb)**2).mean()))

        Iterlist.append(epoch)
        Error_decom.append(0.5*(gap1.cpu().detach().numpy()+gap2.cpu().detach().numpy()))

        # Error_stan.append(0.5*(gap1.cpu().detach().numpy()+gap2.cpu().detach().numpy()))

    if (epoch) % 100 == 1:
        print("-----------------------------------------------------------------------------")
        print("Epoch:",epoch)
        # print("Lambda PDE Loss :", (mse1_r1 + mse1_r2).item())
        # print("Mu PDE Loss :", (mse2_r1 + mse2_r2).item())
        print("Loss :", (loss).item())
        print(gap1.item())

# Iterlist 저장
np.savetxt("Iterlist.txt", np.array(Iterlist), fmt="%d")

# Error_decom 저장
np.savetxt("Error_decom.txt", np.array(Error_decom), fmt="%.6f")


"""#Plotting Prediction"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

# x_ = np.arange(0,1,0.01)
# y_ = np.arange(0,1,0.01)

# ms_x_ , ms_y_= np.meshgrid(x_,y_)

# x_ = np.ravel(ms_x_).reshape(-1,1)
# y_ = np.ravel(ms_y_).reshape(-1,1)

# pt_x_ = Variable(torch.from_numpy(x_).float(), requires_grad=False).to(device)
# pt_y_ = Variable(torch.from_numpy(y_).float(), requires_grad=False).to(device)

nu1 = np.ones((2500,1))*0.67
nu1 = Variable(torch.from_numpy(nu1).float(), requires_grad=False).to(device)

pt_u3 = ((net_2d_2(pt_x_,pt_y_,nu1) + net_2d_1(pt_x_,pt_y_,nu1))[:,0:1])

# pt_u3 = (net_2d_1(pt_x_,pt_y_))[:,1:2]


print(pt_u3.max())
print(pt_u3.min())

u3 = pt_u3.data.cpu().numpy()
ms_u3 = u3.reshape(ms_x_.shape)

surf3 = ax.plot_surface(ms_x_,ms_y_,ms_u3,cmap=cm.coolwarm,linewidth=0, antialiased=True)

ax.set_zlim(-2, 2)

plt.show()

gap1 = torch.sqrt((((((net_2d_1(pt_x_,pt_y_)+net_2d_2(pt_x_,pt_y_))[:,0:1]) - (pt_x_**4 - 2*pt_x_**3 + pt_x_**2)*(5*pt_y_**4 - 8*pt_y_**3 + 3*pt_y_**2) - torch.sin(torch.pi*pt_x_)*torch.sin(torch.pi*pt_y_)/lamb)**2).mean())/(((((pt_x_**4 - 2*pt_x_**3 + pt_x_**2)*(5*pt_y_**4 - 8*pt_y_**3 + 3*pt_y_**2) + torch.sin(torch.pi*pt_x_)*torch.sin(torch.pi*pt_y_)/lamb)**2).mean())))
gap2 = torch.sqrt((((((net_2d_1(pt_x_,pt_y_)+net_2d_2(pt_x_,pt_y_))[:,1:2]) + (4*pt_x_**3 - 6*pt_x_**2 + 2*pt_x_)*(pt_y_**5 - 2*pt_y_**4 + pt_y_**3) - torch.sin(torch.pi*pt_x_)*torch.sin(torch.pi*pt_y_)/lamb)**2).mean())/(((((4*pt_x_**3 - 6*pt_x_**2 + 2*pt_x_)*(pt_y_**5 - 2*pt_y_**4 + pt_y_**3) - torch.sin(torch.pi*pt_x_)*torch.sin(torch.pi*pt_y_)/lamb)**2).mean())))

print(gap1)
print(gap2)

"""#Plot learning curve"""

import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

# plt.plot(Iterlist[0:40], Error_stan[0:40] ,color = 'blue', marker='x',markersize=0.3,label='Standard',linewidth = "0.8")
plt.plot(Iterlist, Error_decom ,color = 'red', marker='x',markersize=0.3,label='Decompose',linewidth = "0.8")


plt.yscale('log')
plt.legend(loc = 'lower left')

plt.xlabel('epoch')
plt.ylabel('Relative_L2_Error')
plt.show()
plt.close()