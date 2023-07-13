
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# optimization model
class myModel(optGrbModel):
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.num_item = len(weights[0])
        super().__init__()

    def _getModel(self):
        # ceate a model
        m = gp.Model()
        # varibles
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # sense (must be minimize)
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
        return m, x


def projection_func(A,b,x):
    '''
    x is an input, we want x to obey Ax <=b
    If that's not the case return after projecting x into feasible polyhedron
    y = x - A^T (A A^T)^{-1} ( Relu(Ax -b ) )
    '''

    reluop= nn.ReLU()
    AtinvAAt = torch.mm( A.t(),torch.inverse( torch.mm( A, A.t())))
    return x -  torch.einsum('ij,bj->bi', AtinvAAt, ( reluop( torch.einsum('ij,bj->bi', A, x)-b )) )
   



# prediction model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out

# prediction model
class SurrogateRegression(nn.Module):

    def __init__(self):
        super(SurrogateRegression, self).__init__()
        self.linear = nn.Linear( num_item, num_item)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.softmax (self.linear(x) )
        return out

if __name__ == "__main__":

    # generate data
    num_data = 1000 # number of data
    num_feat = 5 # size of feature
    num_item = 10 # number of items
    weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim=3, deg=4, noise_width=0.5, seed=135)
    A = torch.from_numpy(weights).float()
    b = torch.tensor([7.,8.,9.]).float()
    print(A.shape ,b.shape)

    # init optimization model
    optmodel = myModel(weights)

    # init prediction model
    predmodel = LinearRegression()
    surrogatemodel = SurrogateRegression()



    # set optimizer
    predoptimizer = torch.optim.Adam(predmodel.parameters(), lr=1e-2)
    surrogateoptimizer = torch.optim.Adam(surrogatemodel.parameters(), lr=1e-2)


    # # init SPO+ loss
    spo = pyepo.func.SPOPlus(optmodel, processes=4)
    

    # build dataset
    dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
    # nce = pyepo.func.NCE_MAP(optmodel, processes=4, dataset= dataset)
    # get data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # training
    num_epochs = 20
    sense = optmodel.modelSense
    regret_list = []
    for epoch in range(num_epochs):
        proxyregret = 0

        regret = pyepo.metric.regret(predmodel, optmodel, dataloader)
        regret_list.append(regret)
        for (i,data) in enumerate(dataloader):
            x, c, w, z = data
            # forward pass
            cp = predmodel(x)
            wp = surrogatemodel(cp)
            # wp might be infeasible, so we project back to feasible
            wp = projection_func(A,b,wp)
            proxyregret += sense*(c*( wp - w )).sum(1).mean()
            nceloss = (sense*((w-wp)*(cp-c)).sum(1)).max()


            # nceloss = nceloss.max()
            # backward pass
            predoptimizer.zero_grad()
            nceloss.backward()
            predoptimizer.step()


            cp = predmodel(x)
            wp = surrogatemodel(cp)
            surrogateLoss = sense*(cp*wp).sum()
            surrogateoptimizer.zero_grad()
            surrogateLoss.backward()
            surrogateoptimizer.step()
            # print("NCE Loss: {}, Surrogate Loss: {}".format(nceloss.item(), surrogateLoss.item()))

        proxyregret /= i+1
       
        print("Epoch {:2}, On Training Set Regret : {:.4f}, Proxy Regret: {:.4f}".format(epoch, regret, proxyregret))

        

    # eval

    plt.plot(regret_list)
    plt.show()
