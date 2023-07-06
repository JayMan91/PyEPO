#!/usr/bin/env python
# coding: utf-8
"""
Noise Contrastive Estimation Loss function
"""

import multiprocessing as mp

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool
from torch import nn

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo.utlis import getArgs


class NCE(nn.Module):
    """
        An autograd module for the noise contrastive estimation loss.

        For the noise contrastive loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of processes
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(processes)
        print("Num of cores: {}".format(self.processes))
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = dataset.sols.copy()

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()

        if np.random.uniform() <= self.solve_ratio:
            sol = _solve_in_forward(cp, self.optmodel, self.processes, self.pool)
            self.solpool = np.concatenate((self.solpool, sol))

        loss = 0
        for i in range(len(pred_cost)):
            obj_cp_i = torch.matmul(pred_cost[i], true_sol[i])
            solpool_obj_cp_i = torch.matmul(pred_cost[i], torch.from_numpy(self.solpool.T.astype(np.float32)))
            loss += self.optmodel.modelSense * (obj_cp_i - solpool_obj_cp_i).sum()

        return loss / (len(pred_cost)*len(self.solpool))

class NCE_MAP(NCE):
    """
        An autograd module for the noise contrastive MAP estimation loss.

        For the noise contrastive loss, the constraints are known and fixed,
        but the cost vector needs to be predicted from contextual data.
    """

    
    def forward(self, pred_cost, true_cost, true_sol):
        """
        Forward pass
        """
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()

        if np.random.uniform() <= self.solve_ratio:
            sol = _solve_in_forward(cp, self.optmodel, self.processes, self.pool)
            self.solpool = np.concatenate((self.solpool, sol))

        loss = 0
        for i in range(len(pred_cost)):
            obj_cp_i = torch.matmul(  (pred_cost[i] - true_cost[i]), true_sol[i])
            solpool_obj_cp_i = torch.matmul((pred_cost[i] - true_cost[i]), torch.from_numpy(self.solpool.T.astype(np.float32)))
            loss += self.optmodel.modelSense * (obj_cp_i - solpool_obj_cp_i).max()

        return loss / (len(pred_cost)*len(self.solpool))

def _solve_in_forward(cp, optmodel, processes, pool):
    """
    A function to solve optimization in the forward pass
    """
    # number of instance
    ins_num = len(cp)
    # single-core
    if processes == 1:
        sol = []
        for i in range(ins_num):
            # solve
            optmodel.setObj(cp[i])
            solq, _ = optmodel.solve()
            # solution
            sol.append(solq)
    # multi-core
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        res = pool.amap(
              _solveWithObj4Par,
              cp,
              [args] * ins_num,
              [model_type] * ins_num).get()
        # get res
        sol = np.array(list(map(lambda x: x[0], res)))
    return sol


def _solveWithObj4Par(cost, args, model_type):
    """
    A function to solve function in parallel processors

    Args:
        cost (np.ndarray): cost of objective function
        args (dict): optModel args
        model_type (ABCMeta): optModel class type

    Returns:
        tuple: optimal solution (list) and objective value (float)
    """
    # rebuild model
    optmodel = model_type(**args)
    # set obj
    optmodel.setObj(cost)
    # solve
    sol, obj = optmodel.solve()
    return sol, obj


def _check_sol(c, w, z):
    """
    A function to check solution is correct
    """
    ins_num = len(z)
    for i in range(ins_num):
        if abs(z[i] - np.dot(c[i], w[i])) / (abs(z[i]) + 1e-3) >= 1e-3:
            raise AssertionError(
                "Solution {} does not macth the objective value {}.".
                format(np.dot(c[i], w[i]), z[i][0]))
