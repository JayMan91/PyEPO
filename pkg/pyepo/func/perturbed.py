#!/usr/bin/env python
# coding: utf-8
"""
Perturbed optimization function
"""

import multiprocessing as mp

import numpy as np
import torch
from pathos.multiprocessing import ProcessingPool
from torch.autograd import Function
from torch import nn

from pyepo import EPO
from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo.utlis import getArgs

class perturbedOpt(nn.Module):
    """
    A autograd module for differentiable perturbed optimizer, in which random
    perturbed costs are sampled to optimize.

    For the perturbed optimizer, the cost vector need to be predicted from
    contextual data and are perturbed with Gaussian noise.

    The perturbed optimizer differentiable in its inputs with non-zero Jacobian.
    Thus, allows us to design an algorithm based on stochastic gradient descent.
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            seed (int): random state seed
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
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
        # random state
        self.rnd = np.random.RandomState(seed)
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        self.solpool = None
        if self.solve_ratio < 1: # init solution pool
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = dataset.sols.copy()
        # build optimizer
        self.ptb = perturbedOptFunc()

    def forward(self, pred_cost):
        """
        Forward pass
        """
        sols = self.ptb.apply(pred_cost, self.optmodel, self.n_samples,
                              self.sigma, self.processes, self.pool, self.rnd,
                              self.solve_ratio, self)
        return sols


class perturbedOptFunc(Function):
    """
    A autograd function for perturbed optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, optmodel, n_samples, sigma,
                processes, pool, rnd, solve_ratio, module):
        """
        Forward pass for perturbed

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            rnd (RondomState): numpy random state
            solve_ratio (float): the ratio of new solutions computed during training
            module (nn.Module): perturbedOpt module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        # sample perturbations
        noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        # solve with perturbation
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            ptb_sols = _solve_in_forward(ptb_c, optmodel, processes, pool)
            if solve_ratio < 1:
                sols = ptb_sols.reshape(-1, cp.shape[1])
                module.solpool = np.concatenate((module.solpool, sols))
        else:
            ptb_sols = _cache_in_pass(ptb_c, optmodel, module.solpool)
        # solution expectation
        e_sol = ptb_sols.mean(axis=1)
        # convert to tensor
        noises = torch.FloatTensor(noises).to(device)
        ptb_sols = torch.FloatTensor(ptb_sols).to(device)
        e_sol = torch.FloatTensor(e_sol).to(device)
        # save solutions
        ctx.save_for_backward(ptb_sols, noises)
        # add other objects to ctx
        ctx.optmodel = optmodel
        ctx.n_samples = n_samples
        ctx.sigma = sigma
        return e_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed
        """
        ptb_sols, noises = ctx.saved_tensors
        optmodel = ctx.optmodel
        n_samples = ctx.n_samples
        sigma = ctx.sigma
        grad = torch.einsum("nbd,bn->bd",
                            noises,
                            torch.einsum("bnd,bd->bn", ptb_sols, grad_output))
        grad /= n_samples * sigma
        return grad, None, None, None, None, None, None, None, None


class perturbedFenchelYoung(nn.Module):
    """
    A autograd module for Fenchel-Young loss using perturbation techniques. The
    use of the loss improves the algorithmic by the specific expression of the
    gradients of the loss.

    For the perturbed optimizer, the cost vector need to be predicted from
    contextual data and are perturbed with Gaussian noise.

    The Fenchel-Young loss allows to directly optimize a loss between the features
    and solutions with less computation. Thus, allows us to design an algorithm
    based on stochastic gradient descent.
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            seed (int): random state seed
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
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
        # random state
        self.rnd = np.random.RandomState(seed)
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        self.solpool = None
        if self.solve_ratio < 1: # init solution pool
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = dataset.sols.copy()
        # build optimizer
        self.pfy = perturbedFenchelYoungFunc()

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        loss = self.pfy.apply(pred_cost, true_sol, self.optmodel, self.n_samples,
                              self.sigma, self.processes, self.pool, self.rnd,
                              self.solve_ratio, self)
        return loss


class perturbedFenchelYoungFunc(Function):
    """
    A autograd function for Fenchel-Young loss using perturbation techniques.
    """

    @staticmethod
    def forward(ctx, pred_cost, true_sol, optmodel, n_samples, sigma,
                processes, pool, rnd, solve_ratio, module):
        """
        Forward pass for perturbed Fenchel-Young loss

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            rnd (RondomState): numpy random state
            solve_ratio (float): the ratio of new solutions computed during training
            module (nn.Module): perturbedFenchelYoung module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        w = true_sol.detach().to("cpu").numpy()
        # sample perturbations
        noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        # solve with perturbation
        rand_sigma = np.random.uniform()
        if rand_sigma <= solve_ratio:
            ptb_sols = _solve_in_forward(ptb_c, optmodel, processes, pool)
            if solve_ratio < 1:
                sols = ptb_sols.reshape(-1, cp.shape[1])
                module.solpool = np.concatenate((module.solpool, sols))
        else:
            ptb_sols = _cache_in_pass(ptb_c, optmodel, module.solpool)
        # solution expectation
        e_sol = ptb_sols.mean(axis=1)
        # difference
        if optmodel.modelSense == EPO.MINIMIZE:
            diff = w - e_sol
        if optmodel.modelSense == EPO.MAXIMIZE:
            diff = e_sol - w
        # loss
        loss = np.sum(diff**2, axis=1)
        # convert to tensor
        diff = torch.FloatTensor(diff).to(device)
        loss = torch.FloatTensor(loss).to(device)
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed Fenchel-Young loss
        """
        grad, = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None, None, None, None, None, None, None, None


def _solve_in_forward(ptb_c, optmodel, processes, pool):
    """
    A function to solve optimization in the forward pass
    """
    # number of instance
    n_samples, ins_num = ptb_c.shape[0], ptb_c.shape[1]
    # single-core
    if processes == 1:
        ptb_sols = []
        for i in range(ins_num):
            sols = []
            # per sample
            for j in range(n_samples):
                # solve
                optmodel.setObj(ptb_c[j,i])
                sol, _ = optmodel.solve()
                sols.append(sol)
            ptb_sols.append(sols)
    # multi-core
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        ptb_sols = pool.amap(_solveWithObj4Par, ptb_c.transpose(1,0,2),
                             [args] * ins_num, [model_type] * ins_num).get()
    return np.array(ptb_sols)


def _cache_in_pass(ptb_c, optmodel, solpool):
    """
    A function to use solution pool in the forward/backward pass
    """
    # number of samples & instance
    n_samples, ins_num, _ = ptb_c.shape
    # init sols
    ptb_sols = []
    for j in range(n_samples):
        # best solution in pool
        solpool_obj = ptb_c[j] @ solpool.T
        if optmodel.modelSense == EPO.MINIMIZE:
            ind = np.argmin(solpool_obj, axis=1)
        if optmodel.modelSense == EPO.MAXIMIZE:
            ind = np.argmax(solpool_obj, axis=1)
        ptb_sols.append(solpool[ind])
    return np.array(ptb_sols).transpose(1,0,2)


def _solveWithObj4Par(perturbed_costs, args, model_type):
    """
    A global function to solve function in parallel processors

    Args:
        perturbed_costs (np.ndarray): costsof objective function with perturbation
        args (dict): optModel args
        model_type (ABCMeta): optModel class type

    Returns:
        list: optimal solution
    """
    # rebuild model
    optmodel = model_type(**args)
    # per sample
    sols = []
    for cost in perturbed_costs:
        # set obj
        optmodel.setObj(cost)
        # solve
        sol, _ = optmodel.solve()
        sols.append(sol)
    return sols
