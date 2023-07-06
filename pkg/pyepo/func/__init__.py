#!/usr/bin/env python
# coding: utf-8
"""
Pytorch autograd function for SPO training
"""

from pyepo.func.blackbox import blackboxOpt
from pyepo.func.spoplus import SPOPlus
from pyepo.func.perturbed import perturbedOpt, perturbedFenchelYoung
from pyepo.func.nce import NCE, NCE_MAP
from pyepo.func.learning_to_rank import ListwiseLTR, PairwiseLTR, PointwiseLTR
