# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:46:54 2021

@author: hugha
"""

from learning_environment import learning_environment


"""
10x10 britneyworld stumble prob = 0.2
1. vanilla sac
1. sac + entropy tuning
3. sac + eer
4. sac + entropy tuning + ere
"""


learning_environment(10000, ere = False)

learning_environment(10000, ere = False)
