# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:36:06 2021

@author: groes
"""

import task1.assassinworld as r1
import numpy as np

def unittest_move_ass():

    britney_loc = np.array([3,3])
    ass_loc = np.array([[1,1]])
    ass_env = r1.AssassinWorld(10, 0.1, ass_loc)
    
    ass_new_loc = ass_env.move_assassins(britney_loc, ass_loc)
    
    assert ass_new_loc[0][0] == 2.0 and ass_new_loc[0][1] == 2.0
    
    britney_loc = np.array([8,8])
    ass_loc = np.array([[1,3], [5,8], [1,9]])
    
    ass_new_loc = ass_env.move_assassins(britney_loc, ass_loc)
    
    assert (ass_new_loc[0] == np.array([2,4])).all()
    assert (ass_new_loc[1] == np.array([6,8])).all()
    assert (ass_new_loc[2] == np.array([2,8])).all()

unittest_move_ass()