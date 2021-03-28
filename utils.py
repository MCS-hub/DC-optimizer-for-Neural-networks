# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:57:38 2021

@author: luu2
"""
import tensorflow as tf


def norm_square(list_of_tensors):
    return sum([tf.pow(tf.norm(list_of_tensors[pst]),2) for pst in range(len(list_of_tensors))])