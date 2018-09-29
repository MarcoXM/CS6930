#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:00:26 2018

@author: marcowang
"""

import numpy as np
import pandas as pd 
df1 = pd.read_csv('/Users/marcowang/Downloads/spam_train.csv')
df1.head()
X1_train = df1.drop('class').values
y1_train = df1['class'].values
