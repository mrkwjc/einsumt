#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from einsumt import bench_einsumt


print('TEST 0')
a = np.random.rand(100, 100, 10, 10)
b = np.random.rand(50, 10, 50)
bench_einsumt('aijk,bkl->ail', a, b)


print('TEST 1')
a = np.random.rand(10000, 10, 10, 10)
b = np.random.rand(10000, 10, 10, 10)
subs = '...kij,...jik->...ik'
bench_einsumt(subs, a, b)
# same as np.multiply(a, b.transpose(0, 3, 2, 1)).sum(-1).transpose(0, 2, 1)


print('TEST 2')
a = np.random.rand(10000, 10, 10, 10)
b = np.random.rand(10000, 10, 10)
subs = '...ijk,...jk->...ik'
bench_einsumt(subs, a, b)


print('TEST 3')
a = np.random.rand(10000, 10, 10, 10)
b = np.random.rand(10000, 10, 10)
subs = 'aijk,ajk->ik'
bench_einsumt(subs, a, b, idx='a')


print('TEST 4')
a = np.random.rand(10000, 10, 10, 10)
b = np.random.rand(10000, 10, 10)
c = np.random.rand(10, 10, 10)
subs = 'aijk,ajk,kij->ik'
bench_einsumt(subs, a, b, c, idx='a')


print('TEST 5')
a = np.random.rand(10000, 10, 10)
b = np.random.rand(10000, 10, 10)
subs = 'aij,ajk->ik'
bench_einsumt(subs, a, b, idx='a')
# np.matmul(a, b).sum(0) can be much faster then einsumt!


print('TEST 6')
a = np.random.rand(10000, 3, 3, 3, 3, 3)
b = np.random.rand(10000, 3, 6, 3, 3, 3)
subs = 'egmipq, egpqrs -> egmirs'
bench_einsumt(subs, b, a)
