#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from einsumt import einsumt


class TestEinsumt(object):
    def test_summation_reduction(self):
        a = np.random.rand(100, 10, 10)
        b = np.random.rand(10, 10)
        subs = 'aij,ji->ij'
        assert np.allclose(np.einsum(subs, a, b), einsumt(subs, a, b, idx='a'))
    
    def test_concatenation_reduction(self):
        a = np.random.rand(100, 10, 10)
        b = np.random.rand(10, 10)
        subs = 'aij,ji->aij'
        assert np.allclose(np.einsum(subs, a, b), einsumt(subs, a, b, idx='a'))        

    def test_custom_index(self):
        a = np.random.rand(100, 10, 10)
        b = np.random.rand(10, 10)
        subs = 'aij,ji->ij'
        res0 = np.einsum(subs, a, b)
        res1 = einsumt(subs, a, b, idx='a')
        res2 = einsumt(subs, a, b, idx='i')
        res3 = einsumt(subs, a, b, idx='j')
        assert np.allclose(res0, res1)
        assert np.allclose(res0, res2)
        assert np.allclose(res0, res3)

    def test_automatic_index1(self):
        a = np.random.rand(100, 10, 10)
        b = np.random.rand(10, 10)
        subs = 'aij,ji->ij'
        res0 = np.einsum(subs, a, b)
        res1 = einsumt(subs, a, b)
        assert np.allclose(res0, res1)
    
    def test_automatic_index2(self):
        a = np.random.rand(10, 10, 100)
        b = np.random.rand(10, 10)
        subs = 'ija,ji->ij'
        res0 = np.einsum(subs, a, b)
        res1 = einsumt(subs, a, b)
        assert np.allclose(res0, res1)

    def test_ellipsis1(self):
        a = np.random.rand(10, 10, 100)
        b = np.random.rand(10, 10)
        subs = 'ij...,ji->...ij'
        res0 = np.einsum(subs, a, b)
        res1 = einsumt(subs, a, b)
        assert np.allclose(res0, res1)

    def test_ellipsis2(self):
        a = np.random.rand(10, 10, 100)
        b = np.random.rand(10, 10)
        subs = 'ij...,ji->...ij'
        res0 = np.einsum(subs, a, b)
        res1 = einsumt(subs, a, b, idx='j')
        assert np.allclose(res0, res1)

    def test_small_array(self):
        from multiprocessing import cpu_count
        n = cpu_count() - 1
        if n > 0:
            a = np.random.rand(n, 10, 10)
            b = np.random.rand(10, 10)            
            subs = 'aij,ji->ij'
            res0 = np.einsum(subs, a, b)
            res1 = einsumt(subs, a, b, idx='a')        
            assert np.allclose(res0, res1)
        else:
            pytest.skip("not enough CPUs for this test")

    def test_custom_pool(self):
        from multiprocessing.pool import ThreadPool
        a = np.random.rand(100, 10, 10)
        b = np.random.rand(10, 10)
        subs = 'aij,ji->ij'
        res0 = np.einsum(subs, a, b)
        res1 = einsumt(subs, a, b, pool=ThreadPool())
        res2 = einsumt(subs, a, b, pool=5)    
        assert np.allclose(res0, res1)
        assert np.allclose(res0, res2)        

    def test_muti_operand(self):
        e = np.random.rand(50, 10) 
        f = np.random.rand(50, 10, 10)
        g = np.random.rand(50, 10, 20)
        h = np.random.rand(20, 20, 10)
        subs = '...k,...km,...kp,plo->...lom'
        path = np.einsum_path(subs, e, f, g, h)
        res0 = np.einsum(subs, e, f, g, h, optimize=path[0])
        res1 = einsumt(subs, e, f, g, h, optimize=path[0])
        assert np.allclose(res0, res1)        


if __name__ == '__main__':
    import pytest
    pytest.main([str(__file__), '-v'])

    #a = np.random.rand(1000, 10, 10, 10)
    #b = np.random.rand(1000, 10, 10, 10)
    #subs = '...kij,...jik->...ik'
    #res1 = einsump(subs, a, b)
    #res2 = np.einsum(subs, a, b)
    #res3 = np.multiply(a, b.transpose(0, 3, 2, 1)).sum(-1).transpose(0, 2, 1)