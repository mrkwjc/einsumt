#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Aug 10 09:34:57 2019

@author: Marek Wojciechowski
@github: mrkwjc
@licence: MIT
"""
from __future__ import print_function
import numpy as np
from multiprocessing.pool import ThreadPool

alphabet = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
default_thread_pool = ThreadPool()


def einsumt(*operands, **kwargs):
    """
    Multithreaded version of numpy.einsum function.
    
    The additional accepted keyword arguments are:

        pool - specifies the pool of processing threads
               
               If 'pool' is None or it is not given, the default pool with the
               number of threads equal to CPU count is used. If 'pool' is an integer,
               then it is taken as the number of threads and the new fresh pool is
               created. Otherwise, the 'pool' attribute is assumed to be
               a multiprocessing.pool.ThreadPool instance.
               
        idx  - specifies the subscript along which operands are divided into chunks
    
               Argument 'idx' have to be a single subscript letter, and should
               be contained in the given subscripts, otherwise ValueError is risen.
               If 'idx' is None or it is not given, then the longest dimension
               in operands is searched.

    WARNING: Current implementation allows for string subscripts specification only
    """
    pool = kwargs.pop('pool', None)
    idx = kwargs.pop('idx', None)
    # If single processor fall back to np.einsum
    if (pool == 1 or 
        (pool == None and default_thread_pool._processes == 1) or
        (hasattr(pool, '_processes') and pool._processes == 1)):
        return np.einsum(*operands, **kwargs)
    # Assign default thread pool if necessary and get number of threads
    if not hasattr(pool, 'apply_async'):
        pool = default_thread_pool  # mp.pool.ThreadPool(pool)
    nproc = pool._processes
    # Out is popped here becaue it is not used in threads but at return only
    out = kwargs.pop('out', None) 
    # Analyze subs and ops
    # isubs, osub, ops = np.core.einsumfunc._parse_einsum_input(operands)
    subs = operands[0]
    ops = operands[1:]
    iosubs = subs.split('->')
    isubs = iosubs[0].split(',')
    osub = iosubs[1] if len(iosubs) == 2 else ''
    is_ellipsis = '...' in subs
    if is_ellipsis:
        indices = subs.replace('->', '').replace(',', '').replace('...', '')
        free_indices = ''.join(alphabet - set(indices))
        sis = []
        for si, oi in zip(isubs, ops):
            if '...' in si:
                ne = oi.ndim - len(si.replace('...', ''))  # should be determined once?
                si = si.replace('...', free_indices[:ne])
            sis.append(si)
        isubs = sis
        osub = osub.replace('...', free_indices[:ne])  # ne should be always the same
    # Get index along which we will chunk operands
    # If not given we try to search for longest dimension
    if idx is not None:  # and idx in indices...
        if idx not in iosubs[0]:
            raise ValueError("Index '%s' is not present in input subscripts" % idx)
        cidx = idx  # given index for chunks
        cdims = []
        for si, oi in zip(isubs, ops):
            k = si.find(cidx)
            cdims.append(oi.shape[k] if k >=0 else 0)
        if len(set(cdims)) > 2:  # set elements can be 0 and one number
            raise ValueError("Different operand lengths along index '%s'" % idx)
        cdim = max(cdims)  # dimension along cidx
    else:
        maxdim = []
        maxidx = []
        for si, oi in zip(isubs, ops):
            mdim = max(oi.shape)
            midx = si[oi.shape.index(mdim)]
            maxdim.append(mdim)
            maxidx.append(midx)
        cdim = max(maxdim)                    # max dimension of input arrays
        cidx = maxidx[maxdim.index(cdim)]     # index cdim -> chosen index for chunks
    # Position of established index in subscripts
    cpos = [si.find(cidx) for si in isubs]  # positions of index in input operands shape
    opos = osub.find(cidx)                  # position of index in output
    ##
    # Determining chunk ranges
    n, r = divmod(cdim, nproc)  # n - chunk size, r - rest
    # Create chunks and apply np.einsum
    n1 = 0
    n2 = 0
    cpos_slice = [(slice(None),)*c for c in cpos]
    njobs = r if n == 0 else nproc
    res = []
    for i in range(njobs):
        args = (subs,)
        n1 = n2
        n2 += n if i>=r else n+1
        islice = slice(n1, n2)
        for j in range(len(ops)):
            oj = ops[j]
            if cpos[j] >= 0:
                jslice = cpos_slice[j] + (islice,)
                oj = oj[jslice]
            args = args + (oj,)
        res += [pool.apply_async(np.einsum, args=args, kwds=kwargs)]
    res = [r.get() for r in res]
    ###
    ## Determine number of jobs
    #n, r = divmod(cdim, nproc)  # n - chunk size, r - rest
    #njobs = r if n == 0 else nproc    
    ## Split operands along chosen axis
    #ops_split = []
    #for j in range(len(ops)):
        #if cpos[j] < 0:
            #ops_split.append([ops[j]]*njobs)
        #else:
            #ops_split.append(np.array_split(ops[j], njobs, axis=cpos[j]))
    ## Submit jobs to pool
    #res = []
    #for opsi in zip(*ops_split):
        #args = (subs,) + opsi
        #res += [pool.apply_async(np.einsum, args=args, kwds=kwargs)]
    ## Get results
    #res = [r.get() for r in res]
    ###
    # Reduce
    if opos < 0:  # cidx not in output subs, reducing
        res = np.sum(res, axis=0)
    else:
        res = np.concatenate(res, axis=opos)
    # Handle 'out' and return
    if out is not None:
        out[:] = res
    else:
        out = res
    return res


def bench_einsumt(*operands, **kwargs):
    """
    Benchmark function for einsumt.
    
    This function returns a tuple 'res' where res[0] is the execution time
    for np.einsum and res[1] is the execution time for einsumt in miliseconds.
    In addition this information is printed to the screen, unless the keyword
    argument pprint=False is set.
    
    This function accepts all einsumt arguments.
    """
    from time import time
    import platform
    # Prepare kwargs for einsumt
    pprint = kwargs.pop('pprint', True)
    # Preprocess kwargs 
    kwargs1 = kwargs.copy()
    pool = kwargs1.pop('pool', None)
    if pool is None:
        nproc = default_thread_pool._processes
        ptype = 'default'
    elif isinstance(pool, int):
        nproc = pool
        ptype = 'custom'
    else:
        nproc = pool._processes
        ptype = 'custom'
    idx = kwargs1.pop('idx', None)
    # np.einsum timing
    t0 = time()
    np.einsum(*operands, **kwargs1)
    dt1 = time() - t0
    N1 = int(divmod(2., dt1)[0])  # we assume 2s of benchmarking
    t0 = time()
    for i in range(N1):
        res1 = np.einsum(*operands, **kwargs1)
    dt1 += time() - t0
    T1 = 1000*dt1/(N1+1)
    # einsumt timing
    t0 = time()
    einsumt(*operands, **kwargs)
    dt2 = time() - t0
    N2 = int(divmod(2., dt2)[0])  # we assume 2s of benchmarking
    t0 = time()
    for i in range(N2):
        res1 = einsumt(*operands, **kwargs1)
    dt2 += time() - t0
    T2 = 1000*dt2/(N2+1)
    # printing
    if pprint:
        print('Platform:           %s' % platform.system())
        print('CPU type:           %s' % _get_processor_name())
        print('Subscripts:         %s' % operands[0])
        print('Shapes of operands: %s' % str([o.shape for o in operands[1:]])[1:-1])
        print('Leading index:      %s' % (idx if idx is not None else 'automatic'))
        print('Pool type:          %s' % ptype)
        print('Number of threads:  %i' % nproc)
        print('Execution time:')
        print('    np.einsum:      %1.4g ms  (average from %i runs)' % (T1, N1+1))
        print('    einsumt:        %1.4g ms  (average from %i runs)' % (T2, N2+1))
        print('Speed up:           %1.3fx' % (T1/T2,))
        print('')
    return T1, T2


def _get_processor_name():
    import os, platform, subprocess, re, sys
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        info = subprocess.check_output(command)
        if sys.version_info[0] >= 3:
            info = info.decode()
        return info.strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True)
        if sys.version_info[0] >= 3:
            all_info = all_info.decode()
        for line in all_info.strip().split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1).strip()
    return ""


if __name__=="__main__":
    import pytest
    pytest.main(['test_einsumt.py', '-v'])
