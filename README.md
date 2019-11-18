# einsumt
Multithreaded version of numpy.einsum

# Reasoning
Numpy's einsum is a fantastic function which allows for sophisticated array operations with a single, clear line of code. However, this function in general does not benefit from the underlaying multicore architecture and all operations are performed on a single CPU.

The idea is then to split the einsum input operands along the chosen subscript, perform computation in threads and then compose the final result by summation (if subscript is not present in output) or concatenation of partial results.

# Usage
This function can be used as a replacement for numpy's einsum:

    from einsumt import einsumt as einsum
    result = einsum(*operands, **kwargs)

In current implementation first operand *must* be a subscripts string. Other differences will be treated as unintended bugs.

# Benchmarking
In order to test, if `einsumt` would be beneficial in your particular case please run the benchmark, e.g.:

    import numpy as np
    from einsumt import bench_einsumt

    bench_einsumt('aijk,bkl->ail',
                  np.random.rand(100, 100, 10, 10),
                  np.random.rand(50, 10, 50))

    Platform:           Linux
    CPU type:           Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
    Subscripts:         aijk,bkl->ail
    Shapes of operands: (100, 100, 10, 10), (50, 10, 50)
    Leading index:      automatic
    Pool type:          default
    Number of threads:  12
    Execution time:
        np.einsum:      2755 ms  (average from 1 runs)
        einsumt:        507.9 ms  (average from 5 runs)
    Speed up:           5.424x

More exemplary benchmark calls are contained in bench_einsum.py file.

# Disclaimer
Before you start to blame me because of little or no speedups please keep in mind that threading costs additional time (because of splitting and joining data for example), so `einsumt` function would become beneficial for larger arrays only. Note also that in many cases numpy's einsum can be efficiently replaced with combination of optimized dots, tensordots, matmuls, transpositions and so on, instead of `einsumt` (at cost of code clarity of course).
