r'''>>> import numpy as np
    >>> import fgm
	>>> m = np.matrix('1.0 2; 3 4')
    >>> print fgm.double_matrix(m)
    [[ 2.  4.]
     [ 6.  8.]]
'''

if __name__ == '__main__':
    import doctest, test_fgm
    doctest.testmod(test_fgm)
