r'''>>> import fgm
'''

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.getcwd())
    import doctest, test_fgm
    doctest.testmod(test_fgm)
