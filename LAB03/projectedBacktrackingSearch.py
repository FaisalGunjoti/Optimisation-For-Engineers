# Optimization for Engineers - Dr.Johannes Hild
# projected backtracking line search

# Purpose: Find t to satisfy f(x+t*d)< f(x) - sigma/t*norm(x-P(x - t*gradient))**2

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# P: box projection class with method .project()
# x: column vector in R**n (domain point)
# d: column vector in R**n (search direction)
# sigma: value in (0,1), marks quality of decrease. Default value: 1.0e-4.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
# should return t = 0.5

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23006558
    return matrnr


def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, verbose=0):
    xp = P.project(x)
    gradx = f.gradient(xp)
    decrease = gradx.T @ d

    if decrease >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 1:
        raise TypeError('range of sigma is wrong!')

    if verbose:
        print('Start projectedBacktrackingSearch...')

    beta = 0.5
    t = 1
    # INCOMPLETE CODE STARTS
    def W1(t):
        left = f.objective(P.project(xp + t * d))
        right = f.objective(xp) - (sigma / t) * np.linalg.norm(xp - P.project(xp - t * gradx))**2
        return left <= right
    
    while not W1(t):
        t *= beta
                
    t_star = t
    t = np.copy(t_star)


    # INCOMPLETE CODE ENDS

    if verbose:
        print('projectedBacktrackingSearch terminated with t=', t)

    return t


