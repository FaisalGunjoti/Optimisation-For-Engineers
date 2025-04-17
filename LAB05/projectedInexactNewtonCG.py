'''# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA
import simpleValleyObjective as S
import projectionInBox as PP
def matrnr():
    # set your matriculation number here
    matrnr = 23006558
    return matrnr


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp = P.project(x0)
    # INCOMPLETE CODE START
    
    check_eps =  np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
    eta_k = np.min([0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp))))]) * np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
    while check_eps > eps:
        xj = xp.copy()
        rj = f.gradient(xp).copy()
        dj = -rj.copy()
        curvature_fail = False
        while np.linalg.norm(rj) > eta_k:
            dA = PHA.projectedHessApprox(f,P, xj, dj)  
            rho_j = dj.T @ dA
            if rho_j <= eps * (np.linalg.norm(dj)**2):
                curvature_fail = True
                break
            
            tj = (np.linalg.norm(rj)**2) / rho_j
            xj = xj + tj * dj
            r_old = rj
            rj = r_old + tj * dA
            beta_j = np.square(np.linalg.norm(rj)/(np.linalg.norm(r_old)))
            dj = -rj + beta_j * dj
        
        if curvature_fail and np.allclose(xj, xp):
            dk = -f.gradient(xp)
            
        else:
            dk = xj - xp  
        
        t = PB.projectedBacktrackingSearch(f, P, xp, dk)   
        xp = P.project(xp + t * dk)
        eta_k = np.min([0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp))))]) * np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
        countIter +=1
        
    check_eps = np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
    eta_k = np.min([0.5, np.sqrt(check_eps)]) * check_eps

    while check_eps > eps:
        xj = xp.copy()
        rj = f.gradient(xp).copy()
        dj = -rj.copy()
        curvature_fail = False

        while np.linalg.norm(rj) > eta_k:
            dA = PHA.projectedHessApprox(f, P, xj, dj)
            rho_j = dj.T @ dA
            if rho_j <= eps * np.square(np.linalg.norm(dj)):
                curvature_fail = True
                break

            tj = (np.linalg.norm(rj) ** 2) / rho_j
            xj = xj + tj * dj
            r_old = rj
            rj = r_old + tj * dA
            beta_j = np.square(np.linalg.norm(rj) / np.linalg.norm(r_old))
            dj = -rj + beta_j * dj

        if curvature_fail :
            dk = -f.gradient(xp)
        else :
            if not np.allclose(xj, xp):
                dk = xj - xp
            else:
                dk = -f.gradient(xp)

        t = PB.projectedBacktrackingSearch(f, P, xp, dk)
        xp = P.project(xp + t * dk)

        check_eps = np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
        eta_k = np.min([0.5, np.sqrt(check_eps)]) * check_eps
        countIter += 1
    x_star = xp
    xp = np.copy(x_star)    
         

    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp'''


# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA

def matrnr():
    # set your matriculation number here
    matrnr =  23006558
    return matrnr


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    # INCOMPLETE CODE STARTS
    countIter = 0
    xp = P.project(x0)
    eta_k = np.min((0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp)))))) * np.linalg.norm(xp - P.project(xp - f.gradient(xp)))

    while np.linalg.norm(xp - P.project(xp - f.gradient(xp))) > eps:
        x_j = xp.copy()
        #x_j_updated = False
        r_j = f.gradient(xp).copy()
        d_j = -r_j.copy()
        curvaturefail = False
        while np.linalg.norm(r_j) > eta_k:
            d_a = PHA.projectedHessApprox(f, P, xp, d_j)
            rho_j = d_j.T @ d_a
            if rho_j <= eps * np.square(np.linalg.norm(d_j)):
                curvaturefail = True
                break
            t_j = np.square(np.linalg.norm(r_j)) / rho_j
            x_j = x_j + t_j * d_j
            r_old = r_j
            r_j = r_old + t_j * d_a
            beta_j = np.square(np.linalg.norm(r_j)/np.linalg.norm(r_old))
            d_j = -r_j + beta_j * d_j
        if curvaturefail :
            d_k = -f.gradient(xp)
        else :
            #d_k = x_j - xp
            if not np.allclose(x_j, xp):
                d_k = x_j - xp
            else:
                d_k = -f.gradient(xp)
            
        t = PB.projectedBacktrackingSearch(f, P, xp, d_k)
        xp = P.project(xp+ t*d_k)
        eta_k = np.min((0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp)))))) * np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
        countIter = countIter + 1
        
    x_star = xp
    xp = np.copy(x_star)       
    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp
