'''
Author: Loïc Balazi
Date: November 2024
Supplementary material for paper: 
"Inf-sup stable non-conforming finite elements on tetrahedra with second and third order accuracy"
by L. Balazi, G. Allaire, P. Jolivet, P. Omnes 
'''

'''
Python's code for searching all the suitable basis. 
This script returns all the subspaces \Sigma_{n+2} of P_{n+2} that allows to complete the space P_{n+1}
'''


'''
Let K be a tetrahedron with faces F1, F2, F3 and F4.

Let \lambda_1, \lambda_2, \lambda_3 and \lambda_4 be the four barycentric coordinates of K

A monomial P = \lambda_1^{a_1}*\lambda_2^{a_2}*\lambda_3^{a_3}*\lambda_4^{a_4}  is represented by the four exponants i.e. P = [a_1,a_2,a_3,a_4]

We recall that V_{n+2} = P_{n+1} + \Sigma_{n+2} with \Sigma_{n+2} a subspace of P_{n+2} of size n*(n+2).

We assume that \Sigma_{n+2} consists only of monomials of the form \lambda_1^{a_1}*\lambda_2^{a_2}*\lambda_3^{a_3}*\lambda_4^{a_4} with a_1 + a_2 + a_3 + a_4 = n+2 (monomials of degree n+2).


The integrals of products of barycentric coordinates are computed using the following integration formula [1]
$$ \int_D \prod_{i=1}^{d+1} \lambda_i^{a_i} = \frac{\prod_{i=1}^{d+1} a_i!}{ (d + \sum_{i=1}^{d+1} a_i)!}  $$
where D is a unit simplex in R^d

[1]  F. Vermolen and A. Segal. On an integration rule for products of barycentric coordinates over simplexes in Rn. Journal of Computational and Applied Mathematics, 330:289–294, Mar. 2018.
'''


from itertools import combinations
import numpy as np
from fractions import Fraction
from sympy import Matrix
import math
import pickle


n = 1  #corresponds to the n in V_{n+1} = P_{n+1} + \sigma_{n+2}


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                           DEFINITION OF THE BASES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

if n == 1 :
    
    DofF1=[[0,1,0,0],[0,0,1,0],[0,0,0,1]] # Basis of P_1(F_1):  Basis of Lagrange P_1 FE on the face F_1
    DofF2=[[1,0,0,0],[0,0,1,0],[0,0,0,1]] # Basis of P_1(F_2):  Basis of Lagrange P_1 FE on the face F_2
    DofF3=[[1,0,0,0],[0,1,0,0],[0,0,0,1]] # Basis of P_1(F_3):  Basis of Lagrange P_1 FE on the face F_3
    DofF4=[[1,0,0,0],[0,1,0,0],[0,0,1,0]] # Basis of P_1(F_4):  Basis of Lagrange P_1 FE on the face F_4

    DofELEM=[[0,0,0,0]]  # Basis of P_0(K):  Basis of Lagrange P_0 FE in the tetrahedron K

    BasisPn1=[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2],[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1]]  # Basis of P_2(K):  Basis of Lagrange P_2 FE in the tetrahedron K


if n == 2 :
   
    DofF1=[[0,1,1,0],[0,1,0,1],[0,0,1,1],[0,2,0,0],[0,0,2,0],[0,0,0,2]]   # Basis of P_2(F_1):  Basis of Lagrange P_2 FE on the face F_1
    DofF2=[[1,0,1,0],[1,0,0,1],[0,0,1,1],[2,0,0,0],[0,0,2,0],[0,0,0,2]]   # Basis of P_2(F_2):  Basis of Lagrange P_2 FE on the face F_2
    DofF3=[[1,1,0,0],[1,0,0,1],[0,1,0,1],[2,0,0,0],[0,2,0,0],[0,0,0,2]]   # Basis of P_2(F_3):  Basis of Lagrange P_2 FE on the face F_3
    DofF4=[[1,1,0,0],[1,0,1,0],[0,1,1,0],[2,0,0,0],[0,2,0,0],[0,0,2,0]]   # Basis of P_2(F_3):  Basis of Lagrange P_2 FE on the face F_4

    DofELEM= [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]  # Basis of P_1(K):  Basis of Lagrange P_1 FE in the tetrahedron K
   
    BasisPn1=[[3,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,3],
              [1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],
              [2,1,0,0],[1,2,0,0], [2,0,1,0], [1,0,2,0], [2,0,0,1],   
              [1,0,0,2], [0,2,1,0], [0,1,2,0], [0,2,0,1],[0,1,0,2], [0,0,2,1], [0,0,1,2]]    # Basis of P_3(K):  Basis of Lagrange P_3 FE in the tetrahedron K

if n==3 :

    DofF1=[ [0, 3, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 3, 0], [0, 2, 0, 1 ],
            [0, 1, 1, 1], [0, 0, 2, 1], [0, 1, 0, 2], [0, 0, 1, 2], [0, 0, 0, 3] ]   # Basis of P_3(F_1):  Basis of Lagrange P_3 FE on the face F_1

    DofF2=[ [3, 0, 0, 0], [2, 0, 1, 0], [1, 0, 2, 0], [0, 0, 3, 0], [2, 0, 0, 1],
            [1, 0, 1, 1], [0, 0, 2, 1], [1, 0, 0, 2], [0, 0, 1, 2], [0, 0, 0, 3] ]   # Basis of P_3(F_2):  Basis of Lagrange P_3 FE on the face F_2

    DofF3=[ [3, 0, 0, 0], [2, 1, 0, 0], [1, 2, 0, 0], [0, 3, 0,0], [2, 0, 0, 1],
                     [1, 1, 0, 1], [0, 2, 0, 1], [1, 0, 0, 2], [0, 1, 0, 2],[0, 0, 0, 3] ]   # Basis of P_3(F_3):  Basis of Lagrange P_3 FE on the face F_3

    DofF4=[ [3, 0, 0, 0], [2, 1, 0, 0], [1, 2, 0, 0], [0, 3, 0, 0], [2, 0, 1, 0],
                     [1, 1, 1, 0], [0, 2, 1, 0], [1, 0, 2, 0], [0, 1, 2, 0], [0, 0, 3,  0] ]  # Basis of P_3(F_4):  Basis of Lagrange P_3 FE on the face F_4

  
    DofELEM=[ [2, 0, 0, 0], [1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0],
                       [0, 0, 2, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 2] ]   # Basis of P_2(K):  Basis of Lagrange P_2 FE in the tetrahedron K


    BasisPn1=[ [4, 0, 0, 0], [3, 1, 0, 0], [2, 2, 0, 0], [1, 3, 0, 0], [0, 4, 0, 0], [3, 0, 1, 0], [2, 1, 1, 0], [1, 2, 1, 0],
               [0, 3, 1, 0], [2, 0, 2, 0], [1, 1, 2, 0], [0, 2, 2, 0], [1, 0, 3, 0], [0, 1, 3, 0], [0, 0, 4, 0], [3, 0, 0, 1],
               [2, 1, 0, 1], [1, 2, 0, 1], [0, 3, 0, 1], [2, 0, 1, 1], [1, 1, 1, 1], [0, 2, 1, 1], [1, 0, 2, 1], [0, 1, 2, 1],
               [0, 0, 3, 1], [2, 0, 0, 2], [1, 1, 0, 2], [0, 2, 0, 2], [1, 0, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2], [1, 0, 0, 3],
               [0, 1, 0, 3], [0, 0, 1, 3], [0, 0, 0, 4] ]                              # Basis of P_3(K):  Basis of Lagrange P_3 FE in the tetrahedron K




'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                           DIMENSION OF THE SPACES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


DofFACE=np.concatenate((DofF1, DofF2, DofF3, DofF4))   
DofELEM=np.array(DofELEM)

nDofELEM=len(DofELEM)                          #Number of degrees of freedom in the element
nDofFACE=len(DofFACE)                          #Number of degrees of freedom on the faces
nDofFi = len(DofF1)                            #Number of degrees of freedom on one face
nDofTOTAL=nDofFACE+nDofELEM                    #Number total of degrees of freedom 

dimVn1 = n*(n+1)*(n+2)//6 + 4*(n+1)*(n+2)//2   #Dimension of the space V_{n+1}
dimPn1 = (n+2)*(n+3)*(n+4)//6                  #Dimension of the space P_{n+1}
dimSIGMAn2 = n*(n+2)                           #Dimension of the space \Sigma_{n+2}
dimPn2 = (n+3)*(n+4)*(n+5)//6                  #Dimension of the space P_{n+2}


assert(len(BasisPn1) == dimPn1)        
assert(nDofTOTAL == dimVn1)



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                            USEFUL FUNCTIONS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def GenerateAllMonomials():

    """ 
    
    This function generates all the lists [a_1, a_2, a_3, a_4] with a_1 + a_2 + a_3 + a_4 = n + 2, i.e. all the monomials of degree n+2 (a basis of P_{n+2})

    INPUT:
    - n : where n+1 is the order of the non-conforming finite element

    OUTPUT:
    - AllMonomials : a list which contains all the monomials of degree n+2
    
    """

    AllMonomials=[]
    for i in range(0,n+3):                                 #(0, .... n+2)
        for j in range(0, n+3-i):
            for k in range(0,n+3-i-j):
                basis=[i,j,k,n+2-i-j-k]
                AllMonomials.append(basis)
    assert(len(AllMonomials) == (n+3)*(n+4)*(n+5)/6)
    return AllMonomials



def GenerateAllCombinations():

    """
    This function allows to generate all sub-lists of size dimSIGMAn2 among the enumerations of the monomials of degree n+2 (0, ..., dim(P_{n+2})-1]

    OUTPUT : The list containing all the possible combinations
    """

    AllCombinations=[]
    nbre_combinations =0
    facsec=3e7 
    for combination in combinations(range(dimPn2), dimSIGMAn2):
        nbre_combinations +=1
        AllCombinations.append(combination)
        if nbre_combinations>facsec:                                #stop if too many combinations
            break
    return AllCombinations


def ComputeFEmatrix(BasisVn1):

    """

    This function build the matrix N_i(\Phi_j) for   1<=i,j<= dim(V_{n+1}) = card(\mathcal{N}_{n+1})

    INPUT :
     -DofFace : the set of DoFs associated to the faces
     -DofElem : the set of DoFs associated to the element
     -BasisVn1 :  The basis of the space V_{n+1}

     OUTPUT :
      -  M : the matrix N_i(\Phi_j)

    """

    # Initialize matrix with Fractions
    M = [[Fraction(0) for _ in range(dimVn1)] for _ in range(dimVn1)]

    # First loop: degrees of freedom on the faces 
    for i in range(nDofFACE):
        for j in range(dimVn1):
            alpha =  DofFACE[i] + BasisVn1[j]    #Multiplication DofELEM[i]* BasisVn1[j]  ==> addition of the exponents
            coeff = (Fraction(math.factorial(alpha[0])) * 
                    Fraction(math.factorial(alpha[1])) * 
                    Fraction(math.factorial(alpha[2])) * 
                    Fraction(math.factorial(alpha[3]))) / Fraction(math.factorial(sum(alpha) + 2))
            coeff *= (BasisVn1[j][i // nDofFi] == 0)
            M[i][j] +=  coeff 

    # Second loop: degrees of freedom in the element
    for i in range(nDofELEM):
        index = i + nDofFACE
        for j in range(dimVn1):
            alpha = DofELEM[i] + BasisVn1[j]  #Multiplication DofELEM[i]* BasisVn1[j]  ==> addition of the exponents 
            coeff = (Fraction(math.factorial(alpha[0])) * 
                      Fraction(math.factorial(alpha[1])) * 
                      Fraction(math.factorial(alpha[2])) * 
                      Fraction(math.factorial(alpha[3]))) / Fraction(math.factorial(sum(alpha) + 3))
            M[index][j] +=  coeff 

    # Convert to numpy array with dtype=object for Fraction
    M = np.array(M, dtype=object)
    return M



def TestUnisolvence(MAT):
    """
    INPUT : Matrix MAT

    OUTPUT :
      - BOOL = True if unisolvent
    """

    determinant = Matrix(MAT).det()  #Convert in Matrix Sympy

    if determinant != 0 :
        return True
    else:
        return False


def GenerateAllSigma():
    
    """
    
    This function generates a list all of spaces \Sigma_{n+2} that are suitable to complete P_{n+1}

    INPUT :
        - The list of all monomials of order n+2 of size (n+3)(n+4)/2
        - The list all of combinations of n*(n+2)= dim(\Sigma_{n+2}) elements among (n+3)(n+4)/2 elements

    OUTPUT :
        - A list with all space \Sigma_{n+2} that are suitable to complete P_{n+1}

    """
    
    AllMonomials    =  GenerateAllMonomials()
    AllCombinations = GenerateAllCombinations()
    N_combi = len(AllCombinations)
    i1=0
    i2=0
    AllBasisSIGMAn2=[]
    for combination in AllCombinations:
        i1+=1
        BasisVn1=BasisPn1.copy()
        BasisSIGMAn2=[]
        for j in range(0,dimSIGMAn2):
            function = AllMonomials[combination[j]]
            BasisSIGMAn2.append(function)
            BasisVn1.append(function)
        FEMatrix = ComputeFEmatrix(BasisVn1)
        if TestUnisolvence(FEMatrix):
            i2+=1
            AllBasisSIGMAn2.append(BasisSIGMAn2)
        print(f"Test {i1} / {N_combi} . Basis ok : {i2} ")
    N2 = n+2
    with open(f"AllBasisSigma{N2}", "wb") as fp:
        pickle.dump(AllBasisSIGMAn2, fp)



def GenerateAllSigmaRandom():
     
    """
    
    This function generates a list all of spaces \Sigma_{n+2} that are suitable to complete P_{n+1}

    INPUT :
        - The list of all monomials of order n+2 of size (n+3)(n+4)/2
        - The list all of combinations of n*(n+2)= dim(\Sigma_{n+2}) elements among (n+3)(n+4)/2 elements

    OUTPUT :
        - A list with all space \Sigma_{n+2} that are suitable to complete P_{n+1}

    Contrary to the previous functions, not all combinations are tested but random combinations

    """

    N_combi  = 100000
    i1=0
    i2=0
    AllBasisSIGMAn2=[]
    AllMonomials=GenerateAllMonomials()
    for i in range(Ntest):
        combination = np.random.choice(range(dimPn2), dimSIGMAn2, replace = False)
        i1+=1
        BasisVn1=BasisPn1.copy()
        BasisSIGMAn2=[]
        for j in range(0,dimSIGMAn2):
            function = AllMonomials[combination[j]]
            BasisSIGMAn2.append(function)
            BasisVn1.append(function)
        FEMatrix = ComputeFEmatrix(BasisVn1)
        if TestUnisolvence(FEMatrix):
            i2+=1
            AllBasisSIGMAn2.append(BasisSIGMAn2)
            print(f"Test {i1} / {Ntest} . Basis ok : {i2} ")
            print(combination)

    print(f"Number of basis found = {len(AllBasisSIGMAn2)}")
    N2 = n+2
    with open(f"AllBasisSigma{N2}", "wb") as fp:
        pickle.dump(AllBasisSIGMAn2, fp)

########################################################## MAIN ###################################################
GenerateAllSigma()

N2 = n+2
with open(f"AllBasisSigma{N2}", "rb") as fp:
    AllBasis = pickle.load(fp)
    print(AllBasis)