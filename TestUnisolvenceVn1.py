
'''
Author: Loïc Balazi
Date: November 2024
Supplementary material for paper: 
"Inf-sup stable non-conforming finite elements on tetrahedra with second and third order accuracy"
by L. Balazi, G. Allaire, P. Jolivet, P. Omnes 
'''

'''
Python's code for demonstrating the unisolvence of the finite element space V_2 and V_3
presented in Lemma 3.4.
'''

'''
Let K be a tetrahedron with faces F1, F2, F3 and F4.

Let \lambda_1, \lambda_2, \lambda_3 and \lambda_4 be the four barycentric coordinates of K

A monomial P = \lambda_1^{a_1}*\lambda_2^{a_2}*\lambda_3^{a_3}*\lambda_4^{a_4}  is represented by the four exponants i.e. P = [a_1,a_2,a_3,a_4]

The integrals of products of barycentric coordinates are computed using the following integration formula [1]
$$ \int_D \prod_{i=1}^{d+1} \lambda_i^{a_i} = \frac{\prod_{i=1}^{d+1} a_i!}{ (d + \sum_{i=1}^{d+1} a_i)!}  $$
where D is a unit simplex in R^d

[1]  F. Vermolen and A. Segal. On an integration rule for products of barycentric coordinates over simplexes in Rn. Journal of Computational and Applied Mathematics, 330:289–294, Mar. 2018.
'''


from fractions import Fraction
from sympy import Matrix
import numpy as np
import math


n = 1  #corresponds to the n in V_{n+1} = P_{n+1} + \sigma_{n+2}


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                           DEFINITION OF THE BASES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

if n == 1 :
  
  '''
  V_2 = P_2 + span{\lambda_1 \lambda_2^2, \lambda_1 \lambda_3^2, \lambda_2 \lambda_3^2}
  '''

  DofF1 = [[0,1,0,0],[0,0,1,0],[0,0,0,1]]   # Basis of P_1(F_1):  Basis of Lagrange P_1 FE on the face F_1
  DofF2 = [[1,0,0,0],[0,0,1,0],[0,0,0,1]]   # Basis of P_1(F_2):  Basis of Lagrange P_1 FE on the face F_2
  DofF3 = [[1,0,0,0],[0,1,0,0],[0,0,0,1]]   # Basis of P_1(F_3):  Basis of Lagrange P_1 FE on the face F_3
  DofF4 = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]   # Basis of P_1(F_4):  Basis of Lagrange P_1 FE on the face F_4
  
  DofELEM = [[0,0,0,0]]           # Basis of P_0(K):  Basis of Lagrange P_0 FE on the tetrahedra K
  
  BasisPn1 = [[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2],[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1]]  # Basis of P_2(K):  Basis of Lagrange P_2 FE on the tetrahedra K
  
  BasisSigmaN2 = [[1,2,0,0],[1,0,2,0],[0,1,2,0]] # Basis of Sigma_3(K)


if n == 2 :
  
  '''
  V_3 = P_3 + span{\lambda_1^3 \lambda_2, \lambda_2^3 \lambda_3, \lambda_3^3 \lambda_4, \lambda_4^3 \lambda_1, \lambda_2^3 \lambda_1, \lambda_1^3 \lambda_4, \lambda_4^3 \lambda_3, \lambda_3^3 \lambda_2}
  '''
  
  DofF1 = [[0,1,1,0],[0,1,0,1],[0,0,1,1],[0,2,0,0],[0,0,2,0],[0,0,0,2]]   # Basis of P_2(F_1):  Basis of Lagrange P_2 FE on the face F_1
  DofF2 = [[1,0,1,0],[1,0,0,1],[0,0,1,1],[2,0,0,0],[0,0,2,0],[0,0,0,2]]   # Basis of P_2(F_2):  Basis of Lagrange P_2 FE on the face F_2
  DofF3 = [[1,1,0,0],[1,0,0,1],[0,1,0,1],[2,0,0,0],[0,2,0,0],[0,0,0,2]]   # Basis of P_2(F_3):  Basis of Lagrange P_2 FE on the face F_3
  DofF4 = [[1,1,0,0],[1,0,1,0],[0,1,1,0],[2,0,0,0],[0,2,0,0],[0,0,2,0]]   # Basis of P_2(F_4):  Basis of Lagrange P_2 FE on the face F_4

  DofELEM = [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]                     # Basis of P_1(K):  Basis of Lagrange P_1 FE on the tetrahedra K

  BasisPn1 = [[3,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,3],
              [1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],
              [2,1,0,0],[1,2,0,0],[2,0,1,0],[1,0,2,0],
              [2,0,0,1],[1,0,0,2],[0,2,1,0],[0,1,2,0],
              [0,2,0,1],[0,1,0,2],[0,0,2,1],[0,0,1,2]]   # Basis of P_3(K):  Basis of Lagrange P_3 FE on the tetrahedra K

  BasisSigmaN2 = [[3,1,0,0],[0,3,1,0],[0,0,3,1],[1,0,0,3],[1,3,0,0],[3,0,0,1],[0,0,1,3],[0,1,3,0]]  # Basis of Sigma_4(K)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                          CONSTRUCTION OF THE SPACES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


DofFACE=np.concatenate((DofF1, DofF2, DofF3, DofF4))
DofELEM=np.array(DofELEM)

nDofELEM=len(DofELEM)
nDofFACE=len(DofFACE)
nDofFi = len(DofF1)
nDofTOTAL=nDofFACE+nDofELEM
dimVn1 = n*(n+1)*(n+2)//6 + 4*(n+1)*(n+2)//2
dimPn1 = (n+2)*(n+3)*(n+4)//6
dimSIGMAn2 = n*(n+2)


BasisVn1=BasisPn1.copy()
for i in range(0,dimSIGMAn2):
    BasisVn1.append(BasisSigmaN2[i])

assert(len(BasisPn1) == dimPn1)
assert(len(BasisVn1)== dimVn1)
assert(nDofTOTAL == dimVn1)


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

  # First loop: DofFACE
  for i in range(nDofFACE):
    for j in range(dimVn1):
      alpha = DofFACE[i] + BasisVn1[j]   #Multiplication DofELEM[i]*BasisVn1[j] ==> addition of the exponents
    
      coeff = (Fraction(math.factorial(alpha[0])) * 
              Fraction(math.factorial(alpha[1])) * 
              Fraction(math.factorial(alpha[2])) * 
              Fraction(math.factorial(alpha[3]))) / Fraction(math.factorial(sum(alpha) + 2))
      coeff *= (BasisVn1[j][i // nDofFi] == 0)
      M[i][j] +=  coeff 

  # Second loop: DofELEM
  for i in range(nDofELEM):
    index = i + nDofFACE
    for j in range(dimVn1):
      alpha = DofELEM[i] + BasisVn1[j]   #Multiplication DofELEM[i]*BasisVn1[j] ==> addition of the exponents
      coeff = (Fraction(math.factorial(alpha[0])) * 
              Fraction(math.factorial(alpha[1])) * 
              Fraction(math.factorial(alpha[2])) * 
              Fraction(math.factorial(alpha[3]))) / Fraction(math.factorial(sum(alpha) + 3))
     
      M[index][j] +=  coeff 

    # Convert to numpy array with dtype=object for Fraction
  M = np.array(M, dtype=object)
  return M


############################### TEST OF UNISOLVENCE ###############################################


MatrixVn1=ComputeFEmatrix(BasisVn1)


print(f"The determinant of the matrix of the finite element V_{n+1} is: \n \n {MatrixVn1}")

#print(MatrixVn1)



determinantMatrixVn1= Matrix(MatrixVn1).det()


print(f"The determinant of the matrix of the finite element V_{n+1} is : {determinantMatrixVn1}")


