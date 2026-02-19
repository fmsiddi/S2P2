import numpy as np
from numpy.linalg import eigh
import torch

def DPLR_HiPPO(P):
    
    # HiPPO
    P_arr = np.arange(P)
    M = np.sqrt(2*P_arr + 1) # creates a P-dimensional array according to HiPPO-LegS
    A = -np.tril(np.outer(M,M)) + np.diag(P_arr) # A_nk in Appendix C.1 of S4 paper
    
    # DPLR
    Q = np.sqrt(P_arr + 0.5) # low-rank array (i.e. vector) part of the "low rank correction". See "adding..." in C.1 of S4 paper
    VΛV = A + np.outer(Q,Q)
    Λ_real = np.diagonal(VΛV) # this is just a P-dimensional array of -0.5's
    Λ_imag, V = eigh(VΛV * -1j) # the purely diagonal part will be ignored by eigh, leaving a Hermitian matrix which shares the same eigenvectors as VΛV*
    Q_tilde = V.conj() @ Q
    
    B = np.sqrt(2 * P_arr + 1.0)
    B_tilde = V.conj() @ B   
    
    return (
        # note the imaginary eigenvalues are being multiplied by i to account for the -i scaling above
        # recall that \mu = -i\lambda, so to recover \lambda we just multiply by i
        torch.tensor(np.asarray(Λ_real + 1j * Λ_imag), dtype=torch.complex64),
        torch.tensor(np.asarray(Q_tilde)),
        torch.tensor(np.asarray(B_tilde)),
        torch.tensor(np.asarray(V), dtype=torch.complex64),
        torch.tensor(np.asarray(B)),
    ) 
    