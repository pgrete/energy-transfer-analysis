import numpy as np 

def getHelicalDecomposition(FTquant, kx, ky, kz):
        """ perform helicity decomposition of solenoidal vector field in Fourier space FTquant
            returns positive and negative helicity components B_plus, B_minus,
            such that FTquant = B_plus + B_minus.  
        """

        if FTquant.shape[0] != 3:
            raise SystemExit("Helicity decomposition only implemented for 3D vector fields")
    
        # construct helical basis:
        
        # find vector orthogonal to k:
        # compute perpendicular norm safely
        kmag_perp = np.sqrt(kx**2 + ky**2)
        kmag_perp_safe = np.where(kmag_perp==0, 1.0, kmag_perp)

        # e1 in xy-plane
        e1 = np.array([-ky/kmag_perp_safe, kx/kmag_perp_safe, np.zeros_like(kx)])

        # handle exact k along z
        mask = (kmag_perp == 0)
        e1[:, mask] = np.array([[1.0],[0.0],[0.0]])  # broadcasting works safely

        # second unit vector e2 = k x e1:
        e2 = np.cross(np.stack([kx, ky, kz], axis=0).T, e1.T).T

        # normalize e2
        e2_norm = np.sqrt(np.sum(e2*e2, axis=0))
        e2_norm_safe = np.where(e2_norm==0, 1.0, e2_norm)
        e2 = e2 / e2_norm_safe

        # move to helical basis: 
        h_plus  = (e1 + 1j*e2)/np.sqrt(2)
        h_minus = (e1 - 1j*e2)/np.sqrt(2)

        # decompose FTB into helical modes:
        B_plus  = np.sum(np.conj(h_plus)  * FTquant, axis=0) * h_plus
        B_minus = np.sum(np.conj(h_minus)* FTquant, axis=0) * h_minus

        return B_plus, B_minus 