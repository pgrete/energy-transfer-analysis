import numpy as np
from mpi4py_fft import PFFT, newDistArray
from FFTHelperFuncs import get_local_wavenumbermesh
import time
import pickle
import sys
import h5py
import os
from scipy.stats import binned_statistic
from math import ceil
from MPIderivHelperFuncs import MPIderiv2, MPIXdotGradY, MPIdivX, MPIdivXY, MPIgradX, MPIrotX

# Aurora Cossairt's comments associated with command:
# srun -n 8 --mem-per-cpu=10G python ~/src/energy-transfer-analysis/run_analysis.py --res 256 --data_path /mnt/research/compastro-REU/cossairt_simulation/a-1.00/id0/Turb.0010.vtk  --data_type Athena --type flow --eos isothermal -forced --outfile /mnt/research/compastro-REU/cossairt_simulation/a-1.00/0010.hdf5

class FlowAnalysis:
    
    def __init__(self, MPI, args, fields):  # We're doing some parallel programming!
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.res = args['res']

        self.comm.Barrier()
        time_start = MPI.Wtime()

        if args['extrema_file'] is not None:
            try:
                self.global_min_max = pickle.load(open(args['extrema_file'], 'rb'))
                if self.rank == 0:
                    print("Successfully loaded global extrema dict: %s" % 
                          self.global_min_maxFile)
            except FileNotFoundError:
                raise SystemExit('Extrema file not found: ', args['extrema_file'])
        else:
            self.global_min_max = {}

        self.rho = fields['rho']  # These are coming from the fields argument, which we gathered from args in run_analysis.py
        self.U = fields['U']  # These variables are density, velocity, magnetic field, acceleration, pressure
        self.B = fields['B']  # equation of state, adiabatic gamma index, and the existence of magnetic fields 
        self.Acc = fields['Acc']
        self.P = fields['P']

        self.eos = args['eos']
        self.gamma = args['gamma']
        self.has_b_fields = args['b']
        
        self.kernel = args['kernel']  # Added
        
        if self.rank == 0:
            self.outfile_path = args['outfile']

        # maximum integer wavenumber
        k_max_int = ceil(self.res*0.5*np.sqrt(3.))   # How does this math work?
        self.k_bins = np.linspace(0.5,k_max_int + 0.5,k_max_int+1)  # Seems to imply how we split up the x axis according to wavenumbers (k's)?
        print("K bins shape: ", self.k_bins.shape)
        # if max k does not fall in last bin, remove last bin
        if self.res*0.5*np.sqrt(3.) < self.k_bins[-2]:
            self.k_bins = self.k_bins[:-1]

        N = np.array([self.res,self.res,self.res], dtype=int)
        # using L = 2pi as we work (e.g. when binning) with integer wavenumbers
            # I think we use 2*pi because it makes the math convenient when working with wavenumbers
        L = np.array([2*np.pi, 2*np.pi, 2*np.pi], dtype=float)
        self.FFT = PFFT(self.comm, N, axes=(0,1,2), collapse=False,
                        slab=True, dtype=np.complex128)

        self.localK = get_local_wavenumbermesh(self.FFT, L)  # Use FFT to define each k value along the x axis (?)
        self.localKmag = np.linalg.norm(self.localK,axis=0)  

        self.localKunit = np.copy(self.localK)  # Not sure what this means...
        # fixing division by 0 for harmonic part
            # What's the "harmonic part"?
        if self.rank == 0:
            if self.localKmag[0,0,0] == 0.:
                self.localKmag[0,0,0] = 1.
            else:
                raise SystemExit("[0] kmag not zero where it's supposed to be")
        else:
            if (self.localKmag == 0.).any():
                raise SystemExit("[%d] kmag not zero where it's supposed to be" % self.rank)

        self.localKunit /= self.localKmag
        if self.rank == 0:
            self.localKmag[0,0,0] = 0.
    
    def run_analysis(self):  # This looks like the actual running analysis part!
        
        rho = self.rho
        U = self.U
        B = self.B
        Acc = self.Acc
        P = self.P
        
        if self.rank == 0:
            self.outfile = h5py.File(self.outfile_path, "w")

        self.vector_power_spectrum('u',U) 
        self.vector_power_spectrum('rhoU',np.sqrt(rho)*U)
        self.vector_power_spectrum('rhoThirdU',rho**(1./3.)*U)
        
        vec_harm, vec_sol, vec_dil = self.decompose_vector(U)
        self.vector_power_spectrum('u_s',vec_sol)
        self.vector_power_spectrum('u_c',vec_dil)
        self.vector_power_spectrum('rhou_s',np.sqrt(rho)*vec_sol)
        self.vector_power_spectrum('rhou_c',np.sqrt(rho)*vec_dil)
        del vec_harm, vec_sol, vec_dil

        self.get_and_write_statistics_to_file(rho,"rho")  # Is the file 'rho'?
        self.get_and_write_statistics_to_file(np.log(rho),"lnrho")
        self.scalar_power_spectrum('rho',rho)   # Ok, so this one is a scalar because doesn't contain 'U'?
        self.scalar_power_spectrum('lnrho',np.log(rho))

        V2 = np.sum(U**2.,axis=0)   # That looks like Kinetic Energy (basically)
        self.get_and_write_statistics_to_file(np.sqrt(V2),"u")  # That looks like velocity

        self.get_and_write_statistics_to_file(0.5 * rho * V2,"KinEnDensity")  # KinEn # Does this do this for each rho?
        self.get_and_write_statistics_to_file(0.5 * V2,"KinEnSpecific")  # What is KE specific

        print("Finished KinEnSpecific")

        if Acc is not None:
            print("In Acc is not None")

            Amag = np.sqrt(np.sum(Acc**2.,axis=0))
            self.get_and_write_statistics_to_file(Amag,"a")
            self.vector_power_spectrum('a',Acc)

            corrUA = self.get_corr_coeff(np.sqrt(V2),Amag)
            if self.rank == 0:
                self.outfile.require_dataset('U-A/corr', (1,), dtype='f')[0] = corrUA
            
            UHarm, USol, UDil = self.decompose_vector(U)  # We called these when calculating mean spectra?
            self.get_and_write_statistics_to_file(
                np.sum(Acc*U,axis=0)/(  # Acceleration times u (propogation speed?) What does that mean?
                    np.linalg.norm(Acc,axis=0)*np.linalg.norm(U,axis=0)),"Angle_u_a")
            self.get_and_write_statistics_to_file(
                np.sum(Acc*USol,axis=0)/(
                    np.linalg.norm(Acc,axis=0)*np.linalg.norm(USol,axis=0)),"Angle_uSol_a")
            self.get_and_write_statistics_to_file(
                np.sum(Acc*UDil,axis=0)/(
                    np.linalg.norm(Acc,axis=0)*np.linalg.norm(UDil,axis=0)),"Angle_uDil_a")

        DivU = MPIdivX(self.comm,U)
        self.get_and_write_statistics_to_file(np.abs(DivU),"AbsDivU")
        self.get_and_write_statistics_to_file(np.sqrt(np.sum(MPIrotX(self.comm,U)**2.,axis=0)),"AbsRotU")

        print("Just wrote statistics for AbsRotU")

        if self.eos == 'adiabatic':
            self.gamma = self.gamma
            if self.rank == 0:
                print("Using gamma = %.3f for adiabatic EOS" % self.gamma)
            
            c_s2 = self.gamma * P / rho
            Ms2 = V2 / (c_s2)
            self.get_and_write_statistics_to_file(np.sqrt(Ms2),"Ms")
            self.get_2d_hist('rho-P',rho,P)
            self.get_and_write_statistics_to_file(P,"P")
            self.get_and_write_statistics_to_file(np.log10(P),"log10P")
            T = P / (self.gamma - 1.0) /rho
            self.get_and_write_statistics_to_file(T,"T")
            self.get_and_write_statistics_to_file(np.log10(T),"log10T")
            self.get_2d_hist('rho-T',rho,T)

            K = T/rho**(2./3.)
            self.get_2d_hist('T-K',T,K)
            self.get_2d_hist('rho-K',rho,K)

            self.co_spectrum('PD',P,DivU)

            self.scalar_power_spectrum('eint',np.sqrt(rho*c_s2))

        #########################
        # Method B: equation 33 #
        #########################
        
        print("In Method B")
        
        # Set up array of filter sizes
        delta_x = 1/self.res
        m = np.arange(1, self.res/2, 1)
        print("M is: ", m)
        k_lm = lm = np.zeros(int(self.res/2)-1)
        lm[:] = np.array(2*m[:]*delta_x)
        k_lm[:] = 1/lm[:]

        momentum = rho * U
        epsilon = np.zeros(len(k_lm)) # Used dtype = complex
        N = self.res * self.res * self.res
        
        for i, k in enumerate(k_lm):
            print("Beginning new loop with k: ", k)
            # Set up for calculating cumulative spectrum (epsilon) for each filter length scale
            dim = newDistArray(self.FFT,False,rank=1)  
            self.FT_momentum = newDistArray(self.FFT,rank=1)
            self.momentum_filtered = newDistArray(self.FFT,False,rank=1)
            self.FT_rho = newDistArray(self.FFT,rank=0)   
            self.rho_filtered = newDistArray(self.FFT,False,rank=0) 
            
            # Check ranks
            my_rank = self.comm.Get_rank()
            my_size = self.comm.Get_size()
            # print("Hello! I'm rank %d from %d running in total..." % (my_rank, my_size))

            # Momentum has to be put through a loop to account for each of its three components (x, y, and z). See line 141 for an example.
            for i in range(3):
                self.FT_momentum[i] = self.FFT.forward(momentum[i], self.FT_momentum[i])
        
            self.FT_rho = self.FFT.forward(rho, self.FT_rho)
            
            # Calculate the cumulative spectrum (epsilon) for each filter length scale
            filter = self.res/(2*k) # Shouldn't this be an integer?
            print("Filter is: ", filter, " on rank %d" % my_rank)
            FT_G = self.Kernel(filter)  
            self.rho_filtered = (self.FFT.backward(FT_G * self.FT_rho, self.rho_filtered)).real
            
            for j in range(3):    
                self.momentum_filtered[j] = (self.FFT.backward(FT_G * self.FT_momentum[j], self.momentum_filtered[j])).real
            
            for q in range(3):
                dim[q] = 0.5 * abs(self.momentum_filtered[q]**2) / abs(self.rho_filtered) 
                # creates an array of shape (256, 256, 256). This gets saved to one of the dimensions of dim.
                # so dim[k] contains all the energies in all cells associated with dimension k
 
            # Adds up the energies of all dimensions. Result is still an array of shape (256, 256, 256)
            all_dims = np.sum(dim, axis = 0) 
            local_sum = np.sum(all_dims)
            total = (self.comm.allreduce(local_sum)).real
            print("\nTotal for k ", k, " is: ", total, " on rank %d: " % my_rank)
            epsilon[i] = total / N
            print("Epsilon[", i, "] is: ", epsilon[i], " on rank %d: " % my_rank)

        # Calculate the energy spectrum for each filter length scale (use equation 33)
        print("Epsilon outside the for loop is: ", epsilon[:], " on rank %d " % my_rank)
        TotEnergy = np.zeros(len(k_lm)-1)
        
        # So, we start with TotEnergy[0] = epsilon[m=2] - epsilon[m=1], then do epsilon[m=3]-epsilon[m=2], and so on
        # Ending with TotEnergy[125] = epsilon[m=127] - epsilon[m=126]
        for i in range(len(k_lm) - 1):
            TotEnergy[i] = epsilon[i+1] - epsilon[i]
            #print("epsilon[i+1] is: ", epsilon[i+1])
            #print("epsilon[i] is: ", epsilon[i])
            #print("Total energy at position ", i, " is: ", TotEnergy[i])
            
        # Make final m_bins (same as m but there's no m = 1, because that would require an m = 0)
        k_lm_bins = np.zeros(len(k_lm)-1)
        k_lm_bins[:] = (k_lm[0:-1] - k_lm[1:]) / 2 + k_lm[1:]

        if self.rank == 0:
            print("Printing outfiles")
            self.outfile.require_dataset('TotEnergy/MethodB_PowSpec/Bins', (1,len(k_lm)-1), dtype='f')[0] = k_lm_bins
            self.outfile.require_dataset('TotEnergy/MethodB_PowSpec/Full', (1,len(k_lm)-1), dtype='f')[0] = TotEnergy

        print("Finished Method B", file=sys.stderr)

        ###################
        # End of Method B #
        ###################
        
        if not self.has_b_fields:  # We don't have b fields... is this where we close?
            print("In not self.has_b_fields")
            if self.rank == 0:
                self.outfile.close()
            return

        B2 = np.sum(B**2.,axis=0)  # Eyy it's some magnetic pressure! (or magnetic energy? idk...)

        self.get_and_write_statistics_to_file(np.sqrt(B2),"B")  # seems like there's a lot contained in these files
        self.get_and_write_statistics_to_file(0.5 * B2,"MagEnDensity") # There's magnetic energy density:)

        if self.eos == 'adiabatic':
            TotPres = P + B2/2.
            corrPBcomp = self.get_corr_coeff(P,np.sqrt(B2)/rho**(2./3.))
            self.get_2d_hist('P-B',P,np.sqrt(B2))
            self.get_2d_hist('P-MagEnDensity',P,0.5 * B2)
            
            plasmaBeta = 2.* P / B2
        elif self.eos == 'isothermal':  # That's us!
            if self.rank == 0:
                print('Warning: assuming c_s = 1 for isothermal EOS')
            TotPres = rho + B2/2.  # Why is this math true?
            corrPBcomp = self.get_corr_coeff(rho,np.sqrt(B2)/rho**(2./3.))
            
            plasmaBeta = 2.*rho/B2   # Ratio of gas pressure to magnetic pressure
        else:
            raise SystemExit('Unknown EOS', self.eos)

        self.get_and_write_statistics_to_file(TotPres,"TotPres")
        self.get_and_write_statistics_to_file(plasmaBeta,"plasmabeta")
        self.get_and_write_statistics_to_file(np.log10(plasmaBeta),"log10plasmabeta")

        if self.rank == 0:
            self.outfile.require_dataset('P-Bcomp/corr', (1,), dtype='f')[0] = corrPBcomp

        AlfMach2 = V2*rho/B2
        AlfMach = np.sqrt(AlfMach2)  # Ratio of characteristic velocity to isothermal speed of sound

        self.get_and_write_statistics_to_file(AlfMach,"AlfvenicMach")

        self.vector_power_spectrum('B',B)  # What does this vector_power_spectrum() actually do?

        # this is cheap... and only works for slab decomp on x-axis
        # np.sum is required for slabs with width > 1
        if rho.shape[-1] != self.res or rho.shape[-2] != self.res:   # What's this negative indexing??
            raise SystemExit('Calculation of dispersion measures only works for slabs')

        DM = self.comm.allreduce(np.sum(rho,axis=0))/float(self.res)  # What is DM and RM? Maybe domain?
        RM = self.comm.allreduce(np.sum(B[0]*rho,axis=0))/float(self.res)
        chunkSize = int(self.res/self.size)
        endIdx = int((self.rank + 1) * chunkSize)
        if endIdx == self.size:
            endIdx = None
        self.get_and_write_statistics_to_file(DM[self.rank*chunkSize:endIdx,:],"DM_x")
        self.get_and_write_statistics_to_file(np.log(DM[self.rank*chunkSize:endIdx,:]),"lnDM_x")
        self.get_and_write_statistics_to_file(RM[self.rank*chunkSize:endIdx,:],"RM_x")
        self.get_and_write_statistics_to_file(RM[self.rank*chunkSize:endIdx,:]/DM[self.rank*chunkSize:endIdx,:],"LOSB_x")

        DM = np.mean(rho,axis=1)
        RM = np.mean(B[1]*rho,axis=1)
        self.get_and_write_statistics_to_file(DM,"DM_y")
        self.get_and_write_statistics_to_file(np.log(DM),"lnDM_y")
        self.get_and_write_statistics_to_file(RM,"RM_y")
        self.get_and_write_statistics_to_file(RM/DM,"LOSB_y")

        DM = np.mean(rho,axis=2)
        RM = np.mean(B[2]*rho,axis=2)
        self.get_and_write_statistics_to_file(DM,"DM_z")
        self.get_and_write_statistics_to_file(np.log(DM),"lnDM_z")
        self.get_and_write_statistics_to_file(RM,"RM_z")
        self.get_and_write_statistics_to_file(RM/DM,"LOSB_z")

        corrRhoB = self.get_corr_coeff(rho,np.sqrt(B2))
        if self.rank == 0:
            self.outfile.require_dataset('rho-B/corr', (1,), dtype='f')[0] = corrRhoB

        if self.eos == 'adiabatic':
            rhoToGamma = rho**self.gamma
            corrRhoToGammaB = self.get_corr_coeff(rhoToGamma,np.sqrt(B2))
            if self.rank == 0:
                self.outfile.require_dataset('rhoToGamma-B/corr', (1,), dtype='f')[0] = corrRhoToGammaB

            self.get_2d_hist('rhoToGamma-B',rhoToGamma,np.sqrt(B2))
            self.get_2d_hist('rhoToGamma-MagEnDensity',rhoToGamma,0.5 * B2)

        self.get_2d_hist('rho-B',rho,np.sqrt(B2))
        self.get_2d_hist('log10rho-B',np.log10(rho),np.sqrt(B2))


        if self.rank == 0:
            self.outfile.close()

    def Kernel(self,DELTA,factor=None):
    # This function creates G^hat_l (the convolution kernel for a particular filtering scale)
        """generate filter kernels
        KERNEL - type of kernel to be used   # Need to read this in as an argument?
        RES    - linear numerical resolution    # Replaced with self.res
        DELTA  - filter width (in grid cells)   # This is 'l', the filtering scale (aka spatial cutoff length)
        factor - (optional) multiplicative factor of the filter width
                                            """                                        
        print("In Kernel()")

        k = self.localKmag # array of wavenumbers # NO! WRONG!
        KERNEL = self.kernel
        pi = np.pi
        if factor is None:
            factor = 1 
        else:
            factor = np.int(factor)
        
        if KERNEL == "KernelBox":   # Box or top hat filter
            localKern = np.zeros((self.res,self.res,self.res))  # Create a cube with dimensions of resolution (initialize with zeros)
            for i in range(-factor*np.int(DELTA)/2,factor*np.int(DELTA)/2+1):   # These are all the same
                for j in range(-factor*np.int(DELTA)/2,factor*np.int(DELTA)/2+1):    # Fill in the 3D box
                    for k in range(-factor*np.int(DELTA)/2,factor*np.int(DELTA)/2+1):
                        localFac = 1.   # Every localFac gets set to 1
                        # Changed to 0.5 if this condition is met (magnitude of step is 1/2 filter width times factor)
                        if np.abs(i) == factor*np.int(DELTA)/2: 
                            localFac *= 0.5     # What is this localFac?
                        if np.abs(j) == factor*np.int(DELTA)/2:
                            localFac *= 0.5                    
                        if np.abs(k) == factor*np.int(DELTA)/2:
                            localFac *= 0.5 
                    
            localKern[i,j,k] = localFac / float(factor*DELTA)**3.   # What is localKern?
            #print("FFT of localKern (top hat kernel) is: " fftn(localKern))
            return fftn(localKern)   # Appears to be using a Fast Fourier Transform? What is this `function fftn()`?
            # Returns the localKern in spectral space
            # Then what does it do with it?
            # Never deals with cutoff wave number... weird?
            
        elif KERNEL == "KernelSharp":
            localKern = np.ones_like(k)   # Set all kernels equal to 1 to start
            localKern[k > np.float(self.res)/(2. * factor * np.float(DELTA))] = 0.  # Change kernels to zero if k > k_c (getting rid of small scales)
            # Looks very different from eq. 2.44 and 2.45
            print("localKern (for KernelSharp) is: ", localKern)
            return localKern
            # Returns localKern in real space
        elif KERNEL == "KernelGauss":   # Gaussian filter
            print("DELTA shape is: ", DELTA.shape)
            # print("Gaussian kernel is: ", np.exp(-(pi * factor * DELTA/self.res * k)**2. /6.) )
            print("Shape of Gaussian kernel is: ", np.array(np.exp(-(pi * factor * DELTA/self.res * k)**2. /6.)).shape)
            return np.exp(-(pi * factor * DELTA/self.res * k)**2. /6.)  # Looks different from eq. 2.43
            
        else:
            sys.exit("Unknown kernel used")                                                                                                                                                                                                     
    def normalized_spectrum(self,k,quantity):
        """ Calculate normalized power spectra
        """
        histSum = binned_statistic(k,quantity,bins=self.k_bins,statistic='sum')[0]  # What is binned_statistic?
        kSum = binned_statistic(k,k,bins=self.k_bins,statistic='sum')[0]
        histCount = np.histogram(k,bins=self.k_bins)[0]

        totalHistSum = self.comm.reduce(histSum.astype(np.float64))
        totalKSum = self.comm.reduce(kSum.astype(np.float64))
        totalHistCount = self.comm.reduce(histCount.astype(np.float64))  # how many histograms to make?

        if self.rank == 0:

            if (totalHistCount == 0.).any():
                print("totalHistCount is 0. Check desired binning!")
                print(self.k_bins)
                print(totalHistCount)
                sys.exit(1)

            # calculate corresponding k to to bin
            # this help to overcome statistics for low k bins
            centeredK = totalKSum / totalHistCount


            # calculate corresponding k to to bin
            # this help to overcome statistics for low k bins
            centeredK = totalKSum / totalHistCount

            ###  "integrate" over k-shells
            # normalized by mean shell surface
            valsShell = 4. * np.pi * centeredK**2. * (totalHistSum / totalHistCount)
            # normalized by mean shell volume
            # this help to overcome statistics for low k bins
            centeredK = totalKSum / totalHistCount

            ###  "integrate" over k-shells
            # normalized by mean shell surface
            valsShell = 4. * np.pi * centeredK**2. * (totalHistSum / totalHistCount)
            # normalized by mean shell volume
            valsVol = 4. * np.pi / 3.* (self.k_bins[1:]**3. - self.k_bins[:-1]**3.) * (totalHistSum / totalHistCount)
            # unnormalized
            valsNoNorm = totalHistSum

            return [centeredK,valsShell,valsVol,valsNoNorm]
        else:
            return None

    def get_rotation_free_vec_field(self, vec):
        """
        returns the rotation free component of a 3D 3 component vector field
        based on 2nd order finite central differences by solving
        discrete La Place eqn div V = - div (grad phi)
        """

        # set up left side in Fourier space
        div_vec = MPIdivX(self.comm, vec)

        ft_div_vec = newDistArray(self.FFT, rank=0)
        ft_div_vec = self.FFT.forward(div_vec, ft_div_vec)

        # discrete fourier representation of -div grad based on consecutive
        # 2nd order first derivatives
        denom = -1/2. * self.res**2. * (np.cos(4.*np.pi*self.localK[0]/self.res) +
                                        np.cos(4.*np.pi*self.localK[1]/self.res) +
                                        np.cos(4.*np.pi*self.localK[2]/self.res) - 3.)

        # these are 0 in the nominator anyway, so set this to 1 to avoid
        # division by zero
        denom[denom == 0.] = 1.

        ft_div_vec /= denom
        phi = newDistArray(self.FFT, False, rank=0)
        phi = self.FFT.backward(ft_div_vec, phi).real

        return - MPIgradX(self.comm, phi)

    def decompose_vector(self, vec):
        """ decomposed input vector into harmonic, rotational and compressive part
        """
        
        N = float(self.comm.allreduce(vec.size))
        total = self.comm.allreduce(np.sum(vec,axis=(1,2,3)))
        # dividing by N/3 as the FFTs are per dimension, i.e., normal is N^3 but N is 3N^3
        harm = total / (N/3)

        dil = self.get_rotation_free_vec_field(vec) # Here is harm dil and sol!
        sol = vec - harm.reshape((3,1,1,1)) - dil

        return harm, sol, dil


    def scalar_power_spectrum(self,name,field):  # Oh hey! This is one of the things!

        FT_field = newDistArray(self.FFT)
        FT_field = self.FFT.forward(field, FT_field)

        FT_fieldAbs2 = np.abs(FT_field)**2.
        PS_Full = self.normalized_spectrum(self.localKmag.reshape(-1),FT_fieldAbs2.reshape(-1))

        if self.rank == 0: # This appears to be where we're writing files
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full  # What does this function return?

# e.g. (38) in https://arxiv.org/pdf/1101.0150.pdf
    def co_spectrum(self,name,fieldA,fieldB):

        FT_fieldA = newDistArray(self.FFT)
        FT_fieldA = self.FFT.forward(fieldA, FT_fieldA)

        FT_fieldB = newDistArray(self.FFT)
        FT_fieldB = self.FFT.forward(fieldB, FT_fieldB)

        FT_CoSpec = FT_fieldA * np.conj(FT_fieldB)
        PS_Abs = self.normalized_spectrum(self.localKmag.reshape(-1),np.abs(FT_CoSpec).reshape(-1))
        PS_Real = self.normalized_spectrum(self.localKmag.reshape(-1),np.real(FT_CoSpec).reshape(-1))

        if self.rank == 0:
            self.outfile.require_dataset(name + '/CoSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/CoSpec/Abs', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Abs
            self.outfile.require_dataset(name + '/CoSpec/Real', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Real

    def vector_power_spectrum(self, name, vec):  # Eyyy there it is
        FT_vec = newDistArray(self.FFT,rank=1)
        for i in range(3):
            FT_vec[i] = self.FFT.forward(vec[i], FT_vec[i])  # Does exactly the same thing as scalar, but for all components of an array

        FT_vecAbs2 = np.linalg.norm(FT_vec,axis=0)**2.
        PS_Full = self.normalized_spectrum(self.localKmag.reshape(-1),FT_vecAbs2.reshape(-1))
        
        totPowFull = self.comm.allreduce(np.sum(FT_vecAbs2))
        
        if self.rank == 0:
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full
            self.outfile.require_dataset(name + '/PowSpec/TotFull', (1,), dtype='f')[0] = totPowFull

        # project components
        localVecDotKunit = np.sum(FT_vec*self.localKunit,axis = 0)

        FT_Dil = localVecDotKunit * self.localKunit  # Still have no idea what "Dil" and "Sol" mean
        FT_DilAbs2 = np.linalg.norm(FT_Dil,axis=0)**2.
        PS_Dil = self.normalized_spectrum(self.localKmag.reshape(-1),FT_DilAbs2.reshape(-1))
        
        FT_Sol = FT_vec - FT_Dil
        if self.rank == 0:
            # remove harmonic part from solenoidal component
            FT_Sol[:,0,0,0] = 0.
        FT_SolAbs2 = np.linalg.norm(FT_Sol,axis=0)**2.
        PS_Sol = self.normalized_spectrum(self.localKmag.reshape(-1),FT_SolAbs2.reshape(-1))



        totPowDil = self.comm.allreduce(np.sum(FT_DilAbs2))
        totPowSol = self.comm.allreduce(np.sum(FT_SolAbs2))
        totPowHarm = np.linalg.norm(FT_vec[:,0,0,0],axis=0)**2.

        if self.rank == 0:  # Is this where it actually makes the hd5f file?
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full
            self.outfile.require_dataset(name + '/PowSpec/Dil', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Dil
            self.outfile.require_dataset(name + '/PowSpec/Sol', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Sol
            self.outfile.require_dataset(name + '/PowSpec/TotSol', (1,), dtype='f')[0] = totPowSol
            self.outfile.require_dataset(name + '/PowSpec/TotDil', (1,), dtype='f')[0] = totPowDil
            self.outfile.require_dataset(name + '/PowSpec/TotHarm', (1,), dtype='f')[0] = totPowHarm



    def get_and_write_statistics_to_file(self,field,name,bounds=None):  # Aha! Explains field
        """
            field - 3d scalar field to get statistics from  
            name - human readable name of the field
            bounds - tuple, lower and upper bound for histogram, if None then min/max
        """
        
        N = float(self.comm.allreduce(field.size))  # some nice statistical outputs
        total = self.comm.allreduce(np.sum(field))
        mean = total / N

        totalSqrd = self.comm.allreduce(np.sum(field**2.))
        rms = np.sqrt(totalSqrd / N)

        var = self.comm.allreduce(np.sum((field - mean)**2.)) / (N - 1.)
        
        stddev = np.sqrt(var)

        skew = self.comm.allreduce(np.sum((field - mean)**3. / stddev**3.)) / N
        
        kurt = self.comm.allreduce(np.sum((field - mean)**4. / stddev**4.)) / N - 3.

        globMin = self.comm.allreduce(np.min(field),op=self.MPI.MIN)
        globMax = self.comm.allreduce(np.max(field),op=self.MPI.MAX)

        globAbsMin = self.comm.allreduce(np.min(np.abs(field)),op=self.MPI.MIN)
        globAbsMax = self.comm.allreduce(np.max(np.abs(field)),op=self.MPI.MAX)

        if self.rank == 0:  # First node (santa) saves all the means, rms's, other statistical children
            self.outfile.require_dataset(name + '/moments/mean', (1,), dtype='f')[0] = mean
            self.outfile.require_dataset(name + '/moments/rms', (1,), dtype='f')[0] = rms
            self.outfile.require_dataset(name + '/moments/var', (1,), dtype='f')[0] = var
            self.outfile.require_dataset(name + '/moments/stddev', (1,), dtype='f')[0] = stddev
            self.outfile.require_dataset(name + '/moments/skew', (1,), dtype='f')[0] = skew
            self.outfile.require_dataset(name + '/moments/kurt', (1,), dtype='f')[0] = kurt
            self.outfile.require_dataset(name + '/moments/min', (1,), dtype='f')[0] = globMin
            self.outfile.require_dataset(name + '/moments/max', (1,), dtype='f')[0] = globMax
            self.outfile.require_dataset(name + '/moments/absmin', (1,), dtype='f')[0] = globAbsMin
            self.outfile.require_dataset(name + '/moments/absmax', (1,), dtype='f')[0] = globAbsMax

        if bounds is None:
            bounds = [globMin,globMax]
            HistBins = "Snap"
        else:
            HistBins = "Sim"

        Bins = np.linspace(bounds[0],bounds[1],129)
        hist = np.histogram(field.reshape(-1),bins=Bins)[0]  # Histograms?
        totalHist = self.comm.allreduce(hist)

        if self.rank == 0:
            tmp  = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,129), dtype='f')
            tmp[0] = Bins
            tmp[1,:-1] = totalHist.astype(float)

        if name in self.global_min_max.keys():
            Bins = np.linspace(self.global_min_max[name][0],self.global_min_max[name][1],129)
            hist = np.histogram(field.reshape(-1),bins=Bins)[0]
            totalHist = self.comm.allreduce(hist)

            HistBins = 'globalMinMax'

            if self.rank == 0:
                tmp  = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,129), dtype='f')
                tmp[0] = Bins
                tmp[1,:-1] = totalHist.astype(float)


    def get_corr_coeff(self,X,Y):
        N = float(self.comm.allreduce(X.size))
        meanX = self.comm.allreduce(np.sum(X)) / N
        meanY = self.comm.allreduce(np.sum(Y)) / N

        stdX = np.sqrt(self.comm.allreduce(np.sum((X - meanX)**2.)))
        stdY = np.sqrt(self.comm.allreduce(np.sum((Y - meanY)**2.)))

        cov = self.comm.allreduce(np.sum( (X - meanX)*(Y - meanY)))

        return cov / (stdX*stdY)

    def get_2d_hist(self,name,X,Y,bounds = None):  # Getting a 2D hist?
        if bounds is None:
            globXMin = self.comm.allreduce(np.min(X),op=self.MPI.MIN)
            globXMax = self.comm.allreduce(np.max(X),op=self.MPI.MAX)
            
            globYMin = self.comm.allreduce(np.min(Y),op=self.MPI.MIN)
            globYMax = self.comm.allreduce(np.max(Y),op=self.MPI.MAX)

            bounds = [[globXMin,globXMax],[globYMin,globYMax]]
            HistBins = "Snap"
        else:
            HistBins = "Sim"
        
        
        XBins = np.linspace(bounds[0][0],bounds[0][1],129)  # Why 129?
        YBins = np.linspace(bounds[1][0],bounds[1][1],129)  # Why upside down?

        hist = np.histogram2d(X.reshape(-1),Y.reshape(-1),bins=[XBins,YBins])[0]
        totalHist = self.comm.allreduce(hist)

        if self.rank == 0:
            tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,129), dtype='f')
            tmp[0] = XBins
            HistBins = 'globalMinMax'

            if self.rank == 0:
                tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,129), dtype='f')
                tmp[0] = XBins
                tmp[1] = YBins
                tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/counts', (128,128), dtype='f')
                tmp[:,:] = totalHist.astype(float)

    def run_test(self):
        """ simple tests (to be expanded and separated)
        """
        # using velocity field for no reason, but it's probably most likely to be avail
        vec = self.U

        # TESTING VECTOR DECOMPOSITION
        vec_harm, vec_sol, vec_dil = self.decompose_vector(vec)
        np.testing.assert_allclose(vec,
                                   vec_harm.reshape((3,1,1,1)) + vec_sol + vec_dil,
                                   err_msg="Mismatch bw/ decomposed and original vector.")

        msg = "solenoidal part is not divergence free"
        assert np.sum(np.abs(MPIdivX(self.comm, vec_sol)))/vec_sol.size/3 < 1e-13, msg
        
        msg = "compressive part is not rotation free"
        assert np.sum(np.linalg.norm(MPIrotX(self.comm, vec_dil),axis=0))/vec_dil.size/3 < 1e-13, msg


        
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 ai

