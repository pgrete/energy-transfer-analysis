import numpy as np
import FFTHelperFuncs
from mpi4py_fft import newDistArray
import time
import pickle
import sys
import h5py
import os
from scipy.stats import binned_statistic
from math import ceil
from MPIderivHelperFuncs import MPIderiv2, MPIXdotGradY, MPIdivX, MPIdivXY, MPIgradX, MPIrotX

class FlowAnalysis:
    
    def __init__(self, MPI, args, fields):
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
                          args['extrema_file'])
            except FileNotFoundError:
                raise SystemExit('Extrema file not found: ', args['extrema_file'])
        else:
            self.global_min_max = {}

        self.num_bins = args['num_bins']

        self.rho = fields['rho']
        self.U = fields['U']
        self.B = fields['B']
        self.Acc = fields['Acc']
        self.P = fields['P']
        self.eos = args['eos']
        self.gamma = args['gamma']
        self.cs = args['cs']
        self.has_b_fields = args['b']
        self.kernels= args['kernels']

        if self.rank == 0:
            self.outfile_path = args['outfile']

        # maximum integer wavenumber
        k_max_int = ceil(self.res*0.5*np.sqrt(3.))
        self.k_bins = np.linspace(0.5,k_max_int + 0.5,k_max_int+1)
#        # if max k does not fall in last bin, remove last bin
        if self.res*0.5*np.sqrt(3.) < self.k_bins[-2]:
            self.k_bins = self.k_bins[:-1]

#        resolution_exp = np.log(self.res/8)/np.log(2) * 4 + 1
#        self.k_bins = np.concatenate(
#            (np.array([0.]), 4.* 2** ((np.arange(0,resolution_exp + 1) - 1.) /4.)))

        self.FFT = FFTHelperFuncs.FFT
        self.localK = FFTHelperFuncs.local_wavenumbermesh
        self.localKmag = np.linalg.norm(self.localK,axis=0)

        self.localKunit = np.copy(self.localK)
        # fixing division by 0 for harmonic part
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

    def calculate_filtering_spectrum(self, kernel):
        """
        following Sadek and Aluie (2018) for calculating kinetic energy density filtering spectrum spectrum of kinetic energy
        kernel - one of the following kernel types: Box, Gauss, Sharp
        """
        # set up array of filters widths
        delta_x = 1/self.res
        m = np.arange(self.res/2 - 1, 0, -1) # array of integer number of grid cells (see Sadek and Aluie eqn. 31)
        k_lm = lm = np.zeros(int(self.res/2)-1)
        lm = np.array(2*m*delta_x) # array of length scales (see Sadek and Aluie eqn. 31)
        k_lm = 1/lm # array of wavenumbers (see Sadek and Aluie eqn. 32)

        momentum = self.rho * self.U
        epsilon = np.zeros(len(k_lm))
        N = self.res * self.res * self.res
        
        for i, k in enumerate(k_lm):
            # set up for calculating cumulative spectrum (epsilon) for each filter length scale
            energy = newDistArray(self.FFT,False,rank=1)
            FT_momentum = newDistArray(self.FFT,rank=1)
            momentum_filtered = newDistArray(self.FFT,False,rank=1)
            FT_rho = newDistArray(self.FFT,rank=0)
            rho_filtered = newDistArray(self.FFT,False,rank=0)

            # calculate the cumulative spectrum (epsilon) for each filter length scale
            # (see equations 27 and 6 of Sadek and Aluie)
            FT_rho = self.FFT.forward(self.rho, FT_rho)
            for j in range(3):
                FT_momentum[j] = self.FFT.forward(momentum[j], FT_momentum[j])

            filter_width = self.res/(2*k)
            FT_G = self.Kernel(filter_width, kernel)  # calculate convolution kernel
            # calculate filtered values of momentum and density
            rho_filtered = (self.FFT.backward(FT_G * FT_rho, rho_filtered)).real
            for j in range(3):
                momentum_filtered[j] = (self.FFT.backward(FT_G * FT_momentum[j], momentum_filtered[j])).real

            # compute the average of momentum^2 / density
            filtered_kin_energy = 0.5 * np.sum(np.abs(momentum_filtered**2) / np.abs(rho_filtered), axis=0)
            total_kin_energy = (self.comm.allreduce(np.sum(filtered_kin_energy)))
            # average energy
            epsilon[i] = total_kin_energy / N

        # calculate the energy spectrum for each filter length scale (see equation 33 of Sadek and Aluie)
        KinEnergy = np.zeros(len(k_lm)-1)
        for i in range(len(k_lm) - 1):
            KinEnergy[i] = (epsilon[i+1] - epsilon[i])/(k_lm[i+1] - k_lm[i])

        if self.rank == 0:
            self.outfile.require_dataset('Filtered_' + kernel + '_KinEnergy/PowSpec/Bins', (1,len(k_lm)-1), dtype='f')[0] = k_lm[1:]
            self.outfile.require_dataset('Filtered_' + kernel + '_KinEnergy/PowSpec/Full', (1,len(k_lm)-1), dtype='f')[0] = KinEnergy
            self.outfile.require_dataset('Filtered_' + kernel + '_KinEnergy/PowSpec/low_pass_energies', (1,len(k_lm)), dtype='f')[0] = epsilon

    def run_analysis(self):

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

        self.get_and_write_statistics_to_file(rho,"rho")
        self.get_and_write_statistics_to_file(np.log(rho),"lnrho")
        self.scalar_power_spectrum('rho',rho)
        self.scalar_power_spectrum('lnrho',np.log(rho))

        V2 = np.sum(U**2.,axis=0)
        self.get_and_write_statistics_to_file(np.sqrt(V2),"u")
        self.get_and_write_statistics_to_file(U[0],"u_x")
        self.get_and_write_statistics_to_file(U[1],"u_y")
        self.get_and_write_statistics_to_file(U[2],"u_z")

        self.get_and_write_statistics_to_file(0.5 * rho * V2,"KinEnDensity")
        self.get_and_write_statistics_to_file(0.5 * V2,"KinEnSpecific")

        if Acc is not None:
            Amag = np.sqrt(np.sum(Acc**2.,axis=0))
            self.get_and_write_statistics_to_file(Amag,"a")
            self.vector_power_spectrum('a',Acc)
            self.scalar_power_spectrum('a_x',Acc[0,:,:,:])
            self.scalar_power_spectrum('a_y',Acc[1,:,:,:])
            self.scalar_power_spectrum('a_z',Acc[2,:,:,:])

            vec_harm, vec_sol, vec_dil = self.decompose_vector(Acc)
            self.vector_power_spectrum('a_s',vec_sol)
            self.vector_power_spectrum('a_c',vec_dil)
            self.get_and_write_statistics_to_file(np.sqrt(np.sum(vec_sol**2.,axis=0)),"a_s_mag")
            self.get_and_write_statistics_to_file(np.sqrt(np.sum(vec_dil**2.,axis=0)),"a_c_mag")
            del vec_harm, vec_sol, vec_dil

            corrUA = self.get_corr_coeff(np.sqrt(V2),Amag)
            if self.rank == 0:
                self.outfile.require_dataset('U-A/corr', (1,), dtype='f')[0] = corrUA

            UHarm, USol, UDil = self.decompose_vector(U)
            self.get_and_write_statistics_to_file(
                np.sum(Acc*U,axis=0)/(
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

        if self.eos == 'adiabatic':
            self.gamma = self.gamma
            if self.rank == 0:
                print("Using gamma = %.3f for adiabatic EOS" % self.gamma)
            
            c_s2 = self.gamma * P / rho

            T = P / (self.gamma - 1.0) /rho
            self.get_and_write_statistics_to_file(T,"T")
            self.get_and_write_statistics_to_file(np.log10(T),"log10T")
            self.get_2d_hist('rho-T',rho,T)

            K = T/rho**(2./3.)
            self.get_2d_hist('T-K',T,K)
            self.get_2d_hist('rho-K',rho,K)
        elif self.eos == 'isothermal':
            if self.rank == 0:
                print("Using c_s = %.3f for isothermal EOS" % self.cs)
            c_s2 = self.cs**2.
        else:
            raise SystemExit('Unknown EOS', self.eos)

        Ms2 = V2 / (c_s2)
        self.get_and_write_statistics_to_file(np.sqrt(Ms2),"Ms")
        self.get_2d_hist('rho-P',rho,P)
        self.get_and_write_statistics_to_file(P,"P")
        self.get_and_write_statistics_to_file(np.log10(P),"log10P")

        self.co_spectrum('PD',P,DivU)

        # should include factor of 1/(gamma(gamma-1))
        self.scalar_power_spectrum('eint',np.sqrt(rho*c_s2))
        self.get_and_write_statistics_to_file(rho*c_s2,"IntEnDensity")

        if self.kernels is not None:
            if self.rank == 0:
                print("Using the following kernels for the filtering spectrum: ", self.kernels)
            for kernel in self.kernels:
                if self.rank == 0:
                    print("on kernel: ", kernel)
                self.calculate_filtering_spectrum(kernel)
        
        # we're done with the hydro analysis. If there are no mag fields, quit here.
        if not self.has_b_fields:
            if self.rank == 0:
                self.outfile.close()
            return

        B2 = np.sum(B**2.,axis=0)
        self.get_and_write_statistics_to_file(np.sqrt(B2),"B")
        self.get_and_write_statistics_to_file(0.5 * B2,"MagEnDensity")

        TotPres = P + B2/2.
        corrPBcomp = self.get_corr_coeff(P,np.sqrt(B2)/rho**(2./3.))
        self.get_2d_hist('P-B',P,np.sqrt(B2))
        self.get_2d_hist('P-MagEnDensity',P,0.5 * B2)
        corrPB = self.get_corr_coeff(P,np.sqrt(B2))
        corrPB2 = self.get_corr_coeff(P,B2)
            
        plasmaBeta = 2.* P / B2

        self.get_and_write_statistics_to_file(TotPres,"TotPres")
        self.get_and_write_statistics_to_file(plasmaBeta,"plasmabeta")
        self.get_and_write_statistics_to_file(np.log10(plasmaBeta),"log10plasmabeta")

        if self.rank == 0:
            self.outfile.require_dataset('P-Bcomp/corr', (1,), dtype='f')[0] = corrPBcomp
            self.outfile.require_dataset('P-B/corr', (1,), dtype='f')[0] = corrPB
            self.outfile.require_dataset('P-B2/corr', (1,), dtype='f')[0] = corrPB2

        AlfMach2 = V2*rho/B2
        AlfMach = np.sqrt(AlfMach2)

        self.get_and_write_statistics_to_file(AlfMach,"AlfvenicMach")

        self.vector_power_spectrum('B',B)

        # get the integral length scale of B (TODO: technically there's a rho in here)
        mag_en_spec = self.outfile['B/PowSpec/Full']
        L_B = np.trapz(mag_en_spec[1]/mag_en_spec[0],x=mag_en_spec[0])/np.trapz(mag_en_spec[1],x=mag_en_spec[0])
        if self.rank == 0:
            print("Integral lengthscale of B is %.3f" % L_B)

        # calculate the large scale B field
        B_large = newDistArray(self.FFT,False,rank=1)
        FT_B = newDistArray(self.FFT,rank=1)
        for j in range(3):
            FT_B[j] = self.FFT.forward(B[j],FT_B[j])
        FT_G = self.Kernel(L_B*self.res, "Gauss")
        for j in range(3):
            B_large[j] = (self.FFT.backward(FT_G * FT_B[j], B_large[j])).real

        del FT_G, FT_B

        self.vector_power_spectrum('B_large',B_large)

        # now get the components along and perp to B_large (TODO: substract mean. not important for spec here)
        B_large /= np.linalg.norm(B_large,axis=0) # this is now a unit vector
        B_par = B * B_large
        B_perp = B - B_par

        self.vector_power_spectrum('B_par',B_par)
        self.vector_power_spectrum('B_perp',B_perp)
        del B_par, B_perp

        U_par = U * B_large
        U_perp = U - U_par

        self.vector_power_spectrum('U_par',U_par)
        self.vector_power_spectrum('U_perp',U_perp)
        self.vector_power_spectrum('rhoU_par',np.sqrt(rho)*U_par)
        self.vector_power_spectrum('rhoU_perp',np.sqrt(rho)*U_perp)
        del U_par, U_perp

        self.get_and_write_statistics_to_file(B[0],"B_x")
        self.get_and_write_statistics_to_file(B[1],"B_y")
        self.get_and_write_statistics_to_file(B[2],"B_z")

        self.vector_power_spectrum('z_p',U + B/np.sqrt(rho))
        self.vector_power_spectrum('z_m',U - B/np.sqrt(rho))
        self.vector_power_spectrum('z_p_dens',np.sqrt(rho)*U + B)
        self.vector_power_spectrum('z_m_dens',np.sqrt(rho)*U - B)

        self.get_and_write_statistics_to_file(np.sum(B*U,axis=0),"cross_helicity")

        # this is cheap... and only works for pencil decomp in z axis
        # np.sum is required for slabs with width > 1
        if rho.shape[-1] != self.res:
            raise SystemExit('Calculation of dispersion measures only works for pencils')

        # using subcomms here so that the pencil based slices are not getting mixed
        DM = FFTHelperFuncs.FFT.subcomm[0].allreduce(np.sum(rho,axis=0))/float(self.res)
        RM = FFTHelperFuncs.FFT.subcomm[0].allreduce(np.sum(B[0]*rho,axis=0))/float(self.res)
        # using chunks so that each process only contributes it's own chunk of data
        # as all processes have the full information after the allreduce
        chunkSize = DM.shape[0] // FFTHelperFuncs.FFT.subcomm[0].Get_size()
        startIdx = FFTHelperFuncs.FFT.subcomm[0].Get_rank() * chunkSize
        endIdx = (FFTHelperFuncs.FFT.subcomm[0].Get_rank() + 1) * chunkSize
        self.get_and_write_statistics_to_file(DM[startIdx:endIdx,:],"DM_x")
        self.get_and_write_statistics_to_file(np.log(DM[startIdx:endIdx,:]),"lnDM_x")
        self.get_and_write_statistics_to_file(RM[startIdx:endIdx,:],"RM_x")
        self.get_and_write_statistics_to_file(RM[startIdx:endIdx,:]/DM[startIdx:endIdx,:],"LOSB_x")

        # using subcomms here so that the pencil based slices are not getting mixed
        DM = FFTHelperFuncs.FFT.subcomm[1].allreduce(np.sum(rho,axis=1))/float(self.res)
        RM = FFTHelperFuncs.FFT.subcomm[1].allreduce(np.sum(B[1]*rho,axis=1))/float(self.res)
        # using chunks so that each process only contributes it's own chunk of data
        # as all processes have the full information after the allreduce
        chunkSize = DM.shape[1] // FFTHelperFuncs.FFT.subcomm[1].Get_size()
        startIdx = FFTHelperFuncs.FFT.subcomm[1].Get_rank() * chunkSize
        endIdx = (FFTHelperFuncs.FFT.subcomm[1].Get_rank() + 1) * chunkSize
        self.get_and_write_statistics_to_file(DM[:,startIdx:endIdx],"DM_y")
        self.get_and_write_statistics_to_file(np.log(DM[:,startIdx:endIdx]),"lnDM_y")
        self.get_and_write_statistics_to_file(RM[:,startIdx:endIdx],"RM_y")
        self.get_and_write_statistics_to_file(RM[:,startIdx:endIdx]/DM[:,startIdx:endIdx],"LOSB_y")

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
            corrPRhoToGamma = self.get_corr_coeff(rhoToGamma,P)
            corrTotPresRhoToGamma = self.get_corr_coeff(rhoToGamma,TotPres)
            if self.rank == 0:
                self.outfile.require_dataset('rhoToGamma-B/corr', (1,), dtype='f')[0] = corrRhoToGammaB
                self.outfile.require_dataset('rhoToGamma-P/corr', (1,), dtype='f')[0] = corrPRhoToGamma
                self.outfile.require_dataset('rhoToGamma-TotPres/corr', (1,), dtype='f')[0] = corrTotPresRhoToGamma

            self.get_2d_hist('rhoToGamma-B',rhoToGamma,np.sqrt(B2))
            self.get_2d_hist('rhoToGamma-MagEnDensity',rhoToGamma,0.5 * B2)

        corrPRho = self.get_corr_coeff(rho,P)
        corrTotPresRho = self.get_corr_coeff(rho,TotPres)
        if self.rank == 0:
            self.outfile.require_dataset('rho-P/corr', (1,), dtype='f')[0] = corrPRho
            self.outfile.require_dataset('rho-TotPres/corr', (1,), dtype='f')[0] = corrTotPresRho

        self.get_2d_hist('rho-B',rho,np.sqrt(B2))
        self.get_2d_hist('log10rho-B',np.log10(rho),np.sqrt(B2))

        if self.rank == 0:
            self.outfile.close()

    def Kernel(self,DELTA,KERNEL,factor=None):
        """
        This function creates the convolution kernel for use with a particular filtering scale
        (see equation 6 of "Extracting the spectrum of a flow by spatial filtering" by Sadek and Aluie (2018))
        For an introduction to scale filtering, see chapter 2 of "Large Eddy Simulation for Incompressible Flows" by Pierre Sagaut.
        The three classical filters used here are described in section 1.5 of that chapter.
        KERNEL - the chosen type of convolution kernel (Box, Sharp, or Gaussian)
        DELTA  - filter width (in grid cells): filter_width = self.res/(2k) where k is the wavenumber associated with the filter.
        factor - (optional) multiplicative factor of the filter width
        """
        k = self.localKmag
        pi = np.pi

        if factor is None:
            factor = 1
        else:
            factor = np.int(factor)

        # NOTE: Box Kernel needs to be updated for parallel computing (currently non-functional)
        if KERNEL == "Box":   # aka top hat filter
            sys.exit("Box kernel is currently non-functional")
            localKern = np.zeros((self.res,self.res,self.res))
            for i in range(-factor*np.int(DELTA)//2,factor*np.int(DELTA)//2+1):
                for j in range(-factor*np.int(DELTA)//2,factor*np.int(DELTA)//2+1):
                    for k in range(-factor*np.int(DELTA)//2,factor*np.int(DELTA)//2+1):
                        localFac = 1.
                        if np.abs(i) == factor*np.int(DELTA)//2:
                            localFac *= 0.5
                        if np.abs(j) == factor*np.int(DELTA)//2:
                            localFac *= 0.5
                        if np.abs(k) == factor*np.int(DELTA)//2:
                            localFac *= 0.5 
            localKern[i,j,k] = localFac / float(factor*np.int(DELTA))**3.
            return fftn(localKern)
        elif KERNEL == "Sharp":
            localKern = np.ones_like(k)
            localKern[k > np.float(self.res)/(2. * factor * np.float(DELTA))] = 0. # Remove small scales
            return localKern
        elif KERNEL == "Gauss":
            return np.exp(-(pi * factor * DELTA/self.res * k)**2. /6.)
        else:
            sys.exit("Unknown kernel used")

    def normalized_spectrum(self,k,quantity):
        """ Calculate normalized power spectra
        """
        histSum = binned_statistic(k,quantity,bins=self.k_bins,statistic='sum')[0]
        kSum = binned_statistic(k,k,bins=self.k_bins,statistic='sum')[0]
        histCount = np.histogram(k,bins=self.k_bins)[0]

        totalHistSum = self.comm.reduce(histSum.astype(np.float64))
        totalKSum = self.comm.reduce(kSum.astype(np.float64))
        totalHistCount = self.comm.reduce(histCount.astype(np.float64))

        if self.rank == 0:

            if (totalHistCount == 0.).any():
                print("totalHistCount is 0. Check desired binning!")
                print(self.k_bins)
                print(totalHistCount)
                sys.exit(1)

            # calculate corresponding k to to bin
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

        dil = self.get_rotation_free_vec_field(vec)
        sol = vec - harm.reshape((3,1,1,1)) - dil

        return harm, sol, dil


    def scalar_power_spectrum(self,name,field):

        FT_field = newDistArray(self.FFT)
        FT_field = self.FFT.forward(field, FT_field)

        FT_fieldAbs2 = np.abs(FT_field)**2.
        PS_Full = self.normalized_spectrum(self.localKmag.reshape(-1),FT_fieldAbs2.reshape(-1))

        if self.rank == 0:
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full

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

    def vector_power_spectrum(self, name, vec):
        FT_vec = newDistArray(self.FFT,rank=1)
        for i in range(3):
            FT_vec[i] = self.FFT.forward(vec[i], FT_vec[i])

        FT_vecAbs2 = np.linalg.norm(FT_vec,axis=0)**2.
        PS_Full = self.normalized_spectrum(self.localKmag.reshape(-1),FT_vecAbs2.reshape(-1))
        
        totPowFull = self.comm.allreduce(np.sum(FT_vecAbs2))
        
        if self.rank == 0:
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full
            self.outfile.require_dataset(name + '/PowSpec/TotFull', (1,), dtype='f')[0] = totPowFull

        # project components
        localVecDotKunit = np.sum(FT_vec*self.localKunit,axis = 0)

        FT_Dil = localVecDotKunit * self.localKunit
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

        if self.rank == 0:
            self.outfile.require_dataset(name + '/PowSpec/Bins', (1,len(self.k_bins)), dtype='f')[0] = self.k_bins
            self.outfile.require_dataset(name + '/PowSpec/Full', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Full
            self.outfile.require_dataset(name + '/PowSpec/Dil', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Dil
            self.outfile.require_dataset(name + '/PowSpec/Sol', (4,len(self.k_bins)-1), dtype='f')[:,:] = PS_Sol
            self.outfile.require_dataset(name + '/PowSpec/TotSol', (1,), dtype='f')[0] = totPowSol
            self.outfile.require_dataset(name + '/PowSpec/TotDil', (1,), dtype='f')[0] = totPowDil
            self.outfile.require_dataset(name + '/PowSpec/TotHarm', (1,), dtype='f')[0] = totPowHarm

    def get_and_write_statistics_to_file(self,field,name,bounds=None):
        """
            field - 3d scalar field to get statistics from
            name - human readable name of the field
            bounds - tuple, lower and upper bound for histogram, if None then min/max
        """

        N = float(self.comm.allreduce(field.size))
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

        if self.rank == 0:
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

        Bins = np.linspace(bounds[0],bounds[1],int(self.num_bins + 1))
        hist = np.histogram(field.reshape(-1),bins=Bins)[0]
        totalHist = self.comm.allreduce(hist)

        if self.rank == 0:
            tmp  = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,int(self.num_bins + 1)), dtype='f')
            tmp[0] = Bins
            tmp[1,:-1] = totalHist.astype(float)

        if name in self.global_min_max.keys():
            Bins = np.linspace(self.global_min_max[name][0],self.global_min_max[name][1],int(self.num_bins + 1))
            hist = np.histogram(field.reshape(-1),bins=Bins)[0]
            totalHist = self.comm.allreduce(hist)

            HistBins = 'globalMinMax'

            if self.rank == 0:
                tmp  = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,int(self.num_bins + 1)), dtype='f')
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

    def get_2d_hist(self,name,X,Y,bounds = None):
        if bounds is None:
            globXMin = self.comm.allreduce(np.min(X),op=self.MPI.MIN)
            globXMax = self.comm.allreduce(np.max(X),op=self.MPI.MAX)
            
            globYMin = self.comm.allreduce(np.min(Y),op=self.MPI.MIN)
            globYMax = self.comm.allreduce(np.max(Y),op=self.MPI.MAX)

            bounds = [[globXMin,globXMax],[globYMin,globYMax]]
            HistBins = "Snap"
        else:
            HistBins = "Sim"

        XBins = np.linspace(bounds[0][0],bounds[0][1],int(self.num_bins + 1))
        YBins = np.linspace(bounds[1][0],bounds[1][1],int(self.num_bins + 1))

        hist = np.histogram2d(X.reshape(-1),Y.reshape(-1),bins=[XBins,YBins])[0]
        totalHist = self.comm.allreduce(hist)

        if self.rank == 0:
            tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,int(self.num_bins + 1)), dtype='f')
            tmp[0] = XBins
            tmp[1] = YBins
            tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/counts', (self.num_bins,self.num_bins), dtype='f')
            tmp[:,:] = totalHist.astype(float)

        Xname, Yname = name.split('-')
        if Xname in self.global_min_max.keys() and Yname in self.global_min_max.keys():
            XBins = np.linspace(self.global_min_max[Xname][0],self.global_min_max[Xname][1],int(self.num_bins + 1))
            YBins = np.linspace(self.global_min_max[Yname][0],self.global_min_max[Yname][1],int(self.num_bins + 1))

            hist = np.histogram2d(X.reshape(-1),Y.reshape(-1),bins=[XBins,YBins])[0]
            totalHist = self.comm.allreduce(hist)

            HistBins = 'globalMinMax'

            if self.rank == 0:
                tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,int(self.num_bins + 1)), dtype='f')
                tmp[0] = XBins
                tmp[1] = YBins
                tmp = self.outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/counts', (self.num_bins,self.num_bins), dtype='f')
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
