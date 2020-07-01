import numpy as np
import FFTHelperFuncs
#from mpi4py_fft import newDistArray
from MPIderivHelperFuncs import MPIderiv2, MPIXdotGradYScalar, MPIXdotGradY, MPIdivX, MPIdivXY, MPIgradX, MPIVecLaplacian
import time
import pickle
import sys

class EnergyTransfer:

    
    def __init__(self, MPI, RES, fields, gamma,outfile):
        
        self.gamma = gamma
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.RES = RES
        self.outfile = outfile

        self.rho = fields['rho']
        self.U = fields['U']
        self.B = fields['B']
        self.Acc = fields['Acc']
        self.P = fields['P']

        # Variables that we might (or might not) use later depending on the different definitons of terms
        self.W = None
        self.FT_W = None
        self.S = None
        self.FT_S = None
        self.FT_B = None
        self.FT_Acc = None
        self.FT_P = None
        self.FT_rho = None
        self.FT_U = None

        self.FFT = FFTHelperFuncs.FFT
        
        self.localKmag = np.linalg.norm(FFTHelperFuncs.local_wavenumbermesh,axis=0)


    def getShellX(self,FTquant,Low,Up):
        """ extracts shell X-0.5 < K <X+0.5 of FTquant """

        if FTquant.shape[0] == 3:    
            Quant_X = np.zeros((3,) + FFTHelperFuncs.local_shape,dtype=np.float64)
            for i in range(3):
                tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant[i],0.)
                #print("got here")
                #Quant_X[i] =
                #tmp = self.FFT.fft(self.W[i])
                #print("done FFT",self.FT_W[i].shape, self.FT_W[i].dtype, tmp.shape, tmp.dtype)
                Quant_X[i] = self.FFT.ifft(tmp)
                #print("but not here")
        else:
            #Quant_X = np.zeros(FFTHelperFuncs.local_shape,dtype=np.float64)
            tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant,0.)
            Quant_X = self.FFT.ifft(tmp)        

        return Quant_X
    
    
    def populateResultDict(self,Result,KBins,formalism,Terms,method):
        if self.comm.Get_rank() != 0:
            return
            
        if formalism not in Result.keys():
            Result[formalism] = {}
        
        for term in Terms:
            if term not in Result[formalism].keys():
                Result[formalism][term] = {}
        
            
            if method not in Result[formalism][term].keys():
                Result[formalism][term][method] = {}
                            

            for i in range(len(KBins)-1):
                KBin = "%.2f-%.2f" % (KBins[i],KBins[i+1])
                if KBin not in Result[formalism][term][method].keys():
                    Result[formalism][term][method][KBin] = {}              

    def addResultToDict(self,Result,formalism,term,method,KBin,QBin,value):
        if self.comm.Get_rank() != 0:
            return
            
        if formalism not in Result.keys():
            Result[formalism] = {}
        

        if term not in Result[formalism].keys():
            Result[formalism][term] = {}


        if method not in Result[formalism][term].keys():
            Result[formalism][term][method] = {}

        if KBin not in Result[formalism][term][method].keys():
            Result[formalism][term][method][KBin] = {}
                    
        Result[formalism][term][method][KBin][QBin] = float(value)
                    
    def calcBasicVars(self,formalism):
        """ calculate basic variables for the different formalisms, i.e.
        W and FT_W for WW formalism, and
        """
        
        rho = self.rho
        P = self.P
        U = self.U
        B = self.B

        if self.W is None:
            self.W = np.zeros((3,) + FFTHelperFuncs.local_shape,dtype=np.float64)             
            for i in range(3):
                self.W[i] = np.sqrt(rho) * U[i]

        if self.S is None and self.gamma is not None and P is not None:
            self.S = np.sqrt(self.gamma*P)

        if self.FT_W is None:
            self.FT_W = np.zeros((3,) + self.localKmag.shape,dtype=np.complex128)
            for i in range(3):
                self.FT_W[i] = self.FFT.fft(self.W[i])            
            
        if self.FT_U is None:
            self.FT_U = np.zeros((3,) + self.localKmag.shape,dtype=np.complex128)
            for i in range(3):
                self.FT_U[i] = self.FFT.fft(self.U[i])

        if self.FT_B is None and self.B is not None:
            self.FT_B = np.zeros((3,) + self.localKmag.shape,dtype=np.complex128)
            for i in range(3):
                self.FT_B[i] = self.FFT.fft(self.B[i])    
        
        if self.FT_P is None and self.P is not None:
            self.FT_P = np.zeros(self.localKmag.shape,dtype=np.complex128)
            self.FT_P = self.FFT.fft(self.P)    
        
        if self.FT_S is None and self.S is not None:
            self.FT_S = np.zeros(self.localKmag.shape,dtype=np.complex128)
            self.FT_S = self.FFT.fft(self.S)    
        
        if self.FT_Acc is None and self.Acc is not None:
            self.FT_Acc = np.zeros((3,) + self.localKmag.shape,dtype=np.complex128)
            for i in range(3):
                self.FT_Acc[i] = self.FFT.fft(self.Acc[i])    
            
    
    def getTransferWWAnyToAny(self, Result, KBins, QBins, Terms):
        """ return what
                    formalism -- determined by the definiton of the spectral kinetic energy density
                "WW": E_kin(k) = 1/2 |FT(sqrt(rho)U)|^2
        
        Args:
            Result -- a (potentially empty) dictionary to store the results in
            Ks -- range of destination shell wavenumber
            Qs -- range of source shell wavenumbers
            Terms -- list of terms that should be analyzed, could be
                "UUA": Kinetic to kinetic by advection        
        """
        #self.populateResultDict(Result,KBins,"WW",Terms,"AnyToAny")
        self.calcBasicVars("WW")
        
        
        rho = self.rho
        U = self.U
        B = self.B
        S = self.S
        W = self.W
        FT_W = self.FT_W
        FT_U = self.FT_U
        FT_S = self.FT_S
        FT_B = self.FT_B
        FT_P = self.FT_P
        FT_Acc = self.FT_Acc

        startTime = time.time()

        # clear Q terms
        W_Q = None
        U_Q = None
        S_Q = None
        B_Q = None
        SDivW_QoverGammaSqrtRho = None
        OneOverGammaSqrtRhogradSS_Q = None
        OneOverTwoSqrtRhogradBB_Q = None
        UdotGradW_Q = None
        UdotGradS_Q = None
        UdotGradB_Q = None
        bDotGradB_Q = None
        BdotGradW_QoverSqrtRho = None
        DivbW_Q = None
        bdotGradW_Q = None
        W_QoverSqrtRho = None
        W_QoverSqrtRhoDotGradB = None
        DivW_QoverSqrtRho = None        
        DivW_Qb = None
        W_QdotGradb = None
        DivW_Q = None
        BDivW_Qover2SqrtRho = None
        OneOverSqrtRhoGradP_Q = None
        SqrtRhoAcc_Q = None
        SqrtRhoDeltaU_Q = None
        SqrtRhoDelta2U_Q = None
        DeltaB_Q = None
        Delta2B_Q = None
        GradDivU_Q = None
        
        DivU = None
        b = None
        Divb = None
        
        for q in range(len(QBins)-1):
            QBin = "%.2f-%.2f" % (QBins[q],QBins[q+1])

            # clear K terms
            W_K = None
            S_K = None	
            B_K = None
            
            for k in range(len(KBins)-1):
                
                KBin = "%.2f-%.2f" % (KBins[k],KBins[k+1])

                #  - W_K * (U dot grad) W_Q - 0.5 W_K W_Q DivU
                if "UU" in Terms:
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                        
                        
                    if UdotGradW_Q is None:
                        UdotGradW_Q = MPIXdotGradY(self.comm,U,W_Q)                        
                    
                    if DivU is None:
                        DivU = MPIdivX(self.comm,U)
                    
                    
                    localSum = - np.sum(W_K * UdotGradW_Q)              

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - np.sum(0.5 * W_K * W_Q * DivU)                    

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                    
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UUA","AnyToAny",KBin,QBin,totalSumA)
                        self.addResultToDict(Result,"WW","UUC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UU","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))   
                
                #  - S_K * (U dot grad) S_Q - 0.5 S_K S_Q DivU
                if "SS" in Terms:
                    if S_K is None:
                        S_K = self.getShellX(FT_S,KBins[k],KBins[k+1])
                    
                    if S_Q is None:
                        S_Q = self.getShellX(FT_S,QBins[q],QBins[q+1])                        
                        
                    if UdotGradS_Q is None:
                        UdotGradS_Q = MPIXdotGradYScalar(self.comm,U,S_Q)                        
                    
                    if DivU is None:
                        DivU = MPIdivX(self.comm,U)
                    
                    
                    localSum = - 2./self.gamma/(self.gamma - 1.) * np.sum(S_K * UdotGradS_Q)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - 1./self.gamma/(self.gamma - 1.) * np.sum(S_K * S_Q * DivU)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                    
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","SSA","AnyToAny",KBin,QBin,totalSumA)
                        self.addResultToDict(Result,"WW","SSC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","SS","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with SS for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))   
                
                if "BB" in Terms:
                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])
                    
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])                        
                        
                    if UdotGradB_Q is None:
                        UdotGradB_Q = MPIXdotGradY(self.comm,U,B_Q)                        
                    
                    if DivU is None:
                        DivU = MPIdivX(self.comm,U)
                    
                    
                    localSum = - np.sum(B_K * UdotGradB_Q)              

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - np.sum(0.5 * B_K * B_Q * DivU)                    

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                    
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BBA","AnyToAny",KBin,QBin,totalSumA)
                        self.addResultToDict(Result,"WW","BBC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","BB","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with BB for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                

                # W_K * (1/sqrt(rho) B dot grad) B_Q
                if "BUT" in Terms:
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                        
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])
                    
                    if b is None:
                        b = B/np.sqrt(rho)
                        
                    if bDotGradB_Q is None:
                        bDotGradB_Q = MPIXdotGradY(self.comm,b,B_Q)                        
                        
                   
                    localSum = np.sum(W_K * bDotGradB_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BUT","AnyToAny",KBin,QBin,totalSum)
                        print("done with BUT for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                
                if "UBT" in Terms:
                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                                          
                        
                    #if BdotGradW_QoverSqrtRho is None:
                    #    BdotGradW_QoverSqrtRho = MPIXdotGradY(self.comm,B,W_Q/np.sqrt(rho))    

                    ## B_K * (B dot grad) W_Q/sqrt(rho) - Moss et al
                    #localSum = np.sum(B_K * BdotGradW_QoverSqrtRho)

                    #totalSum = None
                    #totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    #if self.comm.Get_rank() == 0:
                    #    self.addResultToDict(Result,"WW","UBTa","AnyToAny",KBin,QBin,totalSum)
                    #    print("done with UBTa for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                    if b is None:
                        b = B/np.sqrt(rho)
                        
                    if DivbW_Q is None:
                        DivbW_Q = MPIdivXY(self.comm,b,W_Q)
                        
                    # B^K_i pd_j b_j W^Q_i - total term
                    localSum = np.sum(B_K * DivbW_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTb","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBTb for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                        
                    # B^K_i  b_j pd_j W^Q_i - "adv" term                    
                    if bdotGradW_Q is None:
                        bdotGradW_Q = MPIXdotGradY(self.comm,b,W_Q) 
                        
                    localSum = np.sum(B_K * bdotGradW_Q)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTbA","AnyToAny",KBin,QBin,totalSumA)
                        print("done with UBTbA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        

                        
                    # B^K_i  W^Q_i  pd_j b_j - "compr" term                    
                    if Divb is None:
                        Divb = MPIdivX(self.comm,b)
                        
                    localSum = np.sum(B_K * W_Q *Divb)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTbC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UBTbTot","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UBTbC for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                        


                # B * (1/sqrt(rho) W_K dot grad) B_Q
                if "BUP" in Terms:
                    
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])
                    
                    if b is None:
                        b = B/np.sqrt(rho)
                        
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                        
                    W_KDotGradB_Q = MPIXdotGradY(self.comm,W_K,B_Q)
                    
                    localSum = - np.sum(b * W_KDotGradB_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BUP","AnyToAny",KBin,QBin,totalSum)
                        print("done with BUP for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # this is the term with split BB
                if "BUPbb" in Terms:
                    
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])
                        
                    if OneOverTwoSqrtRhogradBB_Q is None:
                        OneOverTwoSqrtRhogradBB_Q = MPIgradX(self.comm, np.sum(B * B_Q,axis=0))/ (2. * np.sqrt(rho))
                        
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                                            
                    
                    localSum = - np.sum(W_K * OneOverTwoSqrtRhogradBB_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BUPbb","AnyToAny",KBin,QBin,totalSum)
                        print("done with BUPbb for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                
                if "UBPa" in Terms:
                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                        
                    
                    if W_QoverSqrtRho is None:
                        W_QoverSqrtRho = W_Q/np.sqrt(rho)
                    
                    if W_QoverSqrtRhoDotGradB is None:
                        W_QoverSqrtRhoDotGradB = MPIXdotGradY(self.comm,W_QoverSqrtRho,B)
                    
                    localSum = - np.sum(B_K * W_QoverSqrtRhoDotGradB)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPaA","AnyToAny",KBin,QBin,totalSumA)
                        print("done with UBPaA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                    if DivW_QoverSqrtRho is None:
                        DivW_QoverSqrtRho = MPIdivX(self.comm,W_QoverSqrtRho) 
    
                    
                    localSum = - np.sum(B_K * B * DivW_QoverSqrtRho)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPaC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UBPaTot","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UBPaC for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))    

                if "UBPbb" in Terms:

                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])  
                        
                    if BDivW_Qover2SqrtRho is None:
                        BDivW_Qover2SqrtRho = B * MPIdivX(self.comm, W_Q/np.sqrt(rho)/2. )                         
                        
                    localSum = - np.sum(B_K * BDivW_Qover2SqrtRho)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                        
                        
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPbb","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBPbb for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                if "UBPb" in Terms:

                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                        
                    
                    if b is None:
                        b = B/np.sqrt(rho)
                        
                    if DivW_Qb is None:
                        DivW_Qb = MPIdivXY(self.comm,W_Q,b)
                        
                    localSum = - np.sum(B_K * DivW_Qb)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                        
                        
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPb","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBPbA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
            
                    
                    if W_QdotGradb is None:
                        W_QdotGradb = MPIXdotGradY(self.comm,W_Q,b)
                    
                    localSum = - np.sum(B_K * W_QdotGradb)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPbA","AnyToAny",KBin,QBin,totalSumA)
                        print("done with UBPbA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                    if DivW_Q is None:
                        DivW_Q = MPIdivX(self.comm,W_Q) 
    
                    
                    localSum = - np.sum(B_K * B * DivW_Q)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPbC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UBPbTot","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UBPbC for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))    
                
                
                if "SU" in Terms:
                    
                    if S_Q is None:
                        S_Q = self.getShellX(FT_S,QBins[q],QBins[q+1])
                        
                    # TODO reuse vars here with BUP terms
                    if OneOverGammaSqrtRhogradSS_Q is None:
                        OneOverGammaSqrtRhogradSS_Q = MPIgradX(self.comm, (S * S_Q))/ (self.gamma * np.sqrt(rho))
                        
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                                            
                    
                    localSum = - np.sum(W_K * OneOverGammaSqrtRhogradSS_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","SU","AnyToAny",KBin,QBin,totalSum)
                        print("done with SU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        

                if "US" in Terms:

                    if S_K is None:
                        S_K = self.getShellX(FT_S,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])  
                    # TODO reuse vars here with BUP terms
                    if SDivW_QoverGammaSqrtRho is None:
                        SDivW_QoverGammaSqrtRho = S * MPIdivX(self.comm, W_Q/np.sqrt(rho)/self.gamma )                         
                        
                    localSum = - np.sum(S_K * SDivW_QoverGammaSqrtRho)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                        
                        
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","US","AnyToAny",KBin,QBin,totalSum)
                        print("done with US for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        

                # - W_K 1/sqrt(rho) grad P_Q
                if "PU" in Terms:

                    if OneOverSqrtRhoGradP_Q is None:
                        P_Q = self.getShellX(FT_P,QBins[q],QBins[q+1])
                        OneOverSqrtRhoGradP_Q = MPIgradX(self.comm, P_Q)/ np.sqrt(rho)
                        del P_Q
                    
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    localSum = - np.sum(W_K * OneOverSqrtRhoGradP_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","PU","AnyToAny",KBin,QBin,totalSum)
                        print("done with PU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                
                # W_K sqrt(rho) Acc_Q
                if "FU" in Terms:

                    if SqrtRhoAcc_Q is None:
                        SqrtRhoAcc_Q = np.sqrt(rho) * self.getShellX(FT_Acc,QBins[q],QBins[q+1])
                    
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    localSum = np.sum(W_K * SqrtRhoAcc_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","FU","AnyToAny",KBin,QBin,totalSum)
                        print("done with FU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # W_K sqrt(rho) nu Delta U_Q (nu is not used here, but later
                # determined empirically
                if "nuU" in Terms:

                    if U_Q is None:
                        U_Q = self.getShellX(FT_U,QBins[q],QBins[q+1])

                    if SqrtRhoDeltaU_Q is None or SqrtRhoDelta2U_Q is None:
                        d2U_Q0dxx = MPIderiv2(self.comm,U_Q[0],0,deriv=2)
                        d2U_Q0dyy = MPIderiv2(self.comm,U_Q[0],1,deriv=2)
                        d2U_Q0dzz = MPIderiv2(self.comm,U_Q[0],2,deriv=2)
                        d2U_Q1dxx = MPIderiv2(self.comm,U_Q[1],0,deriv=2)
                        d2U_Q1dyy = MPIderiv2(self.comm,U_Q[1],1,deriv=2)
                        d2U_Q1dzz = MPIderiv2(self.comm,U_Q[1],2,deriv=2)
                        d2U_Q2dxx = MPIderiv2(self.comm,U_Q[2],0,deriv=2)
                        d2U_Q2dyy = MPIderiv2(self.comm,U_Q[2],1,deriv=2)
                        d2U_Q2dzz = MPIderiv2(self.comm,U_Q[2],2,deriv=2)

                        SqrtRhoDeltaU_Q = np.sqrt(rho) * np.array(
                            [d2U_Q0dxx + d2U_Q0dyy + d2U_Q0dzz,
                             d2U_Q1dxx + d2U_Q1dyy + d2U_Q1dzz,
                             d2U_Q2dxx + d2U_Q2dyy + d2U_Q2dzz])

                        d4U_Q0dxxxx = MPIderiv2(self.comm,d2U_Q0dxx,0,deriv=2)
                        d4U_Q0dyyyy = MPIderiv2(self.comm,d2U_Q0dyy,1,deriv=2)
                        d4U_Q0dzzzz = MPIderiv2(self.comm,d2U_Q0dzz,2,deriv=2)
                        d4U_Q1dxxxx = MPIderiv2(self.comm,d2U_Q1dxx,0,deriv=2)
                        d4U_Q1dyyyy = MPIderiv2(self.comm,d2U_Q1dyy,1,deriv=2)
                        d4U_Q1dzzzz = MPIderiv2(self.comm,d2U_Q1dzz,2,deriv=2)
                        d4U_Q2dxxxx = MPIderiv2(self.comm,d2U_Q2dxx,0,deriv=2)
                        d4U_Q2dyyyy = MPIderiv2(self.comm,d2U_Q2dyy,1,deriv=2)
                        d4U_Q2dzzzz = MPIderiv2(self.comm,d2U_Q2dzz,2,deriv=2)

                        SqrtRhoDelta2U_Q = np.sqrt(rho) * np.array(
                            [d4U_Q0dxxxx + d4U_Q0dyyyy + d4U_Q0dzzzz,
                             d4U_Q1dxxxx + d4U_Q1dyyyy + d4U_Q1dzzzz,
                             d4U_Q2dxxxx + d4U_Q2dyyyy + d4U_Q2dzzzz])

                        del d2U_Q0dxx,d2U_Q0dyy,d2U_Q0dzz,d2U_Q1dxx,d2U_Q1dyy,d2U_Q1dzz
                        del d2U_Q2dxx,d2U_Q2dyy,d2U_Q2dzz,d4U_Q0dxxxx,d4U_Q0dyyyy
                        del d4U_Q0dzzzz,d4U_Q1dxxxx,d4U_Q1dyyyy,d4U_Q1dzzzz
                        del d4U_Q2dxxxx,d4U_Q2dyyyy,d4U_Q2dzzzz

                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])

                    localSum = np.sum(W_K * SqrtRhoDeltaU_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","nuU","AnyToAny",KBin,QBin,totalSum)
                        print("done with nuU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                    localSum = np.sum(W_K * SqrtRhoDelta2U_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","nu2U","AnyToAny",KBin,QBin,totalSum)
                        print("done with nu2U for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # W_K sqrt(rho) nu 1/3 grad div U_Q (nu is not used here, but later
                # determined empirically
                if "nuDivU" in Terms:

                    if U_Q is None:
                        U_Q = self.getShellX(FT_U,QBins[q],QBins[q+1])

                    if GradDivU_Q is None:
                        GradDivU_Q = MPIgradX(self.comm,MPIdivX(self.comm,U_Q))

                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])

                    localSum = np.sum(W_K * GradDivU_Q / 3.0)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","nuDivU","AnyToAny",KBin,QBin,totalSum)
                        print("done with nuDivU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # B_K eta Delta B_Q (eta is not used here, but later
                # determined empirically
                if "etaB" in Terms:

                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])

                    if DeltaB_Q is None or Delta2B_Q is None:
                        d2B_Q0dxx = MPIderiv2(self.comm,B_Q[0],0,deriv=2)
                        d2B_Q0dyy = MPIderiv2(self.comm,B_Q[0],1,deriv=2)
                        d2B_Q0dzz = MPIderiv2(self.comm,B_Q[0],2,deriv=2)
                        d2B_Q1dxx = MPIderiv2(self.comm,B_Q[1],0,deriv=2)
                        d2B_Q1dyy = MPIderiv2(self.comm,B_Q[1],1,deriv=2)
                        d2B_Q1dzz = MPIderiv2(self.comm,B_Q[1],2,deriv=2)
                        d2B_Q2dxx = MPIderiv2(self.comm,B_Q[2],0,deriv=2)
                        d2B_Q2dyy = MPIderiv2(self.comm,B_Q[2],1,deriv=2)
                        d2B_Q2dzz = MPIderiv2(self.comm,B_Q[2],2,deriv=2)

                        DeltaB_Q = np.array(
                            [d2B_Q0dxx + d2B_Q0dyy + d2B_Q0dzz,
                             d2B_Q1dxx + d2B_Q1dyy + d2B_Q1dzz,
                             d2B_Q2dxx + d2B_Q2dyy + d2B_Q2dzz])

                        d4B_Q0dxxxx = MPIderiv2(self.comm,d2B_Q0dxx,0,deriv=2)
                        d4B_Q0dyyyy = MPIderiv2(self.comm,d2B_Q0dyy,1,deriv=2)
                        d4B_Q0dzzzz = MPIderiv2(self.comm,d2B_Q0dzz,2,deriv=2)
                        d4B_Q1dxxxx = MPIderiv2(self.comm,d2B_Q1dxx,0,deriv=2)
                        d4B_Q1dyyyy = MPIderiv2(self.comm,d2B_Q1dyy,1,deriv=2)
                        d4B_Q1dzzzz = MPIderiv2(self.comm,d2B_Q1dzz,2,deriv=2)
                        d4B_Q2dxxxx = MPIderiv2(self.comm,d2B_Q2dxx,0,deriv=2)
                        d4B_Q2dyyyy = MPIderiv2(self.comm,d2B_Q2dyy,1,deriv=2)
                        d4B_Q2dzzzz = MPIderiv2(self.comm,d2B_Q2dzz,2,deriv=2)

                        Delta2B_Q =np.array(
                            [d4B_Q0dxxxx + d4B_Q0dyyyy + d4B_Q0dzzzz,
                             d4B_Q1dxxxx + d4B_Q1dyyyy + d4B_Q1dzzzz,
                             d4B_Q2dxxxx + d4B_Q2dyyyy + d4B_Q2dzzzz])

                        del d2B_Q0dxx,d2B_Q0dyy,d2B_Q0dzz,d2B_Q1dxx,d2B_Q1dyy,d2B_Q1dzz
                        del d2B_Q2dxx,d2B_Q2dyy,d2B_Q2dzz,d4B_Q0dxxxx,d4B_Q0dyyyy
                        del d4B_Q0dzzzz,d4B_Q1dxxxx,d4B_Q1dyyyy,d4B_Q1dzzzz
                        del d4B_Q2dxxxx,d4B_Q2dyyyy,d4B_Q2dzzzz

                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])

                    localSum = np.sum(B_K * DeltaB_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","etaB","AnyToAny",KBin,QBin,totalSum)
                        print("done with etaB for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                    localSum = np.sum(B_K * Delta2B_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","eta2B","AnyToAny",KBin,QBin,totalSum)
                        print("done with eta2B for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # clear K terms
                W_K = None
                S_K = None
                B_K = None
        

            # clear Q terms
            W_Q = None
            U_Q = None
            S_Q = None
            B_Q = None
            OneOverTwoSqrtRhogradBB_Q = None
            SDivW_QoverGammaSqrtRho  = None
            OneOverGammaSqrtRhogradSS_Q = None
            UdotGradW_Q = None
            UdotGradS_Q = None
            UdotGradB_Q = None
            bDotGradB_Q = None
            BdotGradW_QoverSqrtRho = None
            DivbW_Q = None
            bdotGradW_Q = None
            W_QoverSqrtRho = None
            W_QoverSqrtRhoDotGradB = None
            DivW_QoverSqrtRho = None
            DivW_Qb = None
            W_QdotGradb = None
            DivW_Q = None
            BDivW_Qover2SqrtRho = None
            OneOverSqrtRhoGradP_Q = None
            SqrtRhoAcc_Q = None
            SqrtRhoDeltaU_Q = None
            SqrtRhoDelta2U_Q = None
            DeltaB_Q = None
            Delta2B_Q = None
            GradDivU_Q = None
            
            if self.comm.Get_rank() == 0 and True:
                pickle.dump(Result,open(self.outfile + ".tmp","wb")) 
