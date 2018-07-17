import numpy as np
from mpi4py_fft.mpifft import PFFT, Function
from MPIderivHelperFuncs import MPIderiv2, MPIXdotGradYScalar, MPIXdotGradY, MPIdivX, MPIdivXY, MPIgradX
import time
import pickle
import sys

class EnergyTransfer:

    
    def __init__(self, MPI, RES, rho, U, B, Acc, P, gamma):
        
        self.gamma = gamma
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.RES = RES
        self.rho = rho
        self.U = U
        self.B = B
        self.Acc = Acc
        self.P = P
        
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
        
        if self.comm.Get_rank() == 0:
            print("""!!! WARNING - CURRENT PITFALLS !!!
            - data units are ignored
            - data is assumed to live on a 3d uniform grid with L = 1
            - for the FFT L = 2 pi is implicitly assumed to work with integer wavenumbers
            - data is assumed to be split equally on axis 0 among MPI processes            
            """)

        TimeStart = MPI.Wtime()
        
        N = np.array([RES,RES,RES], dtype=int)
        # using L = 2pi as we work (e.g. when binning) with integer wavenumbers
        L = np.array([2*np.pi, 2*np.pi, 2*np.pi], dtype=float)
        self.FFT = PFFT(self.comm,N, axes=(0,1,2),collapse=False, slab=True,dtype=np.float64)

        localK = self.FFT.get_local_wavenumbermesh(L)
        self.localKmag = np.linalg.norm(localK,axis=0)

        TimeDoneSetup = MPI.Wtime() - TimeStart
        TimeDoneSetup = self.comm.gather(TimeDoneSetup)

        if self.comm.Get_rank() == 0:
            print("Setup up FFT and wavenumbers done in %.3g +/- %.3g" %
                (np.mean(TimeDoneSetup), np.std(TimeDoneSetup)))
            sys.stdout.flush()
        
        
    def getShellX(self,FTquant,Low,Up):
        """ extracts shell X-0.5 < K <X+0.5 of FTquant """

        if FTquant.shape[0] == 3:    
            Quant_X = Function(self.FFT,False,tensor=3)
            for i in range(3):
                tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant[i],0.)
                Quant_X[i] = self.FFT.backward(tmp,Quant_X[i])
        else:
            Quant_X = Function(self.FFT,False)
            tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant,0.)
            Quant_X = self.FFT.backward(tmp,Quant_X)        

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
            self.W = Function(self.FFT,False,tensor=3)                                
            for i in range(3):
                self.W[i] = np.sqrt(rho) * U[i]

        if self.S is None and P is not None:
            self.S = np.sqrt(self.gamma*P)

        if self.FT_W is None:
            self.FT_W = Function(self.FFT,tensor=3)
            for i in range(3):
                self.FT_W[i] = self.FFT.forward(self.W[i], self.FT_W[i])            
            
        if self.FT_B is None and self.B is not None:
            self.FT_B = Function(self.FFT,tensor=3)
            for i in range(3):
                self.FT_B[i] = self.FFT.forward(self.B[i], self.FT_B[i])    
        
        if self.FT_P is None and self.P is not None:
            self.FT_P = Function(self.FFT)
            self.FT_P = self.FFT.forward(self.P, self.FT_P)    
        
        if self.FT_S is None and self.S is not None:
            self.FT_S = Function(self.FFT)
            self.FT_S = self.FFT.forward(self.S, self.FT_S)    
        
        if self.FT_Acc is None and self.Acc is not None:
            self.FT_Acc = Function(self.FFT,tensor=3)
            for i in range(3):
                self.FT_Acc[i] = self.FFT.forward(self.Acc[i], self.FT_Acc[i])    
            
    
    def getTransferWWAnyToAny(self,Result,KBins,QBins, Terms = []):
        """ return what
                    formalism -- determined by the definiton of the spectral kinetic energy density
                "WW": E_kin(k) = 1/2 |FT(sqrt(rho)U)|^2
        
        Args:
            Result -- a (potentially empty) dictionary to store the results in
            Ks -- range of destination shell wavenumber
            Qs -- range of source shell wavenumbers
            
        KWArgs:               
            terms -- list of terms that should be analyzed, could be
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
        FT_S = self.FT_S
        FT_B = self.FT_B
        FT_P = self.FT_P
        FT_Acc = self.FT_Acc

        startTime = time.time()

        # clear Q terms
        W_Q = None
        S_Q = None
        B_Q = None
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
                    
                    
                    localSum = - np.sum(S_K * UdotGradS_Q)              

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - np.sum(0.5 * S_K * S_Q * DivU)                    

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
                        
                    if BdotGradW_QoverSqrtRho is None:
                        BdotGradW_QoverSqrtRho = MPIXdotGradY(self.comm,B,W_Q/np.sqrt(rho))    
     
                    # B_K * (B dot grad) W_Q/sqrt(rho) - Moss et al
                    localSum = np.sum(B_K * BdotGradW_QoverSqrtRho)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTa","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBTa for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                        
                        
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

                # clear K terms
                W_K = None
                S_K = None
                B_K = None
        

            # clear Q terms
            W_Q = None
            S_Q = None
            B_Q = None
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
            
            Str = ""
            for Term in Terms:
                Str += "-" + Term            
            if self.comm.Get_rank() != 0 and False:
                pickle.dump(Result,open("tmp%s.pkl" % Str,"wb")) 
