import numpy as np
import pandas as pd
import math
from FBSA import FBSA_algorithm
np.random.seed(32)
bus_data_default = pd.read_csv('./bus_33_data.csv')
line_data_default = pd.read_csv('./branch_33_data.csv')
#Max_iter: maximum iterations, N: populatoin size, Convergence_curve: Convergence curve
# ub = 100
# lb = -100
#dim = 2
N_bus = 32

ub = np.array([[1,1,1],[1,1,1],[N_bus,N_bus,N_bus]])
lb = np.array([[0,0,0],[0,0,0],[2,2,2]])
dim = (3,3)
Max_iter = 50
N = 10



def Fitfunction(x:np.array):
    
    Ploss, Qloss, Iline, V = FBSA(x, fbsa_algo)
  
    # print( Ploss, Qloss, Iline, V)
    return np.sum(Ploss)+np.sum(x[2])



fobj = Fitfunction

fbsa_algo = FBSA_algorithm(line_data=line_data_default, bus_data=bus_data_default)
def FBSA(solution, fbsa_obj):
    #print('needed FBSA_step to calculate ')
    Ploss, Qloss, Iline, V = fbsa_obj._run(solution)
    
    return Ploss, Qloss, Iline, V


def check_and_convert_bound(x:np.array):
    
    for i in range(3):
        for j in range(3):   

            if i == 2: #location number
                if x[i,j]<2:
                    x[i,j]=2
                elif x[i,j]>N_bus:
                    x[i,j]=N_bus
                else:
                    x[i,j] = int(x[i,j])
            else:
                if x[i,j] <0:
                    x[i,j] = 0
                elif x[i,j]>1:
                    x[i,j] = 1
                
    return x



def COOT(N,Max_iter,lb,ub,dim,fobj):
    #if np.isscalar(ub):
        # ub = np.ones((1, dim)) * ub
        # lb = np.ones((1, dim)) * lb
    ub = np.array([[1,1,1],[1,1,1],[N_bus,N_bus,N_bus]])
    lb = np.array([[0,0,0],[0,0,0],[2,2,2]])

    NLeader = int(np.ceil(0.1*N))
    Ncoot = N - NLeader 


    Convergence_curve = np.zeros((1,Max_iter))
    gBest = np.zeros((1,3,3))
    gBestScore = np.inf

    #Initialize the positions of Coots
    CootPos = np.random.rand(Ncoot,3,3)*(ub-lb)+lb
    CootPos[:,2] = CootPos[:,2].astype(np.int64)
    #print(CootPos)
    CootFitness=np.zeros((1,Ncoot))
    #Initialize the locations of Leaders
    LeaderPos=np.random.rand(NLeader,3,3)*(ub-lb)+lb
    LeaderPos[:,2] = LeaderPos[:,2].astype(np.int64)
    LeaderFit=np.zeros((1,NLeader))
    #print(LeaderPos)
    
    #print(CootPos)
    for i in range(Ncoot):
        #Iop, Vload = FBSA()
        CootFitness[0,i] = Fitfunction(CootPos[i,:])
        if gBestScore >  CootFitness[0,i]:
            gBestScore = CootFitness[0,i]
            gBest = CootPos[i,:]

    for i in range(NLeader):
        #Iop, Vload = FBSA()
        LeaderFit[0,i] = Fitfunction(LeaderPos[i,:])
        if gBestScore >  LeaderFit[0,i]:
            gBestScore = LeaderFit[0,i]
            gBest = LeaderPos[i,:]
    Convergence_curve[0] = gBestScore
    Iter = 1
    while Iter < Max_iter:
        for i in range(Ncoot):
            if np.random.rand()>0.5:
                W1 = np.random.rand(3,3)
                r = 1- 2*np.random.rand()
                SL =  i % NLeader
                CootPos[i:,]= LeaderPos[SL,:] +2 *W1 *math.cos(2*r*np.pi)*(LeaderPos[SL,:]-CootPos[i,:])
            else:
                if i ==1: 
                    CootPos[i,:] = 0.5*(CootPos[i-1,:] + CootPos[i,:])
                else:
                    W1 = np.random.rand()
                    H = 1 - Iter/Max_iter
                    K = np.random.rand(1,3,3)*(ub-lb)+lb
                    CootPos[i,:]=CootPos[i,:]+H*W1*(K-CootPos[i,:])

            CootPos[i,:] = check_and_convert_bound(CootPos[i,:])
        for i in range(Ncoot):
            CootFitness[0,i]=Fitfunction(CootPos[i,:])
            k=(i % NLeader); 
            if CootFitness[0,i]<LeaderFit[0,k]:
                Temp=LeaderPos[k,:]
                TemFit= LeaderFit[0,k]
                LeaderFit[0,k]= CootFitness[0,i]
                LeaderPos[k,:]=CootPos[i,:]
                CootFitness[0,i]=TemFit
                CootPos[i,:]=Temp   


        for j in range(NLeader):
            r = 2*np.random.rand()-1
            J= 2-Iter/Max_iter
            if np.random.rand()>0.5:
                W2 = np.random.rand(3,3)
                #LeaderPos[j,:] = j*W2*np.cos(2*np.pi*r)*(gBest-LeaderPos[j,:])+gBest
                Temp = J*W2*np.cos(2*np.pi*r)*(gBest-LeaderPos[j,:])+gBest
            else:
                W2 = np.random.rand()
                #LeaderPos[j,:] = j*W2*np.cos(2*np.pi*r)*(gBest-LeaderPos[j,:])+gBest
                Temp = J*W2*np.cos(2*np.pi*r)*(gBest-LeaderPos[j,:])+gBest
            Temp=check_and_convert_bound(Temp)
            # LeaderPos[j,:] = check_and_convert_bound(LeaderPos[j,:])
            TempFit = Fitfunction(Temp)
            # LeaderFit[0,j]=Fitfunction(LeaderPos[j,:])
            #     # % Update the location of Leader
            # if gBestScore>LeaderFit[0,j] :
            #     LeaderFit[0,j]=gBestScore
            #     LeaderPos[j,:]=gBest
            #     gBestScore=LeaderFit[0,j]
            #     gBest=LeaderPos[0,j]
            
                # % Update the location of Leader
            if gBestScore>TempFit :
                LeaderFit[0,j]=gBestScore
                LeaderPos[j,:]=gBest
                gBestScore=TempFit
                gBest=Temp
        Convergence_curve[0,Iter] = gBestScore
        Iter +=1
        # print(gBest)
    return gBest, gBestScore, Convergence_curve
    
    # return 

solution = np.random.rand(10,3,3)*(ub-lb)+lb

solution[:,2] = solution[:,2].astype(np.int64)


print('start')

print(COOT(lb=lb, ub=ub, dim=dim, fobj=fobj, N=N, Max_iter=Max_iter))

print('end')