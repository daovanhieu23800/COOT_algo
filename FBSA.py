import numpy as np
import pandas as pd

# bus_data = pd.read_csv('/Users/hieudao/Desktop/coot_related_pdf/Coot_algorithm/bus_33_data.csv')
# line_data = pd.read_csv('/Users/hieudao/Desktop/coot_related_pdf/Coot_algorithm/branch_33_data.csv')
class FBSA_algorithm():
    def __init__(self, line_data:pd.DataFrame , bus_data:pd.DataFrame ):
        self.bus_data = bus_data
        self.line_data = line_data


    def _update(self, x:np.array):

        Q_pvu = np.tan(np.arccos(x[1]))*x[0]
        # print(Q_pvu)
        updated_bus_data = self.bus_data
        for i in range(3):
            # print(updated_bus_data.loc[updated_bus_data['Bus_Number']==x[2,i]])
            updated_bus_data.loc[updated_bus_data['Bus_Number']==x[2,i], 'Active_Power_P(kW)'] += x[0,i]
            updated_bus_data.loc[updated_bus_data['Bus_Number']==x[2,i], 'Reactive_Power_Q(kBVar)'] += Q_pvu[i]
            # print(updated_bus_data.loc[updated_bus_data['Bus_Number']==x[2,i]])


        #print(x)
        return updated_bus_data, self.line_data
        # return self.bus_data, self.line_data

    def _run(self, solution:np.array):
        updated_bus_data, up_dateted_line_data =self._update(solution)
        bus_np = updated_bus_data.to_numpy()#.astype(np.float256)
        line_np = up_dateted_line_data.to_numpy()#.astype(np.float256)

        Sbase = 100  # MVA
        Vbase = 12.6   # KV
        Zbase = (Vbase ** 2) / Sbase
        #convert bd, ld
        line_np[:,3:5] = line_np[:,3:5]/Zbase
        bus_np[:,1:3] = bus_np[:,1:3]/(1000*Sbase)
      
        N = int(np.max(line_np[:, 1:3])) 

        Sload = bus_np[:,1] +1j*bus_np[:,2]
        Z = line_np[:,3] +1j *line_np[:,4]

        V = np.ones(bus_np.shape[0], dtype=np.complex256)
        Iline = np.zeros(line_np.shape[0], dtype=np.complex256)
        Max_iter = 200
        for i in range(Max_iter):
            Iload = np.conj(Sload/V)
            # print(np.sum(Iload))
            for j in range(line_np.shape[0]-1,-1,-1):
                c,e =np.where(line_np[:,1:3]==line_np[j,2])
                if (c.shape[0]==1):
                    Iline[int(line_np[j,0])-1]=Iload[int(line_np[j,2])-1]
                else:
                    Iline[int(line_np[j,0])-1]=Iload[int(line_np[j,2])-1]+np.sum(Iline[(line_np[c, 0].astype(np.int64) - 1)]) - Iline[int(line_np[j, 0]) - 1]
                    #print(line_np[c, 0].astype(np.int64),line_np[c, 0].astype(np.int64)-1)
            # print(np.sum(Iline))
            for j in range(line_np.shape[0]):
                V[int(line_np[j, 2]) - 1] = V[int(line_np[j, 1]) - 1] - Iline[int(line_np[j, 0]) - 1] * Z[j]
            # print(np.sum(V),'\n','---------------')
        Voltage=np.abs(V)
        Vangle=np.angle(V)
        Ploss=np.real(Z*(np.abs(Iline**2)))
        Qloss=np.imag(Z*(np.abs(Iline**2)))
        TPl=np.sum(Ploss)*Sbase*1000
        TQl=np.sum(Qloss)*Sbase*1000
        # print(TPl,TQl)

        
        return Ploss, Qloss, Iline, V, TPl, TQl
    
# bus_data_default = pd.read_csv('./bus_33_data.csv')
# line_data_default = pd.read_csv('./branch_33_data.csv')
# test = FBSA_algorithm(line_data=line_data_default, bus_data=bus_data_default)
# Ploss, Qloss, Iline, V, TPl, TQl = test._run(
#    np.array([[0.77128536, 0.28854043 ,0.21724623],
#  [1.    ,     1.     ,    1.        ],
#  [9.    ,     3.      ,   8.        ]]))
# print(TPl, TQl)