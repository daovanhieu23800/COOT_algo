import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
bus_data_default = pd.read_csv('./bus_33_data.csv')
line_data_default = pd.read_csv('./branch_33_data.csv')

class FBSA_algorithm():
    def __init__(self, bus_data ,line_data) -> None:
        self.bus_data = bus_data
        self.line_data = line_data


    def _run(self):
        
        BD = self.bus_data.to_numpy().astype(np.float64)
        LD = self.line_data.to_numpy().astype(np.float64)

        Sbase = 100  # MVA
        Vbase = 12.6   # KV
        Zbase = (Vbase ** 2) / Sbase

        LD[:, 3:5] = LD[:, 3:5] / Zbase
        BD[:, 1:3] = BD[:, 1:3] / (1000 * Sbase)

        N = int((np.max(LD[:, 2:3])))

        Sload = BD[:, 1] + 1j * BD[:, 2]

        V = np.ones(BD.shape[0], dtype=complex)
        Z = LD[:, 3] + 1j * LD[:, 4]

        Iline = np.zeros(LD.shape[0], dtype=complex)

        Iter = 2000
        
        temp = V

        # The Algorithm
        for i in range(Iter):
        #i=0
        #while(True):
            # Backward Sweep
            Iload = np.conj(Sload / V)
            for j in range(LD.shape[0] - 1, -1, -1):
                c, e = np.where(LD[:, 1:3] == LD[j, 2])
                if c.shape[0] == 1:
                    Iline[int(LD[j, 0]) - 1] = Iload[int(LD[j, 2]) - 1]
                else:
                
                    Iline[int(LD[j, 0]) - 1] = Iload[int(LD[j, 2]) - 1] + np.sum(Iline[(LD[c, 0].astype(np.int64) - 1)]) - Iline[int(LD[j, 0]) - 1]
            # Forward Sweep
            for j in range(LD.shape[0]):
                V[int(LD[j, 2]) - 1] = V[int(LD[j, 1]) - 1] - Iline[int(LD[j, 0]) - 1] * Z[j]
            
            
        Voltage = np.abs(V)
        Vangle = np.angle(V)

        loss = np.sum(Z * (Iline ** 2))
        P=np.real(Z*(Iline**2))
        Q=np.imag(Z*(Iline**2))
        print(P,Q)
        #print(np.sum(np.real(Z * (Iline ** 2))))
        return Voltage, loss

    def update(self, data:np.array):
        print(np.sum(self.bus_data))
        return



X = [[0.5,0.5,0.5],[1,1,1],[1,2,3]]


fbsa = FBSA_algorithm(line_data=line_data_default, bus_data=bus_data_default)
#fbsa.update(X)
V, loss = fbsa.run()
plt.plot(V)
plt.show()
print(loss.shape, loss, np.sum(loss)*100000)

