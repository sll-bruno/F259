import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def pendulo_simples(x):
    return 2*pi*np.sqrt(x/9.51)

def pendulo_físico(x):
    return 2*pi*np.sqrt((x+(0.062**2)/x)/9.78)


#Pandas
data = pd.read_csv("Experimento 2 - Pêndulo Físico\Dados experimento 2 - períodos.csv")
data = data.set_index("Testes")

inc_periodo = data["incerteza - período"].str.replace(",",".").apply(lambda x: float(x))
incerteza_distancia = 0.00002*np.ones_like(data['distância'])
data["distância"] = data["distância"].str.replace(",",".").apply(lambda x: float(x)) #Eixo X
data["período"] = data["período"].str.replace(",",".").apply(lambda x: float(x))  #Eixo Y

#Matplotlib 
plt.rcParams['figure.figsize'] = [10, 6]
plt.errorbar(data["distância"],data["período"],yerr = inc_periodo,xerr=incerteza_distancia,fmt ="o", label = "Experimental Data", color = "black")
plt.plot(np.linspace(0,1,500),pendulo_simples(np.linspace(0,1,500)), label = "Pêndulo Símples")
plt.plot(np.linspace(0,1,500),pendulo_físico(np.linspace(0,1,500)),label = "Pêndulo Físico")
plt.legend(loc = "upper right", fontsize = "small", shadow = True)
plt.xlabel("D (m)")
plt.ylabel("T(s)")
plt.title("Gráfico de T (s) x D (m)") 
plt.xlim(0,0.25)
plt.ylim(0,1.5)

plt.show()
