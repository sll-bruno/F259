import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def cosseno(x):
    return 15*(np.cos(8.51*x)) - 90 

data = pd.read_csv(r"Experimento 2 - Pêndulo Físico\Dados experimento 2 - angulação- 9cm.csv")

plt.rcParams['figure.figsize'] = [10, 6]
data["angulo"] = data["angulo"].str.replace(",",".").apply(lambda x: float(x)) 
data["angulo"] = np.degrees(data["angulo"])
data["angulo"] = data["angulo"]
data["tempo"] = data["tempo"].str.replace(",",".").apply(lambda x: float(x)) 
plt.plot(data["tempo"],data["angulo"],"o",markersize = 2, label = "Experimental Data", color = "black")
plt.plot(data["tempo"],cosseno(data["tempo"]),label = "θ(t) = θmaxcos(ωt)")
plt.xlabel("tempo (s)")
plt.ylabel("θ(°)")
plt.legend(loc = "upper right", fontsize = "small", shadow = True)
plt.show()