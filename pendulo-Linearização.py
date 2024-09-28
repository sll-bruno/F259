import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#Pandas
data = pd.read_csv("Experimento 2 - Pêndulo Físico\Dados experimento 2 - períodos.csv")
data = data.set_index("Testes")

data["distância"] = data["distância"].str.replace(",",".").apply(lambda x: float(x))
data["período"] = data["período"].str.replace(",",".").apply(lambda x: float(x))
data['D²'] = data["distância"]**2 #Eixo X
data["T²D"] = data["período"] * data["período"] * data["distância"] #Eixo Y

#Matplotlib 
plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(data["D²"],data["T²D"],"o", label = "Experimental Data", color = "black")
plt.xlabel("D² (m²)")
plt.ylabel("T²D (s²m)")
plt.title("Gráfico de T²D (s²m) x D² (m²)") 
 #Inserir plt.xlim(left = 0.35)
#Inserir plt.ylim(bottom = 0.1)

#sklearn - Gera a Regressão Linear associada aos dados de aceleração de Diferença de massa obtidos
x = pd.DataFrame(data["D²"])
y = data["T²D"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 1)
model = LinearRegression()
model.fit(x_train,y_train)
x_range = np.linspace(x.min(),x.max(),100).reshape(-1,1)
y_pred = model.predict(x_range)

#Adiciona linearização ao gráfico
plt.plot(x_range,y_pred,label = "Linear Fit", color= "blue")
plt.legend(loc = "upper left", fontsize = "small", shadow = True)

#Cálcula as incertezas associadas a cada parâmetro, retornadas na variável Incertezas.
y_test_pred = model.predict(x_test)
residuals = y_test - y_test_pred
n = len(y)
p = x.shape[1] + 1
residual_variance = np.sum(residuals**2) / (n - p)
x_design = np.hstack([np.ones((x.shape[0], 1)), x])
cov_matrix = residual_variance * np.linalg.inv(x_design.T @ x_design)
incerteza = np.sqrt(np.diag(cov_matrix))

a = model.coef_[0]
b = model.intercept_

incerteza[0] = round(incerteza[0],5)
incerteza[1] = round(incerteza[1],5)

print(f"Δm = ({a:.3f}±{incerteza[0]})*Aceleração + ({b:.3f}±{incerteza[1]})")

plt.show()



