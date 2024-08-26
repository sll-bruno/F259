import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Pandas
data = pd.read_csv("dados.csv")
data = data.set_index("Testes")

data['deltaM'] = (data["m1"] - data["m2"]).apply(lambda x: x/1000 )
data_cleaned = data.dropna() #axis = 0  by defalut -  Significa que a função remove todas as linhas que contem NaN values

aceleration = data_cleaned["aceleracao media"] #Eixo X
aceleration = aceleration.str.replace(",",".")
aceleration = aceleration.apply(lambda x: round(float(x),2))

deltaM = data_cleaned["deltaM"] # Eixo Y

#Matplotlib 
plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(aceleration,deltaM,"o",label = "Experimental Data", color = "black")
plt.xlabel("aceleração (m/s²)")
plt.ylabel("m1 - m2 (kg)")
plt.title("Gráfico de Δm (Kg) x Aceleração (m/s²)") 
plt.xlim(left = 0.35)
plt.ylim(bottom = 0.1)

#sklearn - Gera a Regressão Linear associada aos dados de aceleração de Diferença de massa obtidos
x = pd.DataFrame(aceleration)
y = deltaM
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 1)
model = LinearRegression()
model.fit(x_train,y_train)
x_range = np.linspace(aceleration.min(),aceleration.max(),100).reshape(-1,1)
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
incerteza[0] = round(incerteza[0],3)
incerteza[1] = round(incerteza[1],3)

print(f"Δm = ({a:.3f}±{incerteza[0]})*Aceleração + ({b:.3f}±{incerteza[1]:.3f})")

plt.show()
