
import pandas as pd

# Importamos el modelo de arbol de decision
from sklearn.tree import DecisionTreeClassifier

# Importamos una funcion para separar los datos
# en datos de entrenamiento y en datos de test
from sklearn.model_selection import train_test_split

# importamos una funcion para medir como de bien predice
from sklearn.metrics import accuracy_score

# Leemos los datos de entrada
df = pd.read_csv("music.csv")

# las columnas de entrada al modelo, en este caso
# son la edad y si es hombre o mujer
entrada = df.drop(columns=['genre'])

# La salida que es el tipo de musica que le gusta
salida = df.drop(columns=['age', 'gender'])

# Creamos un modelo de clasificacion de arboles
model = DecisionTreeClassifier()

# entrenamos el modelo
model.fit(entrada, salida)

# predecimos para un valor en concreto: un hombre de 20 anios. En
# concreto para este valor no tenemos una repuesta pero gracias al 
# arbol predice el genero de musica que le puede gustar
genero_predecido = model.predict([[21, 1]])[0]
print(genero_predecido)

# Ahora predecir para una mujer de 22 anios 
genero_predecido = model.predict([[22, 0]])[0]
print(genero_predecido)

# Utilizamos de los datos que tenemos el 20% para entrenar
entrada_entrenar, entrada_test, salida_entrenar, salida_test = train_test_split(entrada, salida, test_size=0.2)

# Al separar los datos en una parte para entrenar el modelo
# y otra parte para testear el resultado de nuestro modelo
# reentrenamos el modelo con ese 80% de los datos
model = DecisionTreeClassifier()
model.fit(entrada_entrenar, salida_entrenar)

# Hacemos una prediccion para los datos de entrada que
# hemos reservado para hacer el test
predicciones = model.predict(entrada_test)

# Medimos como de bien ha predicho nuestro algoritmo
score = accuracy_score(salida_test, predicciones)
print(score)


