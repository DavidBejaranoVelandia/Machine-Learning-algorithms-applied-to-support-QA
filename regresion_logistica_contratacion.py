import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer, f1_score, recall_score, roc_auc_score, confusion_matrix, precision_score
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt1
import numpy as np
import time
import psutil

start_time = time.time()


#Luego, cargamos los datos de los candidatos desde un archivo CSV utilizando pandas:
data = pd.read_csv("C:/Users/dgonzalez68/OneDrive - DXC Production/Documents/Proyecto de grado/Contratación archivos y programas/datos_contratado_v6.csv")
#Después, dividimos los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split de scikit-learn:
X_train, X_test, y_train, y_test = train_test_split(data.drop('contratado', axis=1), 
                                                    data['contratado'], test_size=0.3, 
                                                    random_state=42)
#En este caso, estamos separando la columna "contratado" del resto de los datos y asignándolos a las variables X e y, respectivamente. Luego, usamos la función train_test_split para dividir los datos en un conjunto de entrenamiento y un conjunto de prueba, y establecemos el tamaño del conjunto de prueba en un 20%.
#A continuación, creamos un modelo de regresión logística:
lr_model = LogisticRegression(max_iter=1000, tol=1e-4, random_state=42)
#En este caso, estamos utilizando la implementación de la regresión logística de scikit-learn. Establecemos el parámetro random_state en 42 para asegurarnos de que los resultados sean reproducibles.
#Luego, ajustamos el modelo a los datos de entrenamiento:
lr_model.fit(X_train, y_train)
#Una vez que el modelo ha sido ajustado, podemos realizar predicciones en los datos de prueba:
lr_predictions = lr_model.predict(X_test)
#Finalmente, podemos evaluar la precisión del modelo utilizando la métrica de precisión accuracy_score de scikit-learn:
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_f1_score = f1_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
precision = precision_score(y_test, lr_predictions)
roc_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
conf_matrix = confusion_matrix(y_test, lr_predictions)

# Código para entrenar el modelo
end_time = time.time()
training_time = end_time - start_time

print("Precisión de regresión logística: ", lr_accuracy)
print("F1-score de regresión logística: ", lr_f1_score)
print("Recall de regresión logística: ", lr_recall)
print("Precisión del regresion: ", precision)
print("Area bajo la curva del regresion: ", roc_auc)
print("Matriz de confusión del regresion: ", conf_matrix)
print("Tiempo de entrenamiento: ", training_time)

def measure_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Convertir de bytes a megabytes
    return memory_usage

# Llamar a la función para medir el uso de memoria
memory_used = measure_memory_usage()
print(f"Uso de memoria: {memory_used} MB")


# Realizamos validación cruzada con un número de pliegues definidos (por ejemplo, 5 pliegues):
num_folds = 5
f1_scores = cross_val_score(lr_model, X_train, y_train, cv=num_folds, scoring='f1')
recall_scores = cross_val_score(lr_model, X_train, y_train, cv=num_folds, scoring='recall')

# Calculamos el promedio de los resultados de F1-score y recall de la validación cruzada:
average_f1_score = f1_scores.mean()
average_recall_score = recall_scores.mean()

# Imprimimos los resultados promediados de la validación cruzada:
print("Promedio del F1-score en validación cruzada: {:.2f}".format(average_f1_score))
print("Promedio del recall en validación cruzada: {:.2f}".format(average_recall_score))
# Supongamos que los nuevos datos se encuentran en la variable "nuevos_datos"
"""
nuevo_candidato = [[19, 1, 1, 0, 0, 3, 0, 3, 1, 0, 2, 3, 3, 2, 2, 1, 0, 4, 3, 3, 1, 1800000, 4, 0, 1, 4, 6, 2, 3
]]


lr_nuevas_predicciones = lr_model.predict(nuevo_candidato)
if lr_nuevas_predicciones == 1:
    print("El candidato es apto.")
else:
    print("El candidato no es apto.")

# lr_nuevas_predicciones contendrá un arreglo con las predicciones para cada registro en los nuevos datos
"""
train_sizes, train_scores, test_scores = learning_curve(lr_model, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar la curva de aprendizaje
plt1.figure(figsize=(8, 6))
plt1.plot(train_sizes, train_mean, label='Train')
plt1.plot(train_sizes, test_mean, label='Cross Validation')

# Graficar las áreas de incertidumbre (± una desviación estándar)
plt1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt1.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

# Configurar etiquetas y título
plt1.xlabel('Training set size')
plt1.ylabel('Score')
plt1.title('Machine Learning: Logistic Regression \n Learning curve')
plt1.legend()
# Mostrar los valores en cada intercepción
for i in range(len(train_sizes)):
    plt1.text(train_sizes[i], train_mean[i], f'{train_mean[i]:.2f}', ha='center', va='bottom')
    plt1.text(train_sizes[i], test_mean[i], f'{test_mean[i]:.2f}', ha='center', va='top')

# Mostrar la gráfica
plt1.show()