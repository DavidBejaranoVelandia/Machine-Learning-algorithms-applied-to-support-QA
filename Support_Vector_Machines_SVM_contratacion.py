import pandas as pd
import sklearn
import matplotlib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, make_scorer, f1_score, recall_score, roc_auc_score, confusion_matrix, precision_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt1
import numpy as np
import time
import psutil

start_time = time.time()

datos = pd.read_csv("C:/Users/dgonzalez68/OneDrive - DXC Production/Documents/Proyecto de grado/Contratación archivos y programas/datos_contratado_v5.csv")

# Supongamos que los datos de entrenamiento se encuentran en la variable "datos"
X = datos.drop('contratado', axis=1) # variables independientes
y = datos['contratado'] # variable dependiente

# Dividimos los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creamos una instancia de un modelo de SVM
svm_model = SVC(kernel='linear', C=0.1, random_state=42)

# Entrenamos el modelo con los datos de entrenamiento
svm_model.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
svm_predicciones = svm_model.predict(X_test)

# Evaluamos el desempeño del modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, svm_predicciones)
conf_matrix = confusion_matrix(y_test, svm_predicciones)

svm_f1_score = f1_score(y_test, svm_predicciones)
svm_recall = recall_score(y_test, svm_predicciones)

# Realizamos validación cruzada para calcular el F1-score y recall
f1_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='f1')
recall_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='recall')
precision = precision_score(y_test, svm_predicciones)
roc_auc = roc_auc_score(y_test, svm_predicciones)

# Código para entrenar el modelo
end_time = time.time()
training_time = end_time - start_time

print("Exactitud del modelo: {:.2f}%".format(accuracy * 100))
print("Matriz de confusión:")
print(conf_matrix)
print("F1-score del modelo SVM: {:.2f}".format(svm_f1_score))
print("Recall del modelo SVM: {:.2f}".format(svm_recall))
# Imprimimos los resultados promediados de la validación cruzada
print("Promedio del F1-score en validación cruzada: {:.2f}".format(f1_scores.mean()))
print("Promedio del recall en validación cruzada: {:.2f}".format(recall_scores.mean()))
print("Precisión del SVM: ", precision)
print("Area bajo la curva del SVM: ", roc_auc)
print("Tiempo de entrenamiento: ", training_time)

def measure_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Convertir de bytes a megabytes
    return memory_usage

# Llamar a la función para medir el uso de memoria
memory_used = measure_memory_usage()
print(f"Uso de memoria: {memory_used} MB")

"""
# Supongamos que el modelo entrenado se encuentra en la variable "svm_model"
nuevo_candidato = [[23, 1, 0, 0, 0, 2, 0, 2, 0, 0, 4, 3, 3, 1, 4, 1, 0, 2, 4, 4, 1, 1300000, 4, 1, 1, 4, 5, 4, 5
]]
svm_prediccion = svm_model.predict(nuevo_candidato)

# La variable svm_prediccion contendrá el resultado de la predicción
if svm_prediccion == 1:
    print("El candidato es apto.")
else:
    print("El candidato no es apto.")

"""
train_sizes, train_scores, test_scores = learning_curve(svm_model, X_train, y_train, cv=5)

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
plt1.title('Machine Learning: Support Vector Machines (SVM) \n Learning curve')
plt1.legend()
# Mostrar los valores en cada intercepción
for i in range(len(train_sizes)):
    plt1.text(train_sizes[i], train_mean[i], f'{train_mean[i]:.2f}', ha='center', va='bottom')
    plt1.text(train_sizes[i], test_mean[i], f'{test_mean[i]:.2f}', ha='center', va='top')

# Mostrar la gráfica
plt1.show()
