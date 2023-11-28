import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt1
from sklearn.metrics import f1_score, recall_score
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
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
#A continuación, creamos un modelo de Árbol de decisión:
tree_model = DecisionTreeClassifier(random_state=42)
#En este caso, estamos utilizando la implementación de Árboles de decisión de scikit-learn con un parámetro random_state de 42 para asegurar que los resultados sean reproducibles.
#Luego, ajustamos el modelo a los datos de entrenamiento:
tree_model.fit(X_train, y_train)

# Mostrar el gráfico del árbol de decisión
plt.show()
#Una vez que el modelo ha sido ajustado, podemos realizar predicciones en los datos de prueba:
tree_predictions = tree_model.predict(X_test)
#Finalmente, podemos evaluar la precisión del modelo utilizando la métrica de precisión accuracy_score de scikit-learn:
tree_accuracy = accuracy_score(y_test, tree_predictions)
tree_f1_score = f1_score(y_test, tree_predictions)
tree_recall = recall_score(y_test, tree_predictions)
tree_precision = precision_score(y_test, tree_predictions)
tree_roc_auc = roc_auc_score(y_test, tree_predictions)
tree_conf_matrix = confusion_matrix(y_test, tree_predictions)

# Código para entrenar el modelo
end_time = time.time()
training_time = end_time - start_time
print("Accuracy del árbol de decisiones: ", tree_accuracy)
print("F1-score del árbol de decisiones: ", tree_f1_score)
print("Recall del árbol de decisiones: ", tree_recall)
print("Precisión del árbol de decisiones: ", tree_precision)
print("Area bajo la curva del árbol de decisiones: ", tree_roc_auc)
print("Matriz de confusión del árbol de decisiones: ", tree_conf_matrix)
print("Tiempo de entrenamiento: ", training_time)

def measure_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Convertir de bytes a megabytes
    return memory_usage

# Llamar a la función para medir el uso de memoria
memory_used = measure_memory_usage()
print(f"Uso de memoria: {memory_used} MB")

#coring = {'f1': make_scorer(f1_score), 'recall': make_scorer(recall_score)}

# Realizamos la validación cruzada con un número de pliegues definidos (por ejemplo, 5 pliegues):
num_folds = 5
f1_scores = cross_val_score(tree_model, X_train, y_train, cv=num_folds, scoring='f1')
recall_scores = cross_val_score(tree_model, X_train, y_train, cv=num_folds, scoring='recall')

# Calculamos el promedio de los resultados de F1-score y recall de la validación cruzada:
average_f1_score = f1_scores.mean()
average_recall_score = recall_scores.mean()



# Imprimimos los resultados promediados de la validación cruzada:
print("Promedio del F1-score en validación cruzada: {:.2f}".format(average_f1_score))
print("Promedio del recall en validación cruzada: {:.2f}".format(average_recall_score))


"""
# Recopilar información sobre el nuevo candidato
nuevo_candidato = [[23, 1, 0, 0, 0, 2, 0, 2, 0, 0, 4, 3, 3, 1, 4, 1, 0, 2, 4, 4, 1, 1300000, 4, 1, 1, 4, 5, 4, 5
]] # Características del nuevo candidato

# Realizar la predicción
prediccion = tree_model.predict(nuevo_candidato)

# La variable "prediccion" contendrá la predicción binaria (0 o 1)
if prediccion == 1:
    print("El candidato es apto.")
else:
    print("El candidato no es apto.")

#    warnings.simplefilter("ignore") 
'''
'''
"""
classes = tree_model.classes_

print(classes)

#--- Grafica

# Visualizar el árbol de decisión
fig, ax = plt.subplots(figsize=(10, 10))

tree.plot_tree(tree_model, filled=True, feature_names=["edad", "vive bogota", "experiencia laboral en testing", 
                                                       "trabaja actualmente", "semanas para entregar", "conocimiento tipo de pruebas", 
                                                       "graduado profesional", "titulo de grado", "titulo de sistemas", 
                                                       "esta estudiando", "conocimiento pruebas de regresion", 
                                                       "conocimiento creacion de casos de prueba", "conocimiento creacion de defectos", 
                                                       "lenguajes de programacion", "dominio lenguajes que conoce", 
                                                       "conoce automatizacion", "experiencia automatizacion", 
                                                       "conoce metodologias agiles", "calidad fortalezas", 
                                                       "calidad oportunidades de mejora", "trabajando oportunidades de mejora", 
                                                       "aspiracion salarial", "motivacion trabajar pruebas", 
                                                       "recomendado por alguien", "horario extendido", "primer ejercicio", 
                                                       "cuantas ambiguedades", "gesticular", "seguridad"
]
, class_names=["No contratado", "Contratado"], ax=ax)
"""
tree.plot_tree(tree_model, filled=True, feature_names=["edad", "experiencia laboral en testing", "trabaja actualmente", "conocimiento tipo de pruebas", "titulo de grado", "titulo de sistemas", "esta estudiando", "conocimiento pruebas de regresion", "conocimiento creacion de casos de prueba", "conocimiento creacion de defectos", "dominio lenguajes que conoce", "experiencia automatizacion", "conoce metodologias agiles", "calidad fortalezas", "calidad oportunidades de mejora", "trabajando oportunidades de mejora", "aspiracion salarial", "motivacion trabajar pruebas", "primer ejercicio", "cuantas ambiguedades", "seguridad"
]
, class_names=["No contratado", "Contratado"], ax=ax)
"""
# Mostrar el gráfico del árbol de decisión
#plt.show()
#----Otra grafica
# Supongamos que tienes un modelo clf y datos X, y
train_sizes, train_scores, test_scores = learning_curve(tree_model, X_train, y_train, cv=5)

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
plt1.title('Machine Learning: Decision Tree Algorithm \n Learning curve')
plt1.legend()
# Mostrar los valores en cada intercepción
for i in range(len(train_sizes)):
    plt1.text(train_sizes[i], train_mean[i], f'{train_mean[i]:.2f}', ha='center', va='bottom')
    plt1.text(train_sizes[i], test_mean[i], f'{test_mean[i]:.2f}', ha='center', va='top')

# Mostrar la gráfica
plt1.show()
