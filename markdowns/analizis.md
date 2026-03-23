# Análisis de Rendimiento del Modelo CNN en Clasificación de Imágenes

## Descripción general

Se implementó una Red Neuronal Convolucional (CNN) para la clasificación de imágenes en dos clases.
El objetivo de este análisis es evaluar el comportamiento del modelo a lo largo de múltiples ejecuciones, enfocándose en métricas como **precision**, **recall** y **accuracy**, con especial atención al desempeño por clase.

---

## Resultados de las ejecuciones

| Prueba | Precision (C0) | Recall (C0) | Precision (C1) | Recall (C1) | Accuracy |
| ------ | -------------- | ----------- | -------------- | ----------- | -------- |
| 1      | 0.90           | 0.76        | 0.96           | 0.98        | 0.95     |
| 2      | 0.91           | 0.83        | 0.97           | 0.98        | 0.96     |
| 3      | 0.91           | 0.80        | 0.96           | 0.99        | 0.96     |
| 4      | 0.90           | 0.78        | 0.96           | 0.98        | 0.95     |
| 5      | 0.92           | 0.84        | 0.97           | 0.99        | 0.96     |
| 6      | 0.91           | 0.81        | 0.96           | 0.98        | 0.96     |
| 7      | 0.90           | 0.79        | 0.96           | 0.98        | 0.95     |
| 8      | 0.92           | 0.85        | 0.97           | 0.99        | 0.96     |
| 9      | 0.91           | 0.82        | 0.96           | 0.98        | 0.96     |
| 10     | 0.92           | 0.84        | 0.97           | 0.99        | 0.96     |

---

## Análisis del comportamiento del modelo

### Accuracy general

El modelo presenta una **accuracy entre 0.95 y 0.96**, lo cual indica un desempeño alto y consistente a lo largo de las ejecuciones.
Las variaciones son mínimas, lo que sugiere estabilidad en el entrenamiento.

---

## Análisis por clase

### Clase 1

* Precision: entre 0.96 y 0.97
* Recall: entre 0.98 y 0.99

Interpretación:

El modelo tiene un desempeño excelente en esta clase.
El recall cercano a 1 indica que casi todos los ejemplos reales son correctamente identificados.
Esto sugiere que la red ha aprendido de forma efectiva los patrones asociados a esta categoría.

---

### Clase 0

* Precision: entre 0.90 y 0.92
* Recall: entre 0.76 y 0.85

Interpretación:

El rendimiento es aceptable pero inferior al de la Clase 1.
El recall más bajo indica que el modelo no logra identificar correctamente todos los ejemplos de esta clase, generando falsos negativos.

---

## Análisis del recall

El recall permite observar la capacidad del modelo para detectar correctamente cada clase:

* La Clase 1 mantiene un recall muy alto en todas las ejecuciones.
* La Clase 0 presenta variabilidad y valores más bajos.

Esto indica que el modelo tiene mayor facilidad para reconocer una clase sobre la otra.

---

## Problema identificado: desbalance de clases

El conjunto de datos presenta una diferencia notable en el número de muestras:

* Clase 0: menor cantidad de ejemplos
* Clase 1: mayor cantidad de ejemplos

Este desbalance provoca que:

* El modelo favorezca la clase mayoritaria
* Se reduzca la capacidad de detección de la clase minoritaria
* El recall de la Clase 0 sea menor

---

## Variabilidad entre ejecuciones

Se observa que, aunque el modelo es estable, existen pequeñas variaciones en los resultados debido a:

* Inicialización aleatoria de pesos
* División aleatoria de los datos (train/test)
* Proceso estocástico del entrenamiento

Estas variaciones son normales en modelos de aprendizaje profundo.

---

## Conclusiones

El modelo CNN presenta un desempeño general alto, con valores de accuracy superiores al 95%.
Sin embargo, el análisis del recall evidencia una diferencia importante entre clases.

La Clase 1 es identificada con gran precisión, mientras que la Clase 0 presenta menor capacidad de detección, debido principalmente al desbalance en los datos.

Para mejorar el rendimiento del modelo, especialmente en la Clase 0, se recomienda aplicar técnicas como balanceo de datos, uso de pesos por clase o aumento de datos.

---
