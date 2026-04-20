# PRÁCTICA CERO: ANATOMÍA Y ANÁLISIS DE UN DATASET

## 1. Objetivo

Desarrollar criterio analítico para diferenciar entre:

* Obtención pasiva de datos (búsqueda)
* Creación activa de datos (instrumentación)
* Análisis Exploratorio de Datos (EDA)

---

## 2. Fase 1: Análisis de Casos

### Escenario A: Búsqueda e Integración

En este escenario, los datos ya existen y provienen de fuentes externas.

**Características:**

* No hay control sobre cómo se generaron los datos
* Se requiere integrar múltiples fuentes
* Problemas comunes: valores faltantes, diferentes formatos y frecuencias

**Reto principal:**

* Limpieza de datos
* Alineación temporal
* Manejo de inconsistencias

---

### Escenario B: Creación e Instrumentación

En este caso, los datos se generan desde cero.

**Características:**

* Control total sobre el proceso de medición
* Uso de sensores físicos
* Problemas reales como ruido o errores de medición

**Reto principal:**

* Selección de sensores
* Definición de frecuencia de muestreo
* Garantizar calidad del dato

---

### Comparación

| Aspecto            | Escenario A          | Escenario B                    |
| ------------------ | -------------------- | ------------------------------ |
| Origen de datos    | Externo              | Propio                         |
| Control            | Bajo                 | Alto                           |
| Problema principal | Limpieza             | Adquisición                    |
| Riesgo             | Datos inconsistentes | Datos incorrectos desde origen |

---

## 3. Fase 2: Análisis Exploratorio de Datos (EDA)

### Ejecución del código

Se ejecutó el script proporcionado para generar una serie de tiempo simulada con tendencia, oscilación y ruido.

### Observaciones de la serie de tiempo

La señal generada presenta:

* Una tendencia creciente (de aproximadamente 10 a 50)
* Oscilaciones periódicas tipo seno
* Ruido aleatorio superpuesto

Esto indica que la serie no es estacionaria, ya que su media cambia con el tiempo.

---

### Observaciones del histograma

* La distribución es aproximadamente normal
* El ruido sigue un comportamiento gaussiano
* La mayor concentración de valores se encuentra en el rango medio

---

### Estadística descriptiva

El análisis estadístico muestra:

* Media aproximada en el rango medio de los datos
* Desviación estándar que refleja la variabilidad introducida por el ruido
* Valores mínimos y máximos que corresponden a fluctuaciones normales

---

## 3.1 Inyección de anomalías

Se modificó el código original para introducir valores atípicos y datos faltantes:

```python
valores[100] = 120
valores[250] = -30
valores[300:310] = np.nan
```

### Observaciones tras la modificación

* Se observaron picos abruptos en la serie de tiempo
* El histograma presentó colas más largas, indicando mayor dispersión
* La desviación estándar aumentó significativamente
* Los valores NaN generaron interrupciones en la continuidad de la señal

Esto demuestra cómo pequeñas alteraciones pueden afectar significativamente el comportamiento del dataset.

---

## 4. Diferencias entre CNN y RNN

### CNN (Convolutional Neural Networks)

* Diseñadas para datos espaciales (principalmente imágenes)
* Utilizan filtros convolucionales para detectar patrones locales
* Son eficientes en la extracción de características
* No consideran dependencias temporales

### RNN (Recurrent Neural Networks)

* Diseñadas para datos secuenciales (series de tiempo, texto)
* Mantienen memoria de estados anteriores
* Capturan dependencias temporales
* Pueden presentar problemas como desvanecimiento del gradiente

### Comparación

| Característica    | CNN                           | RNN                                |
| ----------------- | ----------------------------- | ---------------------------------- |
| Tipo de datos     | Espacial                      | Secuencial                         |
| Memoria           | No                            | Sí                                 |
| Uso común         | Imágenes                      | Series de tiempo, NLP              |
| Ventaja principal | Extracción de características | Captura de dependencias temporales |

---

## 5. Fase 3: Conclusiones Críticas

### 1. ¿La serie es estacionaria?

No, la serie no es estacionaria porque su media y varianza cambian con el tiempo debido a la tendencia creciente y las oscilaciones presentes en la señal.

Esto es un factor crítico porque muchos modelos predictivos, especialmente en series de tiempo, asumen estacionariedad. Si esta condición no se cumple, el modelo puede aprender patrones incorrectos y generar predicciones poco confiables.

---

### 2. Factores físicos que generan outliers

En un entorno real, si el dataset fuera capturado mediante instrumentación, los outliers podrían originarse por:

* Ruido electromagnético generado por maquina
