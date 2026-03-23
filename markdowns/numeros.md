# Análisis Exploratorio de Datos (EDA)
## Identificación de los números 1, 2 y 3 mediante imágenes

### 1. Objetivo
El objetivo de este análisis es estudiar un conjunto de imágenes que contienen los números 1, 2 y 3, para entender cómo son, qué diferencias tienen y qué dificultades existen al intentar que una computadora los reconozca automáticamente.

---

### 2. Contexto
El reconocimiento de números es un problema común en la inteligencia artificial. En este caso, se busca que un sistema pueda identificar correctamente los números 1, 2 y 3, aunque estén escritos de muchas formas diferentes.

Las variaciones pueden ser:
- Diferente forma de escribir
- Líneas más gruesas o delgadas
- Inclinación
- Ruido en la imagen
- Diferente tamaño

---

### 3. Descripción del dataset

#### 3.1 Tipo de datos
- Imágenes en blanco y negro (escala de grises)
- Tamaño común: 28x28 píxeles
- Etiquetas: 1, 2 o 3

#### 3.2 Estructura
Cada dato contiene:
- Una imagen representada como números (píxeles)
- Una etiqueta que indica qué número es

---

### 4. Ejemplo de dataset (tabla)

| ID | Pixel_1 | Pixel_2 | ... | Pixel_784 | Label |
|----|--------|--------|-----|-----------|-------|
| 1  | 0      | 0      | ... | 255       | 1     |
| 2  | 12     | 45     | ... | 200       | 2     |
| 3  | 0      | 0      | ... | 180       | 3     |
| 4  | 34     | 60     | ... | 210       | 1     |
| 5  | 0      | 10     | ... | 190       | 2     |

---

### 5. Análisis exploratorio

#### 5.1 Cantidad de datos por número
Es importante revisar cuántas imágenes hay de cada número:
- Número 1: N1 imágenes
- Número 2: N2 imágenes
- Número 3: N3 imágenes

Si hay más imágenes de un número que de otros, el modelo puede aprender más ese número.

---

#### 5.2 Observación de las imágenes
Al ver las imágenes se puede notar:
- Hay muchas formas de escribir el mismo número
- Algunas imágenes son más claras que otras
- Algunas pueden ser difíciles de identificar

---

#### 5.3 Diferencias dentro de cada número

Cada número puede verse diferente dependiendo de quién lo escribió:

- Número 1:
  - Puede ser una línea recta
  - Puede tener una pequeña base o inclinación

- Número 2:
  - Puede ser más curvo o más recto
  - Puede variar en tamaño

- Número 3:
  - Puede tener curvas más abiertas o cerradas

---

#### 5.4 Análisis de píxeles

Cada imagen está formada por píxeles:
- Valores cercanos a 0: fondo (negro)
- Valores cercanos a 255: trazo (blanco)

Esto ayuda a identificar la forma del número.

---

#### 5.5 Calidad de las imágenes

Se pueden encontrar problemas como:
- Ruido (puntos que no deberían estar)
- Imágenes borrosas
- Diferente contraste

---

### 6. Problemas encontrados

- Muchas formas diferentes de escribir el mismo número
- Algunas imágenes pueden confundirse
- Ruido en los datos
- Posible desbalance en las clases

---

### 7. Preparación de los datos

Antes de usar los datos, se recomienda:

- Normalizar los valores (de 0 a 1)
- Asegurar que todas las imágenes tengan el mismo tamaño
- Limpiar ruido
- Crear más datos artificiales (data augmentation):
  - Girar imágenes
  - Moverlas un poco
  - Cambiar tamaño

---

### 8. Métodos para resolver el problema

Se pueden usar diferentes técnicas:

- Métodos básicos:
  - K vecinos más cercanos (KNN)
  - Máquinas de soporte vectorial (SVM)

- Métodos más avanzados:
  - Redes neuronales convolucionales (CNN)

Las CNN son las más recomendadas para imágenes.

---

### 9. Resultados importantes

- Los datos tienen mucha variedad
- Se pueden encontrar patrones en cada número
- La calidad de los datos es muy importante
- Un buen preprocesamiento mejora los resultados

---

### 10. Conclusión

Identificar los números 1, 2 y 3 usando imágenes es posible, pero no es tan fácil debido a las muchas formas en que pueden escribirse. Es importante analizar bien los datos antes de crear un modelo.

---

### 11. Recomendaciones

- Revisar que las clases estén balanceadas
- Mejorar la calidad de las imágenes
- Usar técnicas de aumento de datos
- Utilizar redes neuronales para mejores resultados
