# Análisis Exploratorio de Datos (EDA)
## Juego de un muñeco que esquiva una bala

### 1. Objetivo
El objetivo de este análisis es estudiar los datos de un juego donde un personaje (muñeco) debe esquivar una bala que se mueve a diferentes velocidades y alturas. Se busca entender cómo influyen estas variables en el resultado del juego (esquivar o recibir daño).

---

### 2. Contexto
En este juego, el jugador controla a un muñeco que puede realizar dos acciones:
- Saltar
- Agacharse

La bala se mueve hacia el personaje con diferentes características:
- Velocidad variable
- Altura variable

El reto consiste en elegir la acción correcta en el momento adecuado para evitar la colisión.

---

### 3. Descripción del dataset

#### 3.1 Tipo de datos
- Datos numéricos y categóricos
- Cada registro representa un intento del juego

#### 3.2 Variables principales

- Velocidad_bala: rapidez con la que se mueve la bala
- Altura_bala: posición vertical de la bala
- Acción: lo que hace el jugador (saltar o agacharse)
- Resultado: éxito (esquiva) o fallo (impacto)

---

---

### 5. Análisis exploratorio

#### 5.1 Distribución de velocidades
Se analiza qué tan rápidas son las balas:
- Velocidades bajas: más fáciles de esquivar
- Velocidades altas: más difíciles

---

#### 5.2 Distribución de alturas
Las balas pueden aparecer en diferentes posiciones:
- Baja
- Media
- Alta

Esto influye directamente en la acción correcta.

---

#### 5.3 Relación entre altura y acción

- Bala baja → saltar es más efectivo
- Bala alta → agacharse es más efectivo
- Bala media → puede generar confusión

---

#### 5.4 Relación entre velocidad y resultado

- A mayor velocidad, menor tiempo de reacción
- A menor velocidad, mayor probabilidad de éxito

---

#### 5.5 Análisis de resultados

Se observa:
- Cuántos intentos fueron exitosos
- Cuántos fallaron

Esto ayuda a entender la dificultad del juego.

---

#### 5.6 Combinación de variables

Se analizan combinaciones como:
- Alta velocidad + altura media → mayor dificultad
- Baja velocidad + altura clara → menor dificultad

---

### 6. Problemas encontrados

- Velocidades muy altas pueden hacer el juego injusto
- Alturas medias pueden ser difíciles de interpretar
- El tiempo de reacción puede ser muy corto
- Posibles decisiones incorrectas del jugador

---

### 7. Preparación de los datos

Para mejorar el análisis se puede:

- Convertir "Acción" en valores numéricos
- Convertir "Resultado" en éxito (1) o fallo (0)
- Normalizar la velocidad
- Clasificar mejor las alturas

---

### 8. Posibles modelos

Se pueden usar modelos para predecir la mejor acción:

- Árboles de decisión
- Regresión logística
- Redes neuronales simples

---

### 9. Hallazgos importantes

- La altura de la bala es clave para decidir la acción
- La velocidad afecta el tiempo de respuesta
- Existen combinaciones más difíciles que otras
- El jugador puede mejorar con práctica

---

### 10. Conclusión

El análisis muestra que el juego depende principalmente de la altura y la velocidad de la bala. Para tener éxito, el jugador debe reaccionar correctamente según estas variables. Un buen equilibrio en la dificultad es importante para que el juego sea divertido y justo.

---

### 11. Recomendaciones

- Ajustar las velocidades para evitar dificultad extrema
- Definir claramente las alturas
- Dar tiempo suficiente de reacción al jugador
- Usar los datos para mejorar la jugabilidad
