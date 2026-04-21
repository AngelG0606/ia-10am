# Actividad 2: Anatomía de una Vanilla RNN (Explicación simple y amplia)

---

## 2.1 Mapeo de Variables

Ecuación base:

\[
h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b)
\]

Cada componente representa lo siguiente:

- **\(x_t\)** (entrada actual)  
  Es la información nueva que llega en el tiempo actual.  
  Frase: “Soy la novedad pura, el pulso del instante...”

- **\(h_{t-1}\)** (estado anterior)  
  Es la memoria acumulada del pasado.  
  Frase: “el fantasma del pasado...”

- **\(W_{hx}, W_{hh}\)** (pesos)  
  Son matrices que transforman y ajustan la importancia de la información.  
  Frase: “peajes inmutables...”

- **\(b\)** (bias)  
  Es un ajuste adicional que permite desplazar el resultado.  
  Frase: “un pequeño desvío inevitable”

- **\(\tanh\)** (función de activación)  
  Limita los valores entre -1 y 1 para evitar explosiones numéricas.  
  Frase: “muro curvo que nos comprime entre el -1 y el 1”

- **\(h_t\)** (estado actual)  
  Es el resultado final en ese instante, que también será memoria futura.  
  Frase: “nazco yo, una nueva identidad...”

---

## 2.2 Análisis de Dimensionalidad

Datos:
- \(x_t \in \mathbb{R}^{20}\)
- \(h_{t-1} \in \mathbb{R}^{64}\)

### 1. Dimensión de \(W_{hx}\)

Debe transformar un vector de tamaño 20 en uno de tamaño 64:

\[
W_{hx} \in \mathbb{R}^{64 \times 20}
\]

Explicación: multiplicas una matriz por un vector de 20 elementos y obtienes uno de 64.

---

### 2. Dimensión de \(W_{hh}\)

Debe transformar un vector de tamaño 64 en otro de tamaño 64:

\[
W_{hh} \in \mathbb{R}^{64 \times 64}
\]

Explicación: mantiene la dimensión del estado oculto.

---

### 3. Dimensión de \(h_t\)

El resultado final tiene dimensión:

\[
h_t \in \mathbb{R}^{64}
\]

---

## 2.3 Estrofa del bias

Soy el leve ajuste que inclina la decisión,  
no cambio la forma, pero sí la dirección,  
evito que todo dependa del origen,  
y doy flexibilidad a cada reacción.

Explicación: el bias permite que la red no esté obligada a pasar por el origen, lo que mejora su capacidad de aprendizaje.

---

## 2.4 Saturación de la función tanh

1. La función \(\tanh(z)\) tiene forma de "S" y sus valores van de -1 a 1.  
   Su derivada es:

\[
f'(z) = 1 - \tanh^2(z)
\]

---

2. Si \(z = 500\):

\[
\tanh(500) \approx 1
\]

\[
f'(500) = 1 - 1^2 = 0
\]

---

3. Problema de saturación

Cuando la derivada es cercana a 0:

- El gradiente desaparece
- Los pesos dejan de actualizarse
- La red deja de aprender

Esto se conoce como "vanishing gradient".

---

## 2.5 Trazo del Gradiente (BPTT)

Cuando la red comete un error, este se propaga hacia atrás en el tiempo.

El error pasa por:

- La función \(\tanh\) (muro curvo)
- Las matrices \(W_{hh}\) (peajes)

Matemáticamente:

\[
\frac{\partial L}{\partial h_{t-1}} =
\frac{\partial L}{\partial h_t} \cdot
W_{hh}^T \cdot
(1 - \tanh^2(z))
\]

Explicación simple:

- El error se va multiplicando paso a paso hacia atrás
- Si los valores son pequeños, el error desaparece
- Si son grandes, puede explotar

Esto sigue la regla de la cadena del cálculo diferencial.

---

## 2.6 Depuración del código NumPy

### 1. Error

El operador `*` en NumPy realiza multiplicación elemento a elemento, no multiplicación matricial.

Esto provoca errores matemáticos en redes neuronales.

---

### 2. Código corregido

```python
import numpy as np

def paso_rnn(x_t, h_prev, W_hx, W_hh, b):
    combinacion = np.dot(W_hx, x_t) + np.dot(W_hh, h_prev) + b
    return np.tanh(combinacion)