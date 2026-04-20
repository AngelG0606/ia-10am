# RESULTADO Y ANÁLISIS DE LA ACTIVIDAD RNN BÁSICA VANILLA

**Eduardo Alcaraz**

---

## Suposición del Modelo

Para resolver la actividad se asume que la RNN conserva el **50% del estado anterior**, es decir:

```
h_t = x_t + 0.5 * h_{t-1}
```

Esto significa que cada día la memoria del estado previo se reduce a la mitad, lo que permite observar el efecto de **desvanecimiento de la memoria**.

---

## Misión 1: El Lunes Increíble

**Objetivo:** Analizar cómo un evento fuerte pierde impacto con el tiempo.

### Cálculo paso a paso

* Día 1:
  h₁ = 10

* Día 2:
  h₂ = 0 + 0.5(10) = 5

* Día 3:
  h₃ = 0 + 0.5(5) = 2.5

* Día 4:
  h₄ = 0 + 0.5(2.5) = 1.25

* Día 5:
  h₅ = 0 + 0.5(1.25) = **0.625**

### Análisis

Aunque el evento inicial es muy grande (+10), su efecto disminuye rápidamente debido al factor de memoria (0.5).
Cada día el valor se reduce a la mitad, mostrando cómo la RNN **pierde información relevante con el tiempo** si no hay nuevas entradas.

Esto demuestra el problema clásico de las Vanilla RNN:
➡️ **no pueden mantener información importante por largos periodos**.

---

## Misión 2: El Rescate Emocional

**Objetivo:** Determinar qué tan fuerte debe ser una nueva entrada para revertir un estado negativo acumulado.

### Cálculo

* Día 1:
  h₁ = -6

* Día 2:
  h₂ = -4 + 0.5(-6) = -7

* Día 3:
  h₃ = 0 + 0.5(-7) = -3.5

Para el Día 4:

```
h₄ = x + 0.5(-3.5)
h₄ = x - 1.75
```

Condición:

```
h₄ > 0
```

Resolviendo:

```
x - 1.75 > 0
x > 1.75
```

### Análisis

El sistema arrastra un estado negativo acumulado, pero este se va reduciendo por el factor 0.5.

Para que el estado final sea positivo, el nuevo evento no necesita ser extremadamente grande, solo debe superar **1.75**.

Esto muestra que:

* La memoria negativa **sí influye**, pero
* También **se debilita con el tiempo**, permitiendo que nuevas entradas la superen.

---

## Misión 3: Constancia vs. El Pico

**Objetivo:** Comparar el efecto de un evento grande aislado contra eventos pequeños constantes.

---

### Escenario A: Evento único (pico)

* h₁ = 10
* h₂ = 5
* h₃ = 2.5
* h₄ = 1.25
* h₅ = **0.625**

---

### Escenario B: Eventos constantes

* h₁ = 3
* h₂ = 4.5
* h₃ = 5.25
* h₄ = 5.625
* h₅ = **5.8125**

---

### Análisis

En el Escenario A, el valor inicial es alto, pero disminuye rápidamente debido al desvanecimiento de la memoria.

En el Escenario B, aunque los eventos son más pequeños, su repetición constante permite que el estado se acumule y crezca.

Esto evidencia que la Vanilla RNN:

* Da mayor peso a la **información reciente**
* Favorece **patrones constantes sobre eventos aislados**

---

## Conclusión General

* La memoria en una Vanilla RNN se **desvanece exponencialmente**.
* Eventos pasados pierden impacto si no se refuerzan.
* Entradas constantes generan mayor efecto que un solo evento grande.
* El modelo tiene limitaciones para aprender dependencias a largo plazo.

Este comportamiento explica por qué en problemas más complejos se utilizan variantes como LSTM o GRU, que manejan mejor la memoria.
