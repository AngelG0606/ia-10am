# Actividad manual Parte 2 — Entender Transformers

# Actividad 6 — Matriz de atención completa

## Frase

LA   NIÑA   PEQUEÑA   COME   FRUTA

---

## Paso 1 — Puntajes

| Desde ↓ / Hacia → | LA | NIÑA | PEQUEÑA | COME | FRUTA |
|---|---|---|---|---|---|
| LA | 2 | 8 | 4 | 1 | 1 |
| NIÑA | 3 | 7 | 8 | 9 | 4 |
| PEQUEÑA | 1 | 9 | 7 | 3 | 2 |
| COME | 1 | 7 | 2 | 5 | 9 |
| FRUTA | 1 | 4 | 2 | 9 | 6 |

---

## Paso 2 — Normalización

### Fila LA

Suma:

16

| LA | NIÑA | PEQUEÑA | COME | FRUTA |
|---|---|---|---|---|
| 13% | 50% | 25% | 6% | 6% |

---

### Fila NIÑA

Suma:

31

| LA | NIÑA | PEQUEÑA | COME | FRUTA |
|---|---|---|---|---|
| 10% | 23% | 26% | 29% | 13% |

---

### Fila PEQUEÑA

Suma:

22

| LA | NIÑA | PEQUEÑA | COME | FRUTA |
|---|---|---|---|---|
| 5% | 41% | 32% | 14% | 9% |

---

### Fila COME

Suma:

24

| LA | NIÑA | PEQUEÑA | COME | FRUTA |
|---|---|---|---|---|
| 4% | 29% | 8% | 21% | 38% |

---

### Fila FRUTA

Suma:

22

| LA | NIÑA | PEQUEÑA | COME | FRUTA |
|---|---|---|---|---|
| 5% | 18% | 9% | 41% | 27% |

---

## Interpretación

- COME presta más atención a FRUTA porque el verbo necesita saber qué se está comiendo.
- FRUTA mira mucho a COME porque depende del verbo para entender su función.
- PEQUEÑA mira mucho a NIÑA porque es un adjetivo del sustantivo.

---

## Preguntas de análisis

### ¿La fila de COME se parece a la de FRUTA?

Sí, porque ambas están relacionadas semánticamente:
- COME necesita un objeto.
- FRUTA depende del verbo para tener contexto.

---

### ¿Qué palabra podría repartir atención casi pareja?

Una palabra como LA podría hacerlo, porque tiene menos significado propio y sirve más como conexión gramatical.

---

### Si hubiera 100 palabras, ¿cuántas celdas tendría la tabla?

100 × 100 = 10,000

Porque cada palabra puede atender a todas las demás.

---

# Actividad 7 — Softmax manual

## Datos

| Palabra | Puntaje |
|---|---|
| NIÑA | 3.0 |
| PEQUEÑA | 0.5 |
| COME | 0.2 |
| FRUTA | 1.0 |

---

## Paso 1 — Exponenciales

| Palabra | e^x |
|---|---|
| NIÑA | 20.09 |
| PEQUEÑA | 1.65 |
| COME | 1.22 |
| FRUTA | 2.72 |

Suma:

25.68

---

## Paso 2 — Probabilidades

### NIÑA

20.09 / 25.68 = 0.782 ≈ 78%

### PEQUEÑA

1.65 / 25.68 = 0.064 ≈ 6%

### COME

1.22 / 25.68 = 0.047 ≈ 5%

### FRUTA

2.72 / 25.68 = 0.106 ≈ 11%

---

## Pregunta

### ¿Por qué no basta dividir los puntajes entre la suma?

Porque:
- Puede haber números negativos.
- Softmax exagera diferencias importantes.
- Garantiza probabilidades positivas.
- Hace que el modelo enfoque atención fuerte donde realmente importa.

---

# Actividad 8 — Mezcla ponderada de vectores

## Vectores

| Palabra | Vector |
|---|---|
| LA | (1,1) |
| NIÑA | (4,5) |
| PEQUEÑA | (3,4) |
| COME | (5,1) |
| FRUTA | (6,3) |

---

## Pesos

| Hacia | % |
|---|---|
| LA | 5% |
| NIÑA | 35% |
| PEQUEÑA | 10% |
| COME | 10% |
| FRUTA | 40% |

---

## Paso 1 — Decimales

| Palabra | Peso |
|---|---|
| LA | 0.05 |
| NIÑA | 0.35 |
| PEQUEÑA | 0.10 |
| COME | 0.10 |
| FRUTA | 0.40 |

---

## Paso 2 — Multiplicar

### LA

0.05(1,1) = (0.05,0.05)

### NIÑA

0.35(4,5) = (1.4,1.75)

### PEQUEÑA

0.10(3,4) = (0.3,0.4)

### COME

0.10(5,1) = (0.5,0.1)

### FRUTA

0.40(6,3) = (2.4,1.2)

---

## Paso 3 — Sumar

### Coordenada x

0.05 + 1.4 + 0.3 + 0.5 + 2.4 = 4.65

### Coordenada y

0.05 + 1.75 + 0.4 + 0.1 + 1.2 = 3.5

---

## Resultado final

(4.65, 3.5)

La salida queda más cerca de:
- FRUTA
- NIÑA

porque fueron las palabras con mayor atención.

---

# Actividad 9 — Máscara de padding

## Frase

EL   GATO   COME   —   —

---

## Explicación

Las posiciones 4 y 5 son PAD.

Las palabras reales NO pueden mirar PAD.

Entonces se tachan:
- fila EL → columnas PAD
- fila GATO → columnas PAD
- fila COME → columnas PAD

---

## Pregunta

### ¿Por qué la frase larga necesita menos tachaduras?

Porque casi todas sus posiciones contienen palabras reales.

---

### ¿Qué pasaría si el modelo atendiera PAD?

Aprendería ruido y patrones falsos del relleno.

---

# Actividad 10 — Cross-attention

## Encoder

YO   QUIERO   CAFE

## Decoder

I   WANT   ___

---

## Matriz

| Desde ↓ / Español → | YO | QUIERO | CAFE |
|---|---|---|---|
| Palabra 3 | 1 | 2 | 10 |

---

## Interpretación

La próxima palabra debería mirar principalmente a CAFE porque probablemente quiere generar:

COFFEE

---

## Pregunta

### ¿I podría mirar mucho a YO?

Sí, porque:
- YO ↔ I
- representan el mismo sujeto.

---

# Actividad 11 — MLM estilo BERT

## Frase

EL   GATO   ___   PESCADO

---

## Candidatos

- COME
- DUERME
- VERDE
- RAPIDO

---

## Puntajes ejemplo

| Palabra | Puntaje |
|---|---|
| COME | 10 |
| DUERME | 4 |
| VERDE | 1 |
| RAPIDO | 1 |

---

## Reflexión

- COME tiene sentido con GATO y PESCADO.
- DUERME podría tener algo de sentido con GATO, pero no con PESCADO.
- VERDE no conecta semánticamente.
- BERT necesita mirar a ambos lados para usar contexto completo.

---

# Actividad 12 — Dos capas de atención

## Perfiles

| Palabra | Tras capa 1 |
|---|---|
| LA | 2 |
| NIÑA | 6 |
| PEQUEÑA | 5 |
| COME | 7 |
| FRUTA | 8 |

---

## Atención desde FRUTA

| LA | NIÑA | PEQUEÑA | COME | FRUTA |
|---|---|---|---|---|
| 1 | 6 | 3 | 10 | 7 |

---

## Interpretación

FRUTA ahora entiende mejor que:
- COME es el verbo principal.
- NIÑA está relacionada con la acción.

---

# Actividad 13 — RNN vs Transformer

## RNN

A → B → C → D → E

Saltos de A a E:

4

---

## Transformer

Cada palabra puede mirar directamente a todas.

Con 5 palabras:

5 × 5 = 25

celdas.

---

## Respuestas

### ¿Cuántos saltos necesita la RNN?

4.

### ¿Y atención?

1.

---

## ¿Por qué seguimos usando Transformers?

Porque:
- procesan palabras en paralelo,
- capturan relaciones lejanas mejor,
- entrenan más rápido.

Aunque usan más memoria en textos largos.

---

# Actividad 14 — Escalar por √dk

## Softmax de [8,2,2,2]

| Valor | e^x |
|---|---|
| 8 | 2981 |
| 2 | 7.39 |
| 2 | 7.39 |
| 2 | 7.39 |

Suma:

3003.17

Probabilidad principal:

2981 / 3003 ≈ 99%

---

## Softmax de [4,1,1,1]

| Valor | e^x |
|---|---|
| 4 | 54.6 |
| 1 | 2.71 |
| 1 | 2.71 |
| 1 | 2.71 |

Suma:

62.73

Probabilidad principal:

54.6 / 62.73 ≈ 87%

---

## Conclusión

La palabra fuerte sigue ganando, pero:
- las demás conservan influencia,
- el modelo evita saturarse,
- varias palabras pueden “tener voz”.

Frase importante:

"Dividir entre √dk es bajar el volumen antes de repartir atención, para que varias palabras sigan teniendo voz."