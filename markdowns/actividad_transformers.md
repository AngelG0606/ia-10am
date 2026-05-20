## Actividad 1: Construcción de una matriz de atención

### Enunciado
Frase de análisis: **LA | NIÑA | COME | MANZANAS**

Desde la perspectiva de la palabra **COME**, asignamos valores del 0 al 10 para indicar qué tan importante es cada palabra al interpretar la acción del verbo.

---

### Paso 1: Puntajes de atención (desde "COME")

| | LA | NIÑA | COME | MANZANAS |
| :--- | :---: | :---: | :---: | :---: |
| **Desde COME →** | 1 | 7 | 5 | 8 |

---

### Paso 2: Conversión a porcentajes

* **Suma total:**  
$1 + 7 + 5 + 8 = 21$

* **Fórmula:**  
$(\text{Puntaje} \div 21) \times 100$

| Palabra | Puntaje | División | Porcentaje |
| :--- | :---: | :---: | :---: |
| **LA** | 1 | 0.047 | **5 %** |
| **NIÑA** | 7 | 0.333 | **33 %** |
| **COME** | 5 | 0.238 | **24 %** |
| **MANZANAS** | 8 | 0.381 | **38 %** |
| **TOTAL** | **21** | - | **100 %** |

---

### Paso 3: Interpretación

* **¿Qué palabras recibieron más atención?**  
Las palabras con mayor peso fueron **MANZANAS** y **NIÑA**, porque el verbo necesita identificar quién realiza la acción y qué objeto participa en ella.

* **¿La palabra MANZANAS tendría la misma fila de atención?**  
No. Cada palabra construye su propia perspectiva. MANZANAS probablemente miraría más hacia COME porque necesita entender la acción que se realiza sobre ella.

---

# Actividad 2: Palabras ambiguas y contexto

La palabra **LLAVE** cambia de significado dependiendo del contexto que la rodea.

---

## Frase A: PERDI LA LLAVE DEL CARRO

### Puntajes

| Palabra | PERDI | LA | LLAVE | DEL | CARRO | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Puntuación** | 2 | 1 | 4 | 3 | 10 | **20** |
| **Porcentaje** | 10% | 5% | 20% | 15% | **50%** | **100%** |

---

## Frase B: LA LLAVE ABRE LA TUBERIA

### Puntajes

| Palabra | LA | LLAVE | ABRE | LA | TUBERIA | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Puntuación** | 1 | 3 | 7 | 1 | 8 | **20** |
| **Porcentaje** | 5% | 15% | 35% | 5% | 40% | 100% |

---

## Análisis

1. **¿Dónde LLAVE se relaciona más con un vehículo?**  
En la Frase A, porque CARRO concentra la mayor atención.

2. **¿Dónde LLAVE se interpreta como herramienta?**  
En la Frase B, porque ABRE y TUBERIA cambian el significado.

3. **Conclusión**  
Los Transformers no almacenan un único significado fijo para las palabras. El contexto modifica dinámicamente la representación matemática de cada término.

---

# Actividad 3: Máscara causal

## Frase
**LA NIÑA COME MANZANAS**

---

## Matriz causal

| Palabra actual | LA | NIÑA | COME | MANZANAS |
| :--- | :---: | :---: | :---: | :---: |
| **1º LA** | ✓ | ✗ | ✗ | ✗ |
| **2º NIÑA** | ✓ | ✓ | ✗ | ✗ |
| **3º COME** | ✓ | ✓ | ✓ | ✗ |
| **4º MANZANAS** | ✓ | ✓ | ✓ | ✓ |

---

## Preguntas

* **¿Por qué la última palabra puede mirar todas las anteriores?**  
Porque ya fueron generadas y forman parte del contexto disponible.

* **¿Por qué la primera palabra solo puede verse a sí misma?**  
Porque no existe información previa.

* **¿Para qué sirve esta máscara?**  
Evita que el modelo consulte palabras futuras durante el entrenamiento y obliga a aprender predicción secuencial real.

---

# Actividad 4: Multi-Head Attention

## Frase
**CARLOS | NO | FUE | PORQUE | ESTABA | CANSADO**

Cada cabeza analiza la oración desde un criterio distinto.

---

## Cabeza A — Relación causal

* Suma: $1 + 1 + 2 + 5 + 3 + 8 = 20$

**Porcentajes:**  
CARLOS (5%) | NO (5%) | FUE (10%) | PORQUE (25%) | ESTABA (15%) | CANSADO (40%)

---

## Cabeza B — Identificación del sujeto

* Suma: $9 + 3 + 2 + 0 + 0 + 0 = 14$

**Porcentajes:**  
CARLOS (64%) | NO (21%) | FUE (15%) | PORQUE (0%) | ESTABA (0%) | CANSADO (0%)

---

## Cabeza C — Cercanía al verbo

* Suma: $0 + 8 + 4 + 8 + 0 + 0 = 20$

**Porcentajes:**  
CARLOS (0%) | NO (40%) | FUE (20%) | PORQUE (40%) | ESTABA (0%) | CANSADO (0%)

---

## Reflexión

* **¿Por qué usar varias cabezas?**  
Porque cada cabeza puede especializarse en un patrón diferente:
- relaciones gramaticales,
- causalidad,
- cercanía,
- sujetos,
- objetos.

Esto permite una comprensión mucho más rica del lenguaje.

---

# Actividad 5: Encoder y Decoder

## Explicación

El **Encoder** recibe toda la oración de entrada y genera representaciones internas con significado contextual.

El **Decoder** utiliza esas representaciones para construir una respuesta palabra por palabra.

---

## Preguntas de reflexión

### ¿Qué tarea resulta más sencilla?

Leer toda la oración es más fácil porque el contexto ya está completo.

---

### ¿Cuándo necesita el Decoder mirar atrás?

Cuando debe mantener coherencia entre palabras ya generadas y las nuevas.

---

### ¿Cómo se relaciona con un traductor?

En traducción:
- el Encoder entiende el idioma origen,
- el Decoder produce secuencialmente el idioma destino.

En chatbots sucede algo parecido:
- el mensaje del usuario se interpreta,
- la respuesta se genera token por token.