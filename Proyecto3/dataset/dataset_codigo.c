/**
 * Sistema de Control de Estructuras de Datos y Algoritmos en C
 * Alumno: Angel
 * Proyecto: Asistente de Codigo Personalizado - RNN Vanilla
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Bloque 1: Matematicas y Aritmetica Basica --- */

// funcion: calcular factorial de un numero entero
int calcular_factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * calcular_factorial(n - 1);
}

// funcion: calcular sucesion de fibonacci hasta n
int calcular_fibonacci(int n) {
    int a = 0;
    int b = 1;
    int c = 0;
    for (int i = 0; i < n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return a;
}

// funcion: obtener el valor maximo entre dos enteros
int obtener_maximo(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

// funcion: obtener el valor minimo entre dos enteros
int obtener_minimo(int a, int b) {
    if (a < b) {
        return a;
    }
    return b;
}

// funcion: calcular la potencia de una base elevada a un exponente
int calcular_potencia(int base, int exp) {
    int resultado = 1;
    for (int i = 0; i < exp; i++) {
        resultado *= base;
    }
    return resultado;
}

// funcion: verificar si un numero es primo
int verificar_primo(int n) {
    if (n <= 1) {
        return 0;
    }
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) {
            return 0;
        }
    }
    return 1;
}

// funcion: calcular el maximo comun divisor mcd de dos numeros
int calcular_mcd(int a, int b) {
    while (b != 0) {
        int temporal = b;
        b = a % b;
        a = temporal;
    }
    return a;
}

// funcion: calcular el minimo comun multiplo mcm de dos numeros
int calcular_mcm(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return (a * b) / calcular_mcd(a, b);
}

// funcion: calcular el valor absoluto de un numero decimal double
double calcular_valor_absoluto(double x) {
    if (x < 0) {
        return -x;
    }
    return x;
}

// funcion: calcular el promedio de un arreglo de enteros
double calcular_promedio(int arreglo[], int tamano) {
    double suma = 0.0;
    for (int i = 0; i < tamano; i++) {
        suma += arreglo[i];
    }
    return suma / tamano;
}

/* --- Bloque 2: Manejo de Arreglos y Ordenamiento --- */

// funcion: ordenar un arreglo usando el algoritmo de burbuja
void ordenar_burbuja(int arreglo[], int tamano) {
    for (int i = 0; i < tamano - 1; i++) {
        for (int j = 0; j < tamano - i - 1; j++) {
            if (arreglo[j] > arreglo[j + 1]) {
                int temporal = arreglo[j];
                arreglo[j] = arreglo[j + 1];
                arreglo[j + 1] = temporal;
            }
        }
    }
}

// funcion: ordenar un arreglo usando el algoritmo de seleccion
void ordenar_seleccion(int arreglo[], int tamano) {
    for (int i = 0; i < tamano - 1; i++) {
        int indice_min = i;
        for (int j = i + 1; j < tamano; j++) {
            if (arreglo[j] < arreglo[indice_min]) {
                indice_min = j;
            }
        }
        int temporal = arreglo[indice_min];
        arreglo[indice_min] = arreglo[i];
        arreglo[i] = temporal;
    }
}

// funcion: ordenar un arreglo usando el algoritmo de insercion
void ordenar_insercion(int arreglo[], int tamano) {
    for (int i = 1; i < tamano; i++) {
        int clave = arreglo[i];
        int j = i - 1;
        while (j >= 0 && arreglo[j] > clave) {
            arreglo[j + 1] = arreglo[j];
            j = j - 1;
        }
        arreglo[j + 1] = clave;
    }
}

// funcion: realizar una busqueda lineal en un arreglo
int busqueda_lineal(int arreglo[], int tamano, int objetivo) {
    for (int i = 0; i < tamano; i++) {
        if (arreglo[i] == objetivo) {
            return i;
        }
    }
    return -1;
}

// funcion: realizar una busqueda binaria en un arreglo ordenado
int busqueda_binaria(int arreglo[], int tamano, int objetivo) {
    int izquierda = 0;
    int derecha = tamano - 1;
    while (izquierda <= derecha) {
        int medio = izquierda + (derecha - izquierda) / 2;
        if (arreglo[medio] == objetivo) {
            return medio;
        }
        if (arreglo[medio] < objetivo) {
            izquierda = medio + 1;
        } else {
            derecha = medio - 1;
        }
    }
    return -1;
}

// funcion: invertir el orden de los elementos de un arreglo
void invertir_arreglo(int arreglo[], int tamano) {
    int inicio = 0;
    int fin = tamano - 1;
    while (inicio < fin) {
        int temporal = arreglo[inicio];
        arreglo[inicio] = arreglo[fin];
        arreglo[fin] = temporal;
        inicio++;
        fin--;
    }
}

// funcion: encontrar el valor maximo dentro de un arreglo
int encontrar_maximo_arreglo(int arreglo[], int tamano) {
    int maximo = arreglo[0];
    for (int i = 1; i < tamano; i++) {
        if (arreglo[i] > maximo) {
            maximo = arreglo[i];
        }
    }
    return maximo;
}

// funcion: encontrar el valor minimo dentro de un arreglo
int encontrar_minimo_arreglo(int arreglo[], int tamano) {
    int minimo = arreglo[0];
    for (int i = 1; i < tamano; i++) {
        if (arreglo[i] < minimo) {
            minimo = arreglo[i];
        }
    }
    return minimo;
}

// funcion: sumar todos los elementos de un arreglo
int sumar_elementos_arreglo(int arreglo[], int tamano) {
    int suma = 0;
    for (int i = 0; i < tamano; i++) {
        suma += arreglo[i];
    }
    return suma;
}

// funcion: copiar los elementos de un arreglo origen a un destino
void copiar_arreglo(int origen[], int destino[], int tamano) {
    for (int i = 0; i < tamano; i++) {
        destino[i] = origen[i];
    }
}

/* --- Bloque 3: Cadenas de Caracteres (Strings) --- */

// funcion: obtener la longitud de una cadena de texto
int obtener_longitud_cadena(char cadena[]) {
    int longitud = 0;
    while (cadena[longitud] != '\0') {
        longitud++;
    }
    return longitud;
}

// funcion: copiar una cadena origen en una cadena destino
void copiar_cadena(char origen[], char destino[]) {
    int i = 0;
    while (origen[i] != '\0') {
        destino[i] = origen[i];
        i++;
    }
    destino[i] = '\0';
}

// funcion: concatenar dos cadenas de caracteres
void concatenar_cadenas(char destino[], char origen[]) {
    int i = 0;
    while (destino[i] != '\0') {
        i++;
    }
    int j = 0;
    while (origen[j] != '\0') {
        destino[i] = origen[j];
        i++;
        j++;
    }
    destino[i] = '\0';
}

// funcion: comparar dos cadenas de caracteres de forma alfabetica
int comparar_cadenas(char cadena1[], char cadena2[]) {
    int i = 0;
    while (cadena1[i] != '\0' && cadena2[i] != '\0') {
        if (cadena1[i] != cadena2[i]) {
            return cadena1[i] - cadena2[i];
        }
        i++;
    }
    return cadena1[i] - cadena2[i];
}

// funcion: invertir los caracteres de una cadena de texto
void invertir_cadena(char cadena[]) {
    int longitud = obtener_longitud_cadena(cadena);
    int inicio = 0;
    int fin = longitud - 1;
    while (inicio < fin) {
        char temporal = cadena[inicio];
        cadena[inicio] = cadena[fin];
        cadena[fin] = temporal;
        inicio++;
        fin--;
    }
}

// funcion: convertir todos los caracteres de una cadena a mayusculas
void convertir_a_mayusculas(char cadena[]) {
    int i = 0;
    while (cadena[i] != '\0') {
        if (cadena[i] >= 'a' && cadena[i] <= 'z') {
            cadena[i] = cadena[i] - 32;
        }
        i++;
    }
}

// funcion: convertir todos los caracteres de una cadena a minusculas
void convertir_a_minusculas(char cadena[]) {
    int i = 0;
    while (cadena[i] != '\0') {
        if (cadena[i] >= 'A' && cadena[i] <= 'Z') {
            cadena[i] = cadena[i] + 32;
        }
        i++;
    }
}

// funcion: contar la cantidad de vocales en una cadena de texto
int contar_vocales(char cadena[]) {
    int conteo = 0;
    int i = 0;
    while (cadena[i] != '\0') {
        char c = cadena[i];
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
            c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U') {
            conteo++;
        }
        i++;
    }
    return conteo;
}

// funcion: verificar si una cadena de texto es un palindromo
int verificar_palindromo(char cadena[]) {
    int longitud = obtener_longitud_cadena(cadena);
    int inicio = 0;
    int fin = longitud - 1;
    while (inicio < fin) {
        if (cadena[inicio] != cadena[fin]) {
            return 0;
        }
        inicio++;
        fin--;
    }
    return 1;
}

// funcion: remover el caracter de salto de linea al final de una cadena
void limpiar_nueva_linea(char cadena[]) {
    int longitud = obtener_longitud_cadena(cadena);
    if (longitud > 0 && cadena[longitud - 1] == '\n') {
        cadena[longitud - 1] = '\0';
    }
}

/* --- Bloque 4: Estructuras de Datos Lineales (Pilas y Colas) --- */

struct Nodo {
    int dato;
    struct Nodo* siguiente;
};

// funcion: crear un nuevo nodo en memoria dinamica para estructuras enlazadas
struct Nodo* crear_nodo(int valor) {
    struct Nodo* nuevo = (struct Nodo*)malloc(sizeof(struct Nodo));
    nuevo->dato = valor;
    nuevo->siguiente = NULL;
    return nuevo;
}

// funcion: empilar o meter un elemento en la parte superior de una pila
void empilar_elemento(struct Nodo** superior, int valor) {
    struct Nodo* nuevo = crear_nodo(valor);
    nuevo->siguiente = *superior;
    *superior = nuevo;
}

// funcion: desempilar o sacar el elemento superior de una pila
int desempilar_elemento(struct Nodo** superior) {
    if (*superior == NULL) {
        return -1;
    }
    struct Nodo* temporal = *superior;
    int valor = temporal->dato;
    *superior = (*superior)->siguiente;
    free(temporal);
    return valor;
}

// funcion: mirar el dato que se encuentra en la cima de una pila
int mirar_superior_pila(struct Nodo* superior) {
    if (superior == NULL) {
        return -1;
    }
    return superior->dato;
}

// funcion: verificar si una pila se encuentra vacia
int verificar_pila_vacia(struct Nodo* superior) {
    if (superior == NULL) {
        return 1;
    }
    return 0;
}

// funcion: encolar o meter un elemento al final de una cola
void encolar_elemento(struct Nodo** frente, struct Nodo** fin, int valor) {
    struct Nodo* nuevo = crear_nodo(valor);
    if (*fin == NULL) {
        *frente = nuevo;
        *fin = nuevo;
        return;
    }
    (*fin)->siguiente = nuevo;
    *fin = nuevo;
}

// funcion: desencolar o sacar el primer elemento de una cola
int desencolar_elemento(struct Nodo** frente, struct Nodo** fin) {
    if (*frente == NULL) {
        return -1;
    }
    struct Nodo* temporal = *frente;
    int valor = temporal->dato;
    *frente = (*frente)->siguiente;
    if (*frente == NULL) {
        *fin = NULL;
    }
    free(temporal);
    return valor;
}

// funcion: mirar el dato que esta al frente de la cola
int mirar_frente_cola(struct Nodo* frente) {
    if (frente == NULL) {
        return -1;
    }
    return frente->dato;
}

// funcion: verificar si una cola se encuentra vacia
int verificar_cola_vacia(struct Nodo* frente) {
    if (frente == NULL) {
        return 1;
    }
    return 0;
}

// funcion: liberar de memoria todos los elementos de una lista enlazada
void vaciar_lista(struct Nodo** cabeza) {
    struct Nodo* actual = *cabeza;
    struct Nodo* siguiente = NULL;
    while (actual != NULL) {
        siguiente = actual->siguiente;
        free(actual);
        actual = siguiente;
    }
    *cabeza = NULL;
}

/* --- Bloque 5: Matrices Bidimensionales --- */

// funcion: inicializar todas las celdas de una matriz con un valor fijo
void inicializar_matriz(int matriz[3][3], int valor) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matriz[i][j] = valor;
        }
    }
}

// funcion: sumar dos matrices de dimensiones tres por tres
void sumar_matrices(int m1[3][3], int m2[3][3], int resultado[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            resultado[i][j] = m1[i][j] + m2[i][j];
        }
    }
}

// funcion: multiplicar dos matrices de dimensiones tres por tres
void multiplicar_matrices(int m1[3][3], int m2[3][3], int resultado[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            resultado[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                resultado[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

// funcion: transponer las filas y columnas de una matriz bidimensional
void transponer_matriz(int matriz[3][3], int resultado[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            resultado[j][i] = matriz[i][j];
        }
    }
}

// funcion: obtener la traza o suma de la diagonal principal de una matriz
int obtener_traza_matriz(int matriz[3][3]) {
    int traza = 0;
    for (int i = 0; i < 3; i++) {
        get_traza += matriz[i][i];
    }
    return traza;
}

// funcion: imprimir una matriz de tres por tres en consola
void imprimir_matriz(int matriz[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", matriz[i][j]);
        }
        printf("\n");
    }
}

// funcion: verificar si una matriz bidimensional es una matriz identidad
int verificar_matriz_identidad(int matriz[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j && matriz[i][j] != 1) {
                return 0;
            }
            if (i != j && matriz[i][j] != 0) {
                return 0;
            }
        }
    }
    return 1;
}

// funcion: multiplicar todos los elementos de una matriz por un escalar
void multiplicar_matriz_escalar(int matriz[3][3], int escalar) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matriz[i][j] *= escalar;
        }
    }
}

// funcion: buscar si un numero objetivo existe dentro de una matriz
int buscar_elemento_matriz(int matriz[3][3], int objetivo) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (matriz[i][j] == objetivo) {
                return 1;
            }
        }
    }
    return 0;
}

// funcion: obtener el valor maximo almacenado dentro de una matriz
int obtener_maximo_matriz(int matriz[3][3]) {
    int maximo = matriz[0][0];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (matriz[i][j] > maximo) {
                maximo = matriz[i][j];
            }
        }
    }
    return maximo;
}

/* --- Bloque 6: Geometria y Vectores --- */

struct Punto2D {
    double x;
    double y;
};

// funcion: calcular la distancia euclidiana entre dos puntos en el plano cartesiano
double calcular_distancia_puntos(struct Punto2D p1, struct Punto2D p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

// funcion: calcular la pendiente de una recta que pasa por dos puntos coordenados
double calcular_pendiente(struct Punto2D p1, struct Punto2D p2) {
    if (p2.x - p1.x == 0) {
        return 99999.9;
    }
    return (p2.y - p1.y) / (p2.x - p1.x);
}

// funcion: verificar si dos estructuras Punto2D representan la misma coordenada
int verificar_puntos_iguales(struct Punto2D p1, struct Punto2D p2) {
    if (p1.x == p2.x && p1.y == p2.y) {
        return 1;
    }
    return 0;
}

// funcion: obtener la coordenada del punto medio entre dos puntos dados
struct Punto2D obtener_punto_medio(struct Punto2D p1, struct Punto2D p2) {
    struct Punto2D medio;
    medio.x = (p1.x + p2.x) / 2.0;
    medio.y = (p1.y + p2.y) / 2.0;
    return medio;
}

// funcion: calcular el area de un triangulo con base y altura decimales
double calcular_area_triangulo(double base, double altura) {
    return (base * altura) / 2.0;
}

// funcion: calcular el perimetro de un rectangulo
double calcular_perimetro_rectangulo(double base, double altura) {
    return 2.0 * (base + altura);
}

// funcion: calcular el area de un circulo dado su radio decimal
double calcular_area_circulo(double radio) {
    return 3.1415926535 * radio * radio;
}

// funcion: calcular el perimetro de la circunferencia de un circulo
double calcular_perimetro_circulo(double radio) {
    return 2.0 * 3.1415926535 * radio;
}

// funcion: calcular el volumen de una esfera utilizando pi
double calcular_volumen_esfera(double radio) {
    return (4.0 / 3.0) * 3.1415926535 * pow(radio, 3);
}

// funcion: calcular la hipotenusa usando el teorema de pitagoras
double calcular_hipotenusa(double cateto1, double cateto2) {
    return sqrt((cateto1 * cateto1) + (cateto2 * cateto2));
}

/* --- Bloque 7: Utilidades de Sistema e Informacion --- */

// funcion: mostrar el encabezado visual del sistema en la consola
void mostrar_encabezado_sistema() {
    printf("=========================================\n");
    printf("     SISTEMA DE ASISTENCIA V1.0          \n");
    printf("=========================================\n");
}

// funcion: mostrar el pie de pagina o derechos de autor del sistema
void mostrar_pie_sistema() {
    printf("=========================================\n");
    printf("     Derechos Reservados - Angel 2026    \n");
    printf("=========================================\n");
}

// funcion: convertir un caracter numerico a su correspondiente entero int
int convertir_char_a_int(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    return -1;
}

// funcion: convertir un entero de un solo digito a su caracter ASCII correspondiente
char convertir_int_a_char(int n) {
    if (n >= 0 && n <= 9) {
        return n + '0';
    }
    return ' ';
}