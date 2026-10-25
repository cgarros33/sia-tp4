# Ruido para la recuperación de QR — Ejercicio 2.1

## 1. Objetivo

Definir los modelos de ruido con los que se evalúa la memoria asociativa de
Hopfield sobre los cinco códigos QR de `datos/hopfield/qr_37x37.csv`. El archivo
presenta las opciones, los parámetros, los niveles y las decisiones tomadas
antes de ejecutar el barrido. Los resultados numéricos van a
`docs/decisiones.md` y a los CSV de `resultados/ej2/hopfield/qr_37x37/`.

## 2. Representación

- Cada QR es un vector de 37 × 37 = 1369 bits.
- `1` representa negro y `-1` representa blanco.
- Los cinco patrones son `beverages_menu_qr`, `dessert_menu_qr`,
  `food_menu_qr`, `specials_menu_qr` y `wine_menu_qr`.
- El archivo de entrada es de solo lectura.
- La consulta ruidosa se genera a partir del patrón original y se usa con las
  dos reglas de pesos, Hebb y Personnaz, para que la comparación sea justa.
- `fraccion_afectada` es la fracción de bits que efectivamente cambian respecto
  del original. No es el área nominal del cuadrado, el disco o la línea.
- `hamming_inicial_origen` es la cantidad de bits cambiados por el ruido.
- `hamming_final_origen` es la cantidad de bits que quedan distintos del
  original después de la recuperación.

## 3. Criterios de diseño

1. **Reproducibilidad:** cada consulta usa una semilla explícita y un
   `numpy.random.Generator`. La misma semilla y el mismo nivel producen la
   misma consulta.
2. **Comparabilidad:** todos los modelos registran la fracción afectada real.
   Las figuras globales usan esa fracción, no solo el parámetro nominal.
3. **Realismo:** los modelos cubren desgaste genérico, esquina rota, borrado de
   un finder, rayones, manchas y faltantes grandes.
4. **Independencia del modelo:** la consulta ruidosa no depende de la regla de
   pesos. Hebb y Personnaz reciben exactamente el mismo estado inicial.
5. **Niveles fijados a priori:** los niveles se eligen antes de ver los
   resultados y no se ajustan para mejorar la exactitud.
6. **Media ± desvío:** se usan tres semillas, cinco repeticiones por patrón y
   los cinco patrones. Cada resultado agregado es media ± desvío muestral sobre
   las semillas.
7. **Separación de daño y recuperación:** el generador de ruido solo construye
   el estado inicial. La recuperación, la clasificación y las métricas se
   calculan después.
8. **Trazabilidad:** se guardan la semilla, el nivel, los parámetros efectivos y
   la fracción afectada de cada consulta.

## 4. Opciones de ruido

Se evaluaron seis modelos individuales. Todos se ejecutan por separado y luego
se combinan en una escalera acumulativa.

### 4.1 Comparación general

| # | Modelo | Qué simula | Realismo | Costo | Trazabilidad | Decisión |
|---|---|---|---|---|---|---|
| 1 | `aleatorio` | desgaste genérico, bits sueltos | Medio | Bajo | Alta | Se usa |
| 2 | `esquina` | esquina rota o faltante | Alto | Bajo | Alta | Se usa |
| 3 | `finder` | pérdida de un patrón de localización | Muy alto | Bajo | Alta | Se usa |
| 4 | `rayones` | papel rayado o raspado | Alto | Medio | Alta | Se usa |
| 5 | `manchas` | comida o líquido sobre el papel | Alto | Medio | Alta | Se usa |
| 6 | `bloques` | regiones grandes faltantes | Medio | Bajo | Alta | Se usa |

### 4.2 Ruido aleatorio uniforme

**Definición:** se eligen al azar `round(f · N)` posiciones distintas y se
invierte su bit.

**Parámetros:**

- `f`: fracción de bits invertidos, entre 0 y 1.
- `N = 1369`.
- `semilla`: semilla del generador.

**Niveles:**

```text
0.02, 0.05, 0.10, 0.15, 0.20, 0.25,
0.30, 0.35, 0.40, 0.45, 0.50
```

**Bits aproximados:**

| Nivel | Bits invertidos |
|---:|---:|
| 0.02 | 27 |
| 0.05 | 68 |
| 0.10 | 137 |
| 0.20 | 274 |
| 0.30 | 411 |
| 0.40 | 548 |
| 0.50 | 685 |

**Qué mide:** el tamaño del área de atracción de cada patrón almacenado. Es la
línea de base del ejercicio y la única que usa inversión de bits.

**Ventajas:** simple, reproducible, comparable con el ejercicio 2.2 y con
cualquier red de Hopfield.

**Limitaciones:** el daño real no es independiente por píxel. No modela una
esquina rota, un rayón ni una mancha.

**Ejemplo:**

```text
Original        Consulta con 5% de bits invertidos
#######         ##.#.##
#.....#         #..#..#
#.###.#         #.##..#
#.....#         #....##
#######         ###.###
```

### 4.3 Esquina rota

**Definición:** se elige una esquina y se borra un cuadrado de lado `L`,
poniendo sus píxeles en `-1` (blanco). La esquina se elige al azar entre las
cuatro en cada repetición. La elección se registra.

**Parámetros:**

- `L`: lado del cuadrado, en píxeles.
- `esquina`: superior izquierda, superior derecha, inferior izquierda o
  inferior derecha.
- `modo`: borrar a `-1`.

**Niveles:**

```text
7, 9, 11, 15, 19, 23
```

**Fracción nominal aproximada:**

| Lado | Área | Fracción de N |
|---:|---:|---:|
| 7 | 49 | 3,6% |
| 9 | 81 | 5,9% |
| 11 | 121 | 8,8% |
| 15 | 225 | 16,4% |
| 19 | 361 | 26,4% |
| 23 | 529 | 38,6% |

**Qué mide:** daño localizado en una esquina, motivado por la consigna. Si la
esquina elegida contiene un finder, el daño es más severo para un lector de QR,
pero la red puede reconstruirlo a partir de los otros patrones.

**Ventajas:** directo, interpretable y alineado con el caso de uso.

**Limitaciones:** el cuadrado no respeta la forma real de una rotura. La
fracción efectiva puede ser menor que el área nominal si algunos píxeles ya
eran blancos.

**Ejemplo:**

```text
Original        Esquina inferior derecha borrada
#######         #######
#.....#         #.....#
#.###.#         #.###.#
#.....#         #.....#
#######         ####...
```

### 4.4 Borrado de finder

**Definición:** se elige uno de los tres patrones de localización del QR
(superior izquierda, superior derecha o inferior izquierda) y se borra a
blanco. El nivel puede incluir solo el finder de 7 × 7 o también su separador,
con un cuadrado de 9 × 9. También se prueban dos y tres finders.

**Parámetros:**

- `tamano`: `7` (finder) o `9` (finder + separador).
- `cantidad`: 1, 2 o 3 finders.
- `finder`: posición elegida al azar entre las disponibles.

**Niveles:**

```text
1 finder de 7x7
1 finder de 9x9
2 finders de 7x7
3 finders de 7x7
```

**Fracción nominal aproximada:** 3,6% (7 × 7), 5,9% (9 × 9), 7,2% (dos de
7 × 7) y 10,8% (tres de 7 × 7), antes de descontar píxeles que ya eran blancos.

**Qué mide:** el caso central de la consigna: una esquina rota que afecta un
patrón de localización. Es el daño estructural más específico de un QR.

**Ventajas:** muy interpretable; permite comparar la pérdida de un finder con
el borrado de una esquina genérica.

**Limitaciones:** los finders son iguales en los cinco QR, así que el daño
puede ser más fácil de reconstruir que un daño en la zona de datos.

**Ejemplo:**

```text
Original        Finder superior izquierdo borrado
#######         .......
#.....#         .......
#.###.#         ..###..
#.....#         .......
#######         .......
```

### 4.5 Rayones

**Definición:** se trazan líneas rectas completas sobre la imagen, con
orientación horizontal, vertical o diagonal, y se borran a blanco los píxeles
que caen sobre ellas. El grosor es de un píxel. La posición y la orientación se
eligen al azar y se registran.

**Parámetros:**

- `cantidad`: número de líneas.
- `grosor`: 1 píxel.
- `largo`: completo, de borde a borde.
- `orientacion`: horizontal, vertical o diagonal.

**Niveles:**

```text
1, 2, 4, 8, 12 líneas
```

**Fracción aproximada:** una línea completa afecta alrededor del 2,7% de N; la
fracción real se registra y descuenta solapamientos y píxeles ya blancos.

**Qué mide:** daño fino y alargado, típico de papel rayado o raspado. Puede
cruzar zonas de datos y patrones de sincronización.

**Ventajas:** realista y con una relación área/perímetro muy distinta a la de
una mancha o un bloque.

**Limitaciones:** la posición exacta de un rayón es muy variable; por eso se
promedian varias semillas y repeticiones.

**Ejemplo:**

```text
Original        Dos rayones
#######         ##.####
#.....#         #.#...#
#.###.#         #..##.#
#.....#         #....##
#######         ###.###
```

### 4.6 Manchas

**Definición:** se eligen uno o más centros al azar y se borra un disco de
radio `r`, poniendo sus píxeles en `-1` (blanco). Los centros y radios se
registran.

**Parámetros:**

- `cantidad`: número de manchas.
- `radio`: radio en píxeles.
- `centro`: posición elegida al azar.

**Niveles:**

```text
1 mancha de radio 3
1 mancha de radio 5
1 mancha de radio 7
1 mancha de radio 10
2 manchas de radio 5
3 manchas de radio 5
```

**Fracción nominal aproximada:**

| Configuración | Área aproximada | Fracción de N |
|---|---:|---:|
| 1 × r3 | 29 | 2,1% |
| 1 × r5 | 81 | 5,9% |
| 1 × r7 | 154 | 11,2% |
| 1 × r10 | 317 | 23,2% |
| 2 × r5 | 162 | 11,8% |
| 3 × r5 | 243 | 17,7% |

**Qué mide:** daño contiguo e interior, como una mancha de comida o líquido. Es
el caso cotidiano de la historia de la trattoria.

**Ventajas:** realista y con una forma distinta a la esquina y al bloque.

**Limitaciones:** si la mancha cae sobre una zona blanca, la fracción efectiva
es menor que el área del disco.

**Ejemplo:**

```text
Original        Mancha de radio 5
#######         #######
#.....#         #.....#
#.###.#         #..##.#
#.....#         #....##
#######         #######
```

### 4.7 Bloques borrados

**Definición:** se divide la imagen en una grilla de bloques de 6 × 6 píxeles
(la última fila y la última columna pueden quedar parciales). Se elige una
fracción de bloques al azar y se borran completos a `-1` (blanco).

**Parámetros:**

- `fraccion`: fracción de bloques a borrar.
- `tamano_bloque`: 6 × 6.
- `bloques`: índices elegidos, registrados.

**Niveles:**

```text
5%, 10%, 20%, 30%, 40%
```

**Qué mide:** faltantes grandes y rectangulares, como un papel rasgado o una
zona ilegible. Es el modelo estructural más severo.

**Ventajas:** permite controlar la fracción afectada de forma casi exacta y
comparar con el ruido aleatorio.

**Limitaciones:** la grilla no coincide con la estructura modular del QR; un
bloque puede cortar un finder o una zona de datos de forma arbitraria.

**Ejemplo:**

```text
Original        Bloques borrados
#######         #######
#.....#         #.....#
#.###.#         #.....#
#.....#         ###....
#######         ###....
```

## 5. Combinaciones

La escalera agrega un componente por paso, con un nivel moderado fijado de
antemano. El objetivo es medir el efecto acumulativo del deterioro y comparar
cada paso con el anterior.

| Paso | Componentes activos | Nivel fijo |
|---|---|---|
| 1 | `aleatorio` | 5% de bits |
| 2 | + `esquina` | 9 × 9 |
| 3 | + `finder` | un finder de 7 × 7 |
| 4 | + `rayones` | 2 líneas |
| 5 | + `manchas` | 1 disco de radio 5 |
| 6 | + `bloques` | 10% de bloques |

Reglas de aplicación:

1. Se parte del patrón original.
2. Se aplica primero el ruido aleatorio, si está activo.
3. Se aplican después los componentes estructurales en el orden `esquina`,
   `finder`, `rayones`, `manchas`, `bloques`.
4. El borrado estructural tiene prioridad sobre un bit invertido en la misma
   posición.
5. Cada componente usa un flujo RNG derivado de la semilla y del nombre del
   componente, para que agregar un componente no cambie los anteriores.
6. Se registra la lista de componentes activos, sus parámetros y la fracción
   afectada final.

Los niveles de la escalera se fijan antes de ver los resultados. Si la escalera
satura la exactitud en un paso temprano, se reporta la saturación; no se cambian
los niveles para forzar una curva más linda.

## 6. Niveles y comparabilidad

- Cada modelo tiene su parámetro nominal: fracción de bits, lado del cuadrado,
  cantidad de finders, cantidad de líneas, radio de la mancha o fracción de
  bloques.
- La figura específica de cada modelo usa el parámetro nominal en el eje x.
- La figura global usa la fracción afectada real, agrupada en intervalos de 5%.
- Los intervalos propuestos son: 0–5%, 5–10%, 10–15%, 15–20%, 20–25%, 25–30%,
  30–35%, 35–40%, 40–45% y 45–50%.
- Si un modelo no tiene consultas en un intervalo, la celda del mapa de calor
  queda vacía.
- Las combinaciones usan el número de paso en el eje x y anotan la fracción
  afectada real.

## 7. Métricas de error

Para cada consulta se registra:

| Métrica | Definición |
|---|---|
| `tipo_resultado` | `almacenado_correcto`, `almacenado_incorrecto`, `espurio`, `ciclo` o `no_converge` |
| `exactitud` | 1 si el estado final es exactamente el patrón de origen; 0 si no |
| `tasa_error` | `1 - exactitud` |
| `hamming_inicial_origen` | bits distintos entre la consulta y el origen |
| `hamming_final_origen` | bits distintos entre el estado final y el origen |
| `hamming_final_normalizado` | `hamming_final_origen / N` |
| `fraccion_afectada` | `hamming_inicial_origen / N` |
| `barridos` | barridos completos hasta la convergencia |
| `cambios_totales` | cantidad de bits que cambiaron durante la recuperación |
| `delta_energia` | `energia_final - energia_inicial` |
| `delta_energia_por_neurona` | `delta_energia / N` |
| `monotona` | la energía no aumenta entre barridos consecutivos |

Composición de errores:

- **correcto:** el estado final coincide con el patrón de origen;
- **incorrecto:** el estado final coincide con otro patrón almacenado;
- **espurio:** el estado final es estable pero no es ninguno de los cinco;
- **ciclo:** la dinámica entra en un ciclo;
- **no converge:** se alcanza `max_barridos`.

La comparación entre Hebb y Personnaz usa exactitud, Hamming y barridos. La
energía absoluta no se compara entre reglas porque la matriz de pesos cambia; se
analiza dentro de cada regla con `E/N` y `ΔE/N`.

## 8. Figuras

Todas las figuras se guardan y se regeneran desde los CSV con
`src/ej2/hopfield-figures.py`.

### 8.1 Por modelo

En `resultados/ej2/hopfield/qr_37x37/ruido/<modelo>/figuras/`:

- `fig_exactitud.png`
- `fig_composicion_errores.png`
- `fig_iteraciones.png`
- `fig_energia.png`
- `fig_hamming.png`
- `fig_confusion.png`
- `fig_exactitud_por_patron.png`
- `fig_boxplot_barridos.png`
- `fig_afectacion_real.png`

### 8.2 Combinaciones

En cada paso, las mismas figuras principales. En
`ruido/combinaciones/consolidado/`:

- `fig_combinaciones_exactitud.png`
- `fig_combinaciones_iteraciones.png`
- `fig_combinaciones_energia.png`
- `fig_combinaciones_hamming.png`
- `fig_mapa_calor_exactitud_combinaciones.png`

### 8.3 Globales

En `ruido/consolidado/figuras/`:

- `fig_exactitud_global.png`
- `fig_iteraciones_global.png`
- `fig_energia_global.png`
- `fig_hamming_global.png`
- `fig_comparacion_global.png`
- `fig_comparacion_reglas.png`
- `fig_mapa_calor_exactitud.png`
- `fig_ejemplos_qr.png`
- `fig_basin_hamming.png`
- `fig_trayectorias_energia_ejemplo.png`
- `fig_afectacion_real_global.png`

El nivel de referencia para `fig_confusion.png` es el menor nivel en el que la
exactitud media de Personnaz cae por debajo de 0,95. Si no cae, se usa el nivel
central. Es una decisión de presentación.

## 9. Reproducibilidad

- Semillas: `[0, 1, 2]`.
- Repeticiones por patrón: 5.
- Patrones: los 5 QR.
- Dinámica: asincrónica secuencial, `max_barridos = 100`.
- Cada consulta guarda: modelo, nivel, parámetros efectivos, semilla,
  repetición, patrón de origen, fracción afectada y resultado.
- La misma consulta ruidosa se usa para Hebb y Personnaz.
- No se guardan las matrices completas de estado de todas las consultas; solo
  ejemplos seleccionados para las figuras cualitativas.
- La configuración resuelta de cada corrida se guarda en `config_usada.toml`.
- Las figuras se pueden regenerar sin volver a ejecutar la red.

## 10. Decisiones tomadas

1. Se ejecutan los seis modelos individuales: `aleatorio`, `esquina`, `finder`,
   `rayones`, `manchas` y `bloques`.
2. Se ejecuta una escalera de combinaciones que agrega un componente por paso.
3. El ruido aleatorio invierte bits; los modelos estructurales borran a `-1`.
4. La esquina y el finder se eligen al azar por repetición y se registran.
5. Los rayones, manchas y bloques se eligen al azar y se registran.
6. Se usan tres semillas, cinco repeticiones y los cinco patrones.
7. Los niveles se fijan a priori y no se ajustan con los resultados.
8. Cada modelo se implementa, ejecuta y revisa por separado, con una pausa
   después de cada uno.
9. La escalera de combinaciones se ejecuta completa y se revisa al final.
10. Todas las figuras se guardan; las principales son exactitud, composición de
    errores, iteraciones, energía, Hamming, comparación global y mapa de calor
    de exactitud.
11. La estructura de salida es por modelo, con consolidación posterior.
12. No hay split de entrenamiento y test; los niveles de ruido son parámetros
    de evaluación fijados a priori.

## 11. Preguntas abiertas

No quedan decisiones de modelado pendientes para empezar. Durante la ejecución
solo se resuelven cuestiones de presentación:

- qué nivel de referencia usar para las matrices de confusión, según la regla
  definida en la sección 8.3;
- qué figuras conviene llevar al informe final, sin dejar de guardar todas;
- si alguna figura global necesita una escala logarítmica o un recorte de ejes
  para ser legible.

Si aparece una decisión que cambie los modelos, los niveles o la métrica
principal, se para y se consulta antes de continuar.
