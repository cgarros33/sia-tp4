# Decisiones del proyecto

> Registro vivo de las decisiones del TP4. La idea es que cualquiera del grupo
> pueda leer este archivo y entender el estado actual sin revisar todo el código.

## Estado actual

- TP4 — Ejercicio 1.1: Exploración y representación (EDA + PCA).
- El plan de 1.1 está en `docs/plan/ej1-1.md`.
- El script de 1.1 es `src/ej1/eda-pca.py`.
- Los resultados de 1.1 se guardan en `resultados/ej1/`.
- Ejercicio 1.2: Red de Kohonen.
- El plan de 1.2 está en `docs/plan/ej1-2.md`.
- Etapa actual: 1.2 implementado; `sweep1` y `sweep2` archivados, `sweep3`,
  `sweep4` y `sweep5` ejecutados con grilla fija, y las tres corridas finales
  (`kohonen_k3/`, `kohonen_k4/`, `kohonen_k5/`) terminadas. Queda la
  interpretación de resultados con el grupo, fuera de los archivos del proyecto.
- Ejercicio 1.3: Índice de Oja.
- El plan de 1.3 está en `docs/plan/ej1-3.md`.
- Etapa actual: 1.3 implementado y ejecutado; resultados en
  `resultados/ej1/oja/`.
- Los tests actuales pasan con `python3 -m pytest -q tests/ej1`.

## Datos

- `datos/creadores.csv` tiene 50 filas y 33 columnas.
- 2 columnas son identificadores: `creator_id` y `name`.
- 31 columnas son variables numéricas.
- No hay nulos, IDs duplicados ni columnas constantes.
- Los datos de entrada son de solo lectura: ningún script los modifica.

## Variables para el PCA

- Se excluyen `price_ars`, `estimated_argentina_reach`,
  `total_engagement_per_1000` y `organic_posts_pct`.
- Motivos:
  - `price_ars` es un duplicado exacto de `price_usd × 1500`.
  - `estimated_argentina_reach` es una variable derivada.
  - `total_engagement_per_1000` es la suma exacta de las cuatro métricas de
    engagement.
  - `organic_posts_pct` es el complemento exacto de `sponsored_posts_pct`.
- Se conservan las cuatro categorías de tópico y las cuatro de contenido. Se
  prefiere mantenerlas completas para que el problema se entienda a primera
  vista.
- Quedan 27 variables para el PCA.
- La matriz de correlación tiene rango 25 porque los grupos de tópicos y de
  contenido suman 100. Eso genera dos autovalores nulos. El script los detecta,
  los informa y no los usa para elegir componentes. Las cargas de las 27
  variables se conservan para interpretar el problema.

## PCA

- Se estandarizan las variables (media 0, desvío muestral 1) y se usa la matriz
  de correlación porque las escalas son muy distintas.
- Los autovalores y autovectores se calculan con `numpy.linalg.eigh`.
- No se usa scikit-learn ni otra librería de machine learning.
- El umbral de varianza se pasa por parámetro:
  `--varianza 0.90`, `--varianza 0.95` o `--varianza 0.99`.
- PC1 se orienta para que `followers` tenga carga positiva. Las demás
  componentes se orientan para que su carga de mayor valor absoluto sea
  positiva. El signo es arbitrario, pero queda fijo para que los gráficos sean
  reproducibles.

## Resultados

- Umbral 90%: 10 componentes, 91,53% de varianza acumulada.
- Umbral 95%: 13 componentes, 95,63% de varianza acumulada.
- Umbral 99%: 18 componentes, 99,11% de varianza acumulada.
- El umbral por defecto del script es 95%. Queda pendiente elegir el umbral
  definitivo para el informe.
- PC1 (49,71%): tamaño, alcance y profesionalización. Correlaciona
  aproximadamente 0,87 con `followers` y con `price_usd`.
- PC2 (9,19%): arte/showcase contra audiencia argentina/perros. Correlaciona
  0,63 con `topic_art_pct`, -0,57 con `argentina_audience_pct` y -0,41 con
  `topic_dogs_pct`.
- PC3 (7,00%): publicaciones patrocinadas y fútbol. Correlaciona 0,73 con
  `sponsored_posts_pct` y 0,66 con `topic_football_pct`.
- PC4 (5,83%): maquillaje contra fútbol. Correlaciona 0,64 con
  `topic_makeup_pct` y -0,59 con `topic_football_pct`.
- Advertencia para la parte 3: PC1 se parece mucho a `followers`; un índice
  basado solo en PC1 no necesariamente mide engagement por dólar.
- Los chequeos numéricos dan residuos del orden de `1e-15`, así que la
  descomposición y los scores son consistentes.

## Ejercicio 1.2 — Red de Kohonen

### Alcance

- El plan completo está en `docs/plan/ej1-2.md`.
- El motor será `src/ej1/kohonen.py` y el barrido
  `src/ej1/kohonen-sweep.py`.
- Las configuraciones serán `src/ej1/kohonen.toml` y
  `src/ej1/kohonen-sweep.toml`.
- No se registra interpretación de perfiles de creadores en archivos del
  proyecto. Las tablas, figuras y resúmenes contienen configuración y
  resultados numéricos; la interpretación se discute aparte con el grupo.

### Datos y preprocesamiento

- Se usan las mismas 27 variables numéricas que en el PCA de 1.1.
- Se excluyen `price_ars`, `estimated_argentina_reach`,
  `total_engagement_per_1000` y `organic_posts_pct` por ser duplicado exacto,
  variable derivada, suma exacta y complemento exacto, respectivamente.
- Se conservan completas las cuatro categorías de tópico y las cuatro de
  contenido, igual que en 1.1.
- Se usan los 50 creadores para entrenamiento y selección de hiperparámetros.
  El ejercicio es no supervisado y no tiene conjunto de test. Esta es una
  excepción documentada al punto 2 de `CLAUDE.md`.
- Se estandariza con media 0 y desvío muestral 1 (`ddof=1`), ajustado sobre los
  datos de entrenamiento (los 50 creadores en esta corrida).
- Los outliers se conservan, como en 1.1.
- No se reduce dimensionalidad con PCA para la entrada del SOM: la consigna
  pide usar todas las variables. El PCA queda solo como comparación posterior.

### Modelo

- SOM online: un registro por iteración, épocas mezcladas.
- Grilla de salida `k × k`, rectangular o hexagonal.
- BMU:
  - `euclidiana`: `argmin_j ||z_p − w_j||₂`, criterio por defecto.
  - `exponencial`: se normalizan a norma unitaria el vector de entrada y cada
    vector de pesos, y se elige `argmax_j exp(-||z̃_p − w̃_j||²)`, siguiendo
    la indicación de normalización de las pp. 121-124.
- La QE, el TE y la matriz U usan siempre distancia euclidiana, para que las
  métricas sean comparables entre valores de `distancia`.
- Distancia sobre la grilla:
  - Rectangular: distancia euclidiana. `R = 1` incluye los 4 vecinos axiales;
    `R = sqrt(2)` incluye los 8 vecinos.
  - Hexagonal: distancia de cubo. `R = 1` incluye los 6 vecinos.
  - La vecindad se define con `d_grilla <= R(t)`.
- Regla de Kohonen:
  `w_j ← w_j + η(t)(z_p − w_j)` para los vecinos de la BMU; el resto no cambia.
- Inicialización:
  - `uniforme`: uniforme por variable en `[min, max]` de los datos de
    entrenamiento.
  - `muestras`: muestras aleatorias del conjunto de entrenamiento, con
    reposición.
- Se usan flujos RNG separados para inicialización y orden de muestras,
  derivados de la semilla, para que el orden de muestras sea el mismo entre
  candidatos con la misma semilla.
- Radio: `R(0) = R0`, `R(T−1) = R_f = 1.0`. Esquemas `constante`, `lineal`,
  `exponencial` e `inversa`.
- Tasa de aprendizaje: `η(0) = η0`, `η(T−1) = η_f = 0.001`. Esquemas `lineal`,
  `exponencial` e `inversa`.
- Métricas: QE media, error topográfico (TE), unidades muertas, ocupación y QE
  local por neurona.
- Matriz U: distancia euclidiana media en el espacio de pesos entre cada
  neurona y sus vecinos inmediatos. Se usan 4 vecinos en rectangular y 6 en
  hexagonal. Los bordes y esquinas usan los vecinos disponibles.
- La matriz U no está definida en las pp. 121-124; se adopta esta definición
  estándar.

### Hiperparámetros

La grilla no se barre. Se fija en `3`, `4` y `5`, y se corre un barrido
secuencial completo para cada tamaño (`sweep3/`, `sweep4/`, `sweep5/`). Para
cada `k`, el orden es:

1. `topologia`
2. `distancia`
3. `inicializacion`
4. `radio_inicial_relativo`
5. `decaimiento_radio`
6. `eta_inicial`
7. `decaimiento_eta`
8. `iteraciones`

| Hiperparámetro | Valores a probar | Neutro |
|---|---|---|
| `grillas_fijas` | `3, 4, 5` | — |
| `topologia` | `rectangular, hexagonal` | `rectangular` |
| `distancia` | `euclidiana, exponencial` | `euclidiana` |
| `inicializacion` | `uniforme, muestras` | `muestras` |
| `radio_inicial_relativo` | `0.3k, 0.4k, 0.5k, 0.75k, 1.0k, 1.5k` | `0.5k` |
| `decaimiento_radio` | `constante, lineal, exponencial, inversa` | `lineal` |
| `eta_inicial` | `0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9` | `0.5` |
| `decaimiento_eta` | `lineal, exponencial, inversa` | `lineal` |
| `iteraciones` | `5000, 10000, 15000, 25000` | `10000` |

- `radio_final = 1.0` y `eta_final = 0.001` quedan fijos por la teoría.
- Para `k = 3`, el candidato `radio_inicial_relativo = 0.3` produce
  `R0 = 0.9 < R_f = 1.0`, así que se omite en esa corrida. La lista usada queda
  registrada en el `config_barrido.toml` de `sweep3/`.
- Las listas de valores viven en `src/ej1/kohonen-sweep.toml`, no en el código.
  Si un óptimo queda en el borde de una lista, se amplía la lista y se vuelve a
  correr ese paso sin tocar el código.
- La regla del codo del 5% se retiró y su código se eliminó. En `sweep1` y
  `sweep2` eligió el máximo `k` y produjo mapas cada vez más dispersos. Con la
  grilla fija por corrida, no hay excepción para `grilla`.

### Selección de hiperparámetros

- Para todos los hiperparámetros restantes se elige el menor QE medio sobre las
  semillas.
- No hay excepción para `grilla`: cada corrida usa un `k` fijo de
  `grillas_fijas`.
- Se usan 3 semillas (`[0, 1, 2]`), las mismas para todos los candidatos, y se
  reporta media ± desvío muestral.
- Cada paso incluye el valor neutro vigente entre sus candidatos, para medir si
  el cambio mejora.
- Si el valor elegido queda en el borde de la lista, el script lo advierte y se
  decide si ampliar el rango.

### Acoplamientos fuertes

- Después del barrido secuencial completo se prueban solo estos dos pares:
  - `(eta_inicial, decaimiento_eta)`
  - `(radio_inicial_relativo, decaimiento_radio)`
- Cada par se prueba como producto cartesiano, partiendo de la configuración
  óptima secuencial, con las mismas 3 semillas.
- El resultado conjunto se adopta solo si su QE medio es menor que el de la
  configuración vigente. Si no mejora, se conservan los valores secuenciales.
- Se documentan ambos intentos, la QE base, la mejor QE conjunta, los valores
  elegidos y si se adoptaron.
- No se prueban otros pares.

### Visualizaciones del barrido

- Se generan las opciones A y B dentro de cada `sweep{k}/`, cada una en su
  carpeta:
  - `resultados/ej1/kohonen_sweep/sweep{k}/figuras/opcion_a/`: una figura por
    paso más `fig_progreso.png`. El valor elegido se marca subrayando su
    etiqueta del eje x, sin línea vertical.
  - `resultados/ej1/kohonen_sweep/sweep{k}/figuras/opcion_b/`: dashboard
    enfocado en legibilidad, con los 8 pasos secuenciales y otro dashboard para
    los acoplamientos.
- La opción C (boxplots por paso) queda desactivada por legibilidad. Su código
  se conserva, pero no se genera en el barrido real.
- No se genera la opción D (heatmap global), porque cada paso secuencial tiene
  una configuración base distinta y las corridas no son directamente
  comparables en una única matriz.
- Los CSV quedan en la carpeta de cada corrida (`sweep1/`, `sweep2/` o la raíz
  de una corrida nueva). Las dos carpetas de figuras permiten comparar y elegir
  cuál va a la presentación.

### Resultados finales por tamaño

- Se corrió el mejor `config_optima.toml` de `sweep3`, `sweep4` y `sweep5`.
- Salidas: `resultados/ej1/kohonen_k3/`, `kohonen_k4/` y `kohonen_k5/`.
- Para cada tamaño se generan tres figuras:
  - `fig_neuronas.png`: planos de componentes, un heatmap `k × k` por variable,
    27 paneles en una disposición 5×6, con escala de color compartida.
  - `fig_distribucion.png`: heatmap de cantidad de creadores por neurona y, al
    lado, un listado agrupado y legible de creadores por neurona.
  - `fig_matriz_u.png`: heatmap de la matriz U con valores y cantidad de
    creadores por neurona.
- No se genera la figura opcional de comparación con PC1-PC2. El biplot
  existente en `resultados/ej1/pca_0.95/fig_biplot.png` queda disponible para la
  comparación manual durante la interpretación.
- Las figuras usan la semilla cuyo QE está más cerca de la media; la semilla
  elegida se registra en `resumen.txt` y `config_usada.toml`.
- La comparación final se hace entre los tres tamaños, no solo dentro de uno.

### Salidas

- Primer barrido archivado: `resultados/ej1/kohonen_sweep/sweep1/`.
- Segundo barrido archivado: `resultados/ej1/kohonen_sweep/sweep2/`.
- Nuevos barridos de grilla fija: `sweep3/`, `sweep4/` y `sweep5/`.
- Corridas finales por tamaño: `resultados/ej1/kohonen_k3/`,
  `kohonen_k4/` y `kohonen_k5/`.
- Corridas neutras manuales: `resultados/ej1/kohonen_base/`.
- Cada barrido guarda `config_barrido.toml`, `config_optima.toml`,
  `registro.txt`, `resumen_pasos.csv`, los CSV por paso, `acoplamientos/` y
  las dos carpetas de figuras habilitadas (A y B).
- Cada corrida final guarda `config_usada.toml`, `normalizacion.csv`,
  `pesos.csv`, `neuronas.csv`, `asignaciones.csv`, `matriz_u.csv`,
  `metricas.csv`, `resumen.txt` y las tres figuras.

### Reproducibilidad

- Cada corrida usa una semilla explícita tomada de la configuración.
- La configuración resuelta y las semillas se guardan con los resultados.
- Cada resultado presentado se reporta como media ± desvío sobre al menos 3
  semillas.
- Todos los datos necesarios para regenerar las figuras quedan en los CSV
  guardados.
- No se usan bibliotecas de machine learning; solo NumPy, pandas, matplotlib y
  la biblioteca estándar.

### Resultados del barrido (sweep1)

- Se ejecutó el barrido secuencial completo con 3 semillas y los valores de
  `src/ej1/kohonen-sweep.toml`.
- Salida: `resultados/ej1/kohonen_sweep/sweep1/`.
- Tiempo total del barrido: 734,34 s.

| Paso | Parámetro | Valor elegido | QE media | QE desvío |
|---:|---|---|---:|---:|
| 1 | `grilla` | `10` | 1,414228 | 0,065676 |
| 2 | `topologia` | `rectangular` | 1,414228 | 0,065676 |
| 3 | `distancia` | `exponencial` | 1,357106 | 0,203866 |
| 4 | `inicializacion` | `muestras` | 1,357106 | 0,203866 |
| 5 | `radio_inicial_relativo` | `0.5` | 1,357106 | 0,203866 |
| 6 | `decaimiento_radio` | `inversa` | 0,532888 | 0,129570 |
| 7 | `eta_inicial` | `0.9` | 0,475524 | 0,221087 |
| 8 | `decaimiento_eta` | `lineal` | 0,475524 | 0,221087 |
| 9 | `iteraciones` | `80000` | 0,376768 | 0,109013 |

Acoplamientos fuertes:

| Par | Mejor valor | QE media | QE base | ¿Adoptado? |
|---|---|---:|---:|---|
| `eta_inicial` x `decaimiento_eta` | `0.9` x `lineal` | 0,376768 | 0,376768 | No |
| `radio_inicial_relativo` x `decaimiento_radio` | `0.5` x `inversa` | 0,376768 | 0,376768 | No |

Los dos acoplamientos devolvieron la misma combinación que ya estaba vigente,
así que no se modificó la configuración secuencial.

### Revisión de bordes del barrido

- `grilla = 10` es el máximo de la lista. El promedio de unidades muertas es
  58,33 de 100 neuronas y la ocupación media es 0,4167. El codo del 5% eligió
  `10` porque el mínimo es 1,414228, el límite es 1,484939 y `k = 8` tiene QE
  1,868502, por encima del límite. La lista no se extendió todavía.
- `eta_inicial = 0.9` es el máximo de la lista, con QE 0,475524 y desvío
  0,221087.
- `iteraciones = 80000` es el máximo de la lista, con QE 0,376768 y desvío
  0,109013.
- `radio_inicial_relativo = 0.5` es el mínimo de la lista. El acoplamiento
  probó 0,5; 0,75; 1,0 y 1,5, y 0,5 con `inversa` fue el mejor de esos valores.
  No se probó 0,25.
- Las listas de `topologia`, `distancia`, `inicializacion`,
  `decaimiento_radio` y `decaimiento_eta` son categóricas y exhaustivas; no
  requieren extensión.
- Se decidió ampliar `grilla` a `[6, 7, 8, 9, 10, 11, 12, 13, 14]` y agregar
  `0.3` y `0.4` a `radio_inicial_relativo`. Ese barrido se ejecutó en la raíz de
  `resultados/ej1/kohonen_sweep/` y luego se archivó en `sweep2/`; `sweep1/`
  queda como archivo anterior.
- `eta_inicial` e `iteraciones` se dejan como están por ahora; el próximo
  barrido vuelve a decidirlos.
- `config_optima.toml` de `sweep1/` queda como configuración provisoria del
  primer barrido y no se usa todavía para la corrida final.

### Resultados del barrido (sweep2)

- Se ejecutó el barrido completo con 3 semillas, grilla `6..14`, radio
  `0.3..1.5`, barra de progreso y figuras A/B.
- Salida: `resultados/ej1/kohonen_sweep/sweep2/`.
- Tiempo total: 593,59 s.

| Paso | Parámetro | Valor elegido | QE media | QE desvío |
|---:|---|---|---:|---:|
| 1 | `grilla` | `14` | 0,406708 | 0,099716 |
| 2 | `topologia` | `hexagonal` | 0,253904 | 0,039228 |
| 3 | `distancia` | `euclidiana` | 0,253904 | 0,039228 |
| 4 | `inicializacion` | `uniforme` | 0,247179 | 0,072811 |
| 5 | `radio_inicial_relativo` | `0.3` | 0,113008 | 0,101977 |
| 6 | `decaimiento_radio` | `inversa` | 0,013054 | 0,022610 |
| 7 | `eta_inicial` | `0.9` | 0,000000 | 0,000000 |
| 8 | `decaimiento_eta` | `lineal` | 0,000000 | 0,000000 |
| 9 | `iteraciones` | `80000` | 0,000000 | 0,000000 |

Acoplamientos fuertes:

| Par | Mejor valor | QE media | QE base | ¿Adoptado? |
|---|---|---:|---:|---|
| `eta_inicial` x `decaimiento_eta` | `0.9` x `lineal` | 0,000000 | 0,000000 | No |
| `radio_inicial_relativo` x `decaimiento_radio` | `0.3` x `inversa` | 0,000000 | 0,000000 | No |

Revisión técnica de sweep2:

- La configuración final tiene `grilla = 14`, es decir, 196 neuronas para 50
  creadores.
- La ocupación media es 0,2551: se usan 50 neuronas y quedan 146 muertas.
- La QE media final es 0. La combinación de 196 neuronas, `uniforme`,
  `eta_inicial = 0.9` e `inversa` termina con una neurona dedicada a cada
  muestra; no es un mapa útil para analizar regiones.
- El codo del 5% volvió a elegir el máximo de la lista: el mínimo es 0,406708,
  el límite es 0,427043 y `k = 13` tiene QE 0,640873, por encima del límite.
- `radio_inicial_relativo = 0.3` es el mínimo de la lista, `eta_inicial = 0.9`
  es el máximo e `iteraciones = 80000` es el máximo.
- Comparación con `sweep1`: QE 0,376768; 54,33 unidades muertas de 100;
  ocupación 0,4567. `sweep1` también era disperso, pero no degenerado.
- `config_optima.toml` de `sweep2/` es provisorio y no se usa para la corrida
  final.
- Conclusión metodológica: un `k` grande no es un objetivo válido. Reduce la QE
  porque asigna una neurona casi exacta a cada muestra, deja la mayoría de las
  neuronas muertas y no produce una categorización útil.
- Por eso `sweep2/` queda archivado y el próximo enfoque fija la grilla en
  `k = 3, 4, 5` y corre un barrido completo para cada tamaño.

### Resultados de barridos con grilla fija

Se corrieron los tres barridos completos en una sola invocación. Salidas:
`resultados/ej1/kohonen_sweep/sweep3/`, `sweep4/` y `sweep5/`. Tiempo total:
92,97 s.

| k | Neuronas | Configuración final | QE media | QE desvío | TE | Muertas | Ocupación |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 9 | rectangular, euclidiana, uniforme, radio 1.5, inversa, eta 0.3, iteraciones 5000 | 3,478953 | 0,007685 | 0,160000 | 0 / 9 | 1,0000 |
| 4 | 16 | rectangular, euclidiana, uniforme, radio 1.0, inversa, eta 0.9, iteraciones 10000 | 3,173119 | 0,013589 | 0,046667 | 1 / 16 | 0,9375 |
| 5 | 25 | rectangular, euclidiana, uniforme, radio 0.3, lineal, eta 0.7, exponencial, iteraciones 15000 | 2,807907 | 0,063630 | 0,033333 | 6,33 / 25 | 0,7467 |

Detalle de acoplamientos:

- `k=3`: `eta_inicial` y `decaimiento_eta` no se adoptaron. El acoplamiento
  `radio_inicial_relativo=1.5` con `decaimiento_radio=inversa` sí se adoptó:
  QE 3,478953 contra 3,486754 secuencial.
- `k=4`: `eta_inicial` y `decaimiento_eta` no se adoptaron. El acoplamiento
  `radio_inicial_relativo=1.0` con `decaimiento_radio=inversa` sí se adoptó:
  QE 3,173119 contra 3,180673 secuencial.
- `k=5`: se adoptaron ambos acoplamientos:
  `eta_inicial=0.7` con `decaimiento_eta=exponencial` (2,830909 contra
  2,832545) y luego `radio_inicial_relativo=0.3` con
  `decaimiento_radio=lineal` (2,807907 contra 2,830909).

Observaciones técnicas:

- Para `k=3` se omitió `radio_inicial_relativo=0.3` porque `R0=0.9 < R_f=1.0`.
  El `config_barrido.toml` de `sweep3/` registra la lista filtrada.
- Los tres mapas usan todas o casi todas sus neuronas; no reaparece el caso
  degenerado de QE 0.
- Las figuras generadas son A y B; no se generó `opcion_c/`.

### Configuraciones finales elegidas

| k | Configuración final | QE media | QE desvío | TE | Muertas | Ocupación |
|---:|---|---:|---:|---:|---:|---:|
| 3 | `sweep3/config_optima.toml` | 3,478953 | 0,007685 | 0,160000 | 0 / 9 | 1,0000 |
| 4 | `sweep4/config_optima.toml` | 3,173119 | 0,013589 | 0,046667 | 1 / 16 | 0,9375 |
| 5 | `sweep5/config_optima.toml` | 2,807907 | 0,063630 | 0,033333 | 6,33 / 25 | 0,7467 |

### Corridas finales ejecutadas

- `sweep3/config_optima.toml` → `resultados/ej1/kohonen_k3/`
  - semilla representativa: `0`
  - QE 3,478953; 0 unidades muertas; ocupación 1,0000
- `sweep4/config_optima.toml` → `resultados/ej1/kohonen_k4/`
  - semilla representativa: `2`
  - QE 3,173119; 1 unidad muerta; ocupación 0,9375
- `sweep5/config_optima.toml` → `resultados/ej1/kohonen_k5/`
  - semilla representativa: `0`
  - QE 2,807907; 8 unidades muertas; ocupación 0,6800
  - La ocupación varía entre semillas (0,68; 0,84; 0,72); la figura usa la
    semilla 0.
- Cada salida contiene `config_usada.toml`, `normalizacion.csv`, `pesos.csv`,
  `neuronas.csv`, `asignaciones.csv`, `matriz_u.csv`, `metricas.csv`,
  `resumen.txt` y las tres figuras finales.
- La comparación entre tamaños se hace con QE, TE, unidades muertas, ocupación,
  distribución de asignaciones y las figuras. La interpretación de perfiles de
  creadores queda fuera de los archivos del proyecto.

### Pendientes de 1.2

- Fase B: completada.
- Fase C: completada.
- Fase D: completada.
- Fase E.1: completada.
- Fases E.2 a E.5: completadas.
- Fase E.6: barrido sweep2 ejecutado, revisado y archivado en `sweep2/`;
  completada.
- Paso 2: implementación de grilla fija y eliminación del codo; completada.
- Paso 3: completado; `sweep3/`, `sweep4/` y `sweep5/` ejecutados.
- Paso 4: completado; plan de corrida final por tamaño registrado.
- Fase F: completada; las tres corridas finales se ejecutaron.
- Pendiente: interpretación de resultados con el grupo, fuera de los archivos
  del proyecto.

## Ejercicio 1.3 — Índice de Oja

### Alcance

- El plan completo está en `docs/plan/ej1-3.md`.
- El script es `src/ej1/oja.py` y la configuración `src/ej1/oja.toml`.
- Los tests están en `tests/ej1/test_oja.py`.
- Las salidas están en `resultados/ej1/oja/`.
- No se implementa la red de Sanger: la consigna pide la regla de Oja.

### Datos y preprocesamiento

- Se usan los 50 creadores y las mismas 27 variables numéricas que en 1.1 y
  1.2.
- Se excluyen `price_ars`, `estimated_argentina_reach`,
  `total_engagement_per_1000` y `organic_posts_pct`, por las mismas razones que
  en 1.1.
- No hay conjunto de test: el ejercicio es no supervisado y todos los creadores
  se usan para entrenar y analizar. Es la misma excepción documentada que en
  1.2 al punto 2 de `CLAUDE.md`.
- Estandarización z-score con media 0 y desvío muestral 1 (`ddof=1`), ajustada
  sobre los 50 creadores. Los outliers se conservan.
- El índice queda en unidades estandarizadas; su ranking es equivalente al de
  un índice lineal en escala original.

### Modelo

- Un perceptrón lineal con una sola salida y sin sesgo.
- Vector de pesos `w ∈ R^27`.
- Inicialización uniforme en `[0, 1)` por componente, con semilla explícita, y
  normalización a norma unitaria antes de la primera actualización.
- Forma online estándar: una actualización por muestra, con épocas mezcladas y
  operaciones vectorizadas sobre las 27 dimensiones.
- La regla final es la regla normalizada exacta de Oja, ecuaciones 16.7-16.8:

  ```text
  w ← (w + η s z) / ||w + η s z||
  ```

- También se implementa la forma de Taylor de la ecuación 16.9:

  ```text
  w ← w + η s (z - s w)
  ```

  Se corre con `eta_inicial = 0.5` y con `eta_inicial = 0.05`, como
  comparaciones. No se usa para el índice final.
- Horario de aprendizaje: `η(t) = eta0 / (1 + t)` (`inversa`), con
  `eta0 = 0.5` en la corrida final y `t` el índice global de actualización.
- `epocas = 500` (25.000 actualizaciones). Si las curvas no se estabilizan, se
  amplía el presupuesto antes de la corrida final.
- Semillas `[0, 1, 2]`, las mismas para todas las corridas. Cada resultado se
  reporta como media ± desvío muestral.
- Al final se orienta cada vector de pesos para que `followers` tenga carga
  positiva, igual que PC1 en 1.1.

### Control de divergencia y convergencia

- Al final de cada época se corta la corrida si hay valores no finitos, si
  `||w|| > 5.0`, si `wᵀ C w / λ1 > 5.0` o si la razón de varianza es negativa.
- Se registran `divergio`, `epoca_detencion` y `motivo`.
- Una corrida converge si no divergió, si `wᵀ C w / λ1 >= 0.99` y si la
  correlación de Pearson entre el índice de Oja y los scores de PC1 es
  `>= 0.99`.
- Los umbrales quedan en `config_usada.toml` bajo `[control]`.

### Comparación con PC1

- No se recalcula PC1. Se leen los resultados de 1.1:
  - `resultados/ej1/pca_0.95/cargas.csv`
  - `resultados/ej1/pca_0.95/scores.csv`
  - `resultados/ej1/pca_0.95/autovalores.csv`
  - `resultados/ej1/eda/correlaciones.csv`
- Las cargas se reindexan al orden de las 27 variables del modelo y los scores
  se alinean por `creator_id`.
- Se valida que `v1ᵀ C v1 ≈ λ1`, que `var(pc1_scores, ddof=1) ≈ λ1`, que
  `mean(pc1_scores) ≈ 0`, que la diagonal de `C` sea 1 y que `followers` tenga
  carga positiva en PC1.
- Si alguna validación falla, el script se detiene. No hay recomputación de
  respaldo.
- La comparación usa solo:
  - correlación de Pearson entre el índice de Oja y los scores de PC1;
  - razón de varianza `wᵀ C w / λ1`.
- No se usa Spearman, top-k ni coseno. La comparación de cargas se muestra en
  `pesos.csv` y `fig_pesos.png`.

### Correlaciones del índice

- El índice es `I = Σ_j w_j z_j` sobre las 27 variables del modelo.
- Para cualquier variable `k` de las 31 originales:

  ```text
  corr(I, x_k) = (Σ_j w_j C_jk) / sqrt(wᵀ C_model w)
  ```

- Se usa la matriz de correlación de 1.1, sin recalcular correlaciones desde los
  datos crudos.
- Se guardan las 31 correlaciones con una columna `incluida_en_modelo`.
- No se agrega `engagement_por_dolar` en 1.3; queda para la parte 4.

### Sensibilidad

- Para la regla exacta se prueban:
  - `eta_inicial ∈ {0.01, 0.05, 0.1, 0.2, 0.5}`;
  - `decaimiento_eta ∈ {constante, inversa, raiz}`.
- 3 semillas y 500 épocas.
- Se reportan razón de varianza, Pearson con PC1, norma final y cantidad de
  divergencias.
- Es una métrica de robustez; no se elige la configuración final a partir de
  ella.
- Las combinaciones de sensibilidad no generan configuraciones individuales.

### Configuraciones guardadas

- `config_usada.toml`: configuración resuelta de la corrida exacta final.
- `config_regla_taylor_0.5.toml`: solo si la corrida de Taylor con `η0 = 0.5`
  converge.
- `config_regla_taylor_0.05.toml`: solo si la corrida de Taylor con `η0 = 0.05`
  converge.
- Una corrida que diverge o no converge no genera configuración; queda
  registrada en `comparacion_reglas.csv`.

### Salidas

- `resultados/ej1/oja/` con:
  - `config_usada.toml` y, si convergen, las configuraciones de Taylor;
  - `normalizacion.csv`, `pesos.csv`, `indice.csv`;
  - `comparacion_pc1.csv`, `correlaciones_indice.csv`;
  - `convergencia.csv`;
  - `sensibilidad.csv`, `sensibilidad_resumen.csv`;
  - `comparacion_reglas.csv`, `comparacion_reglas_resumen.csv`;
  - `resumen.txt`;
  - las figuras `fig_convergencia.png`, `fig_pesos.png`, `fig_indice.png`,
    `fig_comparacion_pc1.png`, `fig_indice_vs_negocio.png`,
    `fig_sensibilidad.png` y `fig_comparacion_reglas.png`.

### Reproducibilidad

- Cada corrida usa una semilla explícita.
- La configuración resuelta y las semillas se guardan con los resultados.
- Cada resultado presentado se reporta como media ± desvío sobre 3 semillas.
- Todas las figuras se regeneran desde los CSV guardados.
- No se usan bibliotecas de machine learning; solo NumPy, pandas, matplotlib y
  la biblioteca estándar.

### Resultados de 1.3

- Corrida final: regla exacta, `eta_inicial = 0.5`, decaimiento `inversa`,
  500 épocas y semillas `[0, 1, 2]`.
- Las tres semillas convergieron sin divergencia.
- Razón de varianza final: `1.000000 ± 0.000000`.
- Pearson con PC1: `1.000000 ± 0.000000`.
- Norma final de los pesos: `1.000000`.
- Comparación con PC1:
  - Pearson entre el índice y PC1: `1.0000000000`.
  - Razón de varianza: `0.9999999797`.
  - Varianza del índice: `13.4208673970`.
  - Varianza de PC1: `13.4208680017`.
  - `lambda1`: `13.4208676700`.
- El ranking de Oja coincide exactamente con el ranking de PC1; la diferencia
  máxima de ranking es 0.
- Primeros cinco creadores por índice: `@futbolsinvueltas` (6.503237),
  `@galeriacotidiana` (5.896866), `@brillourbano` (5.690295),
  `@crearenreels` (5.544413) y `@golconmate` (5.436186).
- Correlaciones positivas más altas del índice:
  `average_impressions` (0.918112), `average_video_views` (0.912725),
  `average_reach` (0.908672), `trend_posts_pct` (0.900423),
  `video_posts_pct` (0.879608), `short_form_posts_pct` (0.872668),
  `followers` (0.869469) y `price_usd` (0.869361).
- Correlaciones negativas más altas del índice:
  `average_caption_words` (-0.937431), `saves_per_1000` (-0.840787),
  `profile_visits_per_1000` (-0.772786), `comments_per_1000` (-0.766975),
  `average_watch_time_seconds` (-0.729670), `content_tutorial_pct`
  (-0.729167) y `content_review_pct` (-0.667435).
- `total_engagement_per_1000`: 0.354000.
- Tópicos: `topic_football_pct` 0.241210, `topic_makeup_pct` 0.239463,
  `topic_art_pct` 0.026446 y `topic_dogs_pct` -0.486051.
- Observación técnica: el índice queda alineado con tamaño y alcance, y no con
  una medida de eficiencia. Es consistente con PC1 de 1.1 y con la comparación
  entre ambos rankings.
- Regla de Taylor con `eta_inicial = 0.05`: las tres semillas convergieron;
  razón de varianza `0.998613 ± 0.002108` y Pearson `0.999921 ± 0.000129`.
  Su configuración se guardó en `config_regla_taylor_0.05.toml`.
- Regla de Taylor con `eta_inicial = 0.5`: dos de las tres semillas divergieron
  en la época 1 por valores no finitos y la restante convergió. No se guardó
  configuración porque no todas las semillas convergieron.
- En la comparación de reglas, las medias excluyen las semillas divergidas.
- Sensibilidad de la regla exacta:
  - `raiz` converge para todos los `eta_inicial` probados; la razón de varianza
    queda entre 0.996589 y 1.000000 y el Pearson entre 0.999825 y 1.000000.
  - `inversa` converge para `eta_inicial = 0.05, 0.1, 0.2, 0.5`; con `0.01`
    no alcanza el criterio (razón de varianza 0.342282).
  - `constante` no alcanza el criterio en ningún `eta_inicial`; la razón de
    varianza queda entre 0.592862 y 0.960472.
  - El detalle completo está en `sensibilidad_resumen.csv`.
- Salidas completas en `resultados/ej1/oja/`: `config_usada.toml`,
  `config_regla_taylor_0.05.toml`, `normalizacion.csv`, `pesos.csv`,
  `indice.csv`, `comparacion_pc1.csv`, `correlaciones_indice.csv`,
  `convergencia.csv`, `sensibilidad.csv`, `sensibilidad_resumen.csv`,
  `comparacion_reglas.csv`, `comparacion_reglas_resumen.csv`, `resumen.txt` y
  las siete figuras.

### Pendientes de 1.3

- Fase 1: completada; plan y decisiones registrados.
- Fase 2: completada; `tests/ej1/test_oja.py` con 23 tests.
- Fase 3: completada; `src/ej1/oja.py` y `src/ej1/oja.toml`.
- Fase 4: completada; resultados en `resultados/ej1/oja/`.
- Pendiente: interpretación de resultados con el grupo, fuera de los archivos
  del proyecto.

## Salidas y estructura

- EDA: `resultados/ej1/eda/`.
- PCA: `resultados/ej1/pca_<umbral>/`.
- Tests: `tests/ej1/test_eda_pca.py`.
- Estructura completa: `docs/structure.md`.
- Las figuras y tablas de EDA son independientes del umbral. Las de PCA se
  guardan por umbral para no pisar corridas anteriores.

## Calidad

- `python3 check-cicd.py`: crea o reutiliza `.venv`, actualiza dependencias si
  hace falta y corre todos los chequeos. Es la forma recomendada de verificar
  que el CI va a pasar.
- `python3 -m pytest -q`: 53 tests en verde.
- `ruff check .`: sin errores.
- `ruff format --check .`: sin cambios pendientes.
- La configuración de pytest y ruff está en `pyproject.toml`.
- `ruff` está incluido en `requirements.txt`.
- El workflow `.github/workflows/ci.yml` corre lint, formato y tests en cada
  push y pull request.

## Pendientes

- Decidir el umbral definitivo comparando 90%, 95% y 99%.
- Usar estos resultados como base para el resto de los subejercicios.
- Interpretar en el informe qué significa cada componente para las campañas.
