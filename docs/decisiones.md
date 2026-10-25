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
- Etapa actual: 1.2 implementado y ejecutado dos veces. La corrida actual usa
  `PC1` a `PC10` de 1.1 y está en `resultados/ej1/ej1.2-pca10/`; la corrida con
  las 27 variables quedó archivada en `resultados/ej1/ej1.2-full/`. Queda la
  interpretación de resultados con el grupo, fuera de los archivos del proyecto.
- Ejercicio 1.3: Índice de Oja.
- El plan de 1.3 está en `docs/plan/ej1-3.md`.
- Etapa actual: 1.3 implementado y ejecutado. La corrida final usa 5 épocas
  (`resultados/ej1/oja/epocas_5/`); la corrida de 500 épocas quedó archivada
  en `resultados/ej1/oja/epocas_500/`.
- Ejercicio 2.1: Recuperación de códigos QR deteriorados.
- El plan de 2.1 está en `docs/plan/ej2-1.md` y los modelos de ruido en
  `docs/noise.md`.
- Etapa actual: 2.1 implementado y ejecutado (fases 1 a 10); resultados en
  `resultados/ej2/hopfield/qr_37x37/`. La corrida preliminar queda archivada en
  `qr_hebb/`, `qr_pseudoinversa/` y `analisis_qr/`. Queda la interpretación con
  el grupo.
- Ejercicio 2.2: Recomendación (red de Hopfield de 5 × 5).
- El plan de 2.2 está en `docs/plan/ej2-2.md`.
- Etapa actual: 2.2 implementado y ejecutado; resultados en
  `resultados/ej2/hopfield/letras_5x5/`. Queda la interpretación con el grupo.
- Los tests actuales pasan con `python3 -m pytest -q tests/ej1 tests/ej2`.

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
- La corrida actual se configura con `src/ej1/kohonen-pca10.toml` y
  `src/ej1/kohonen-sweep-pca10.toml`. Las configuraciones con todas las
  variables (`src/ej1/kohonen.toml` y `src/ej1/kohonen-sweep.toml`) quedan para
  la corrida archivada.
- El análisis detallado del barrido está en `docs/sweep.md`.
- No se registra interpretación de perfiles de creadores en archivos del
  proyecto. Las tablas, figuras y resúmenes contienen configuración y
  resultados numéricos; la interpretación se discute aparte con el grupo.

### Datos y preprocesamiento

- La corrida actual usa `PC1` a `PC10` de
  `resultados/ej1/pca_0.90/scores.csv` (10 componentes, 91,53% de varianza
  acumulada). El PCA no se recalcula: se leen los resultados de 1.1.
- Los scores se usan tal como salen de 1.1, sin reestandarizar
  (`datos.estandarizar = false`). El PCA ya los centra y así se conserva el peso
  relativo de cada componente. La QE queda en unidades de score de PCA y no es
  comparable con la de la corrida archivada.
- La corrida archivada con todas las variables usaba las 27 variables numéricas
  del PCA de 1.1 y estandarización z-score (`ddof=1`).
- Esas 27 variables excluían `price_ars`, `estimated_argentina_reach`,
  `total_engagement_per_1000` y `organic_posts_pct` por ser duplicado exacto,
  variable derivada, suma exacta y complemento exacto, respectivamente. Los
  scores de PCA ya heredan esa selección.
- Se conservan completas las cuatro categorías de tópico y las cuatro de
  contenido, igual que en 1.1.
- Se usan los 50 creadores para entrenamiento y selección de hiperparámetros.
  El ejercicio es no supervisado y no tiene conjunto de test. Esta es una
  excepción documentada al punto 2 de `CLAUDE.md`.
- Los outliers se conservan, como en 1.1.
- La decisión anterior de no reducir dimensionalidad para la entrada del SOM
  correspondía a la corrida con todas las variables; queda archivada. La
  corrida actual usa 10 componentes por decisión del grupo.

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
  carpeta. Las rutas son relativas a la corrida:
  - `<corrida>/kohonen_sweep/sweep{k}/figuras/opcion_a/`: una figura por paso
    más `fig_progreso.png`. El valor elegido se marca subrayando su etiqueta
    del eje x, sin línea vertical.
  - `<corrida>/kohonen_sweep/sweep{k}/figuras/opcion_b/`: dashboard enfocado en
    legibilidad, con los 8 pasos secuenciales y otro dashboard para los
    acoplamientos.
- `<corrida>` es `resultados/ej1/ej1.2-full/` para la corrida archivada y
  `resultados/ej1/ej1.2-pca10/` para la actual.
- La opción C (boxplots por paso) queda desactivada por legibilidad. Su código
  se conserva, pero no se genera en el barrido real.
- No se genera la opción D (heatmap global), porque cada paso secuencial tiene
  una configuración base distinta y las corridas no son directamente
  comparables en una única matriz.
- Los CSV quedan en la carpeta de cada corrida (`sweep1/`, `sweep2/` o la raíz
  de una corrida nueva). Las dos carpetas de figuras permiten comparar y elegir
  cuál va a la presentación.

### Resultados finales por tamaño (corrida archivada con todas las variables)

- Se corrió el mejor `config_optima.toml` de `sweep3`, `sweep4` y `sweep5` de
  la corrida con las 27 variables.
- Salidas: `resultados/ej1/ej1.2-full/kohonen_k3/`, `kohonen_k4/` y `kohonen_k5/`.
- Para cada tamaño se generan tres figuras:
  - `fig_neuronas.png`: planos de componentes, un heatmap `k × k` por variable
    (27 paneles en la corrida archivada, 10 en la actual), con escala de color
    compartida.
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

- Barridos archivados con todas las variables:
  `resultados/ej1/ej1.2-full/kohonen_sweep/sweep1/` a `sweep5/`.
- Corridas finales archivadas con todas las variables:
  `resultados/ej1/ej1.2-full/kohonen_k3/`, `kohonen_k4/` y `kohonen_k5/`.
- Barridos actuales con `PC1` a `PC10`:
  `resultados/ej1/ej1.2-pca10/kohonen_sweep/sweep3/`, `sweep4/` y `sweep5/`.
- Corridas finales actuales con `PC1` a `PC10`:
  `resultados/ej1/ej1.2-pca10/kohonen_k3/`, `kohonen_k4/` y `kohonen_k5/`.
- Configuración actual: `src/ej1/kohonen-pca10.toml` y
  `src/ej1/kohonen-sweep-pca10.toml`.
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
- Salida: `resultados/ej1/ej1.2-full/kohonen_sweep/sweep1/`.
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
  `resultados/ej1/ej1.2-full/kohonen_sweep/` y luego se archivó en `sweep2/`; `sweep1/`
  queda como archivo anterior.
- `eta_inicial` e `iteraciones` se dejan como están por ahora; el próximo
  barrido vuelve a decidirlos.
- `config_optima.toml` de `sweep1/` queda como configuración provisoria del
  primer barrido y no se usa todavía para la corrida final.

### Resultados del barrido (sweep2)

- Se ejecutó el barrido completo con 3 semillas, grilla `6..14`, radio
  `0.3..1.5`, barra de progreso y figuras A/B.
- Salida: `resultados/ej1/ej1.2-full/kohonen_sweep/sweep2/`.
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

### Resultados de barridos con grilla fija (corrida archivada con todas las variables)

Se corrieron los tres barridos completos en una sola invocación. Salidas:
`resultados/ej1/ej1.2-full/kohonen_sweep/sweep3/`, `sweep4/` y `sweep5/`. Tiempo total:
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

### Configuraciones finales elegidas (corrida archivada con todas las variables)

| k | Configuración final | QE media | QE desvío | TE | Muertas | Ocupación |
|---:|---|---:|---:|---:|---:|---:|
| 3 | `sweep3/config_optima.toml` | 3,478953 | 0,007685 | 0,160000 | 0 / 9 | 1,0000 |
| 4 | `sweep4/config_optima.toml` | 3,173119 | 0,013589 | 0,046667 | 1 / 16 | 0,9375 |
| 5 | `sweep5/config_optima.toml` | 2,807907 | 0,063630 | 0,033333 | 6,33 / 25 | 0,7467 |

### Corridas finales ejecutadas (corrida archivada con todas las variables)

- `sweep3/config_optima.toml` → `resultados/ej1/ej1.2-full/kohonen_k3/`
  - semilla representativa: `0`
  - QE 3,478953; 0 unidades muertas; ocupación 1,0000
- `sweep4/config_optima.toml` → `resultados/ej1/ej1.2-full/kohonen_k4/`
  - semilla representativa: `2`
  - QE 3,173119; 1 unidad muerta; ocupación 0,9375
- `sweep5/config_optima.toml` → `resultados/ej1/ej1.2-full/kohonen_k5/`
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

### Corrida actual con 10 componentes principales

- Entrada: `PC1` a `PC10` de `resultados/ej1/pca_0.90/scores.csv`.
- El PCA no se recalcula y los scores no se reestandarizan
  (`datos.estandarizar = false`).
- Configuración: `src/ej1/kohonen-pca10.toml` y
  `src/ej1/kohonen-sweep-pca10.toml`.
- Salidas: `resultados/ej1/ej1.2-pca10/kohonen_sweep/` y
  `resultados/ej1/ej1.2-pca10/kohonen_k3/`, `kohonen_k4/` y `kohonen_k5/`.
- El barrido completo de los tres tamaños tardó 215,88 s.

| k | Neuronas | Configuración final | QE media | QE desvío | TE | Muertas | Ocupación |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 9 | rectangular, euclidiana, uniforme, radio 0.4, constante, eta 0.3, exponencial, iteraciones 10000 | 3,109693 | 0,015957 | 0,140000 | 0 / 9 | 1,0000 |
| 4 | 16 | rectangular, exponencial, muestras, radio 1.5, exponencial, eta 0.5, lineal, iteraciones 10000 | 2,806541 | 0,022015 | 0,140000 | 0 / 16 | 1,0000 |
| 5 | 25 | rectangular, euclidiana, muestras, radio 0.3, lineal, eta 0.3, lineal, iteraciones 10000 | 2,450829 | 0,022026 | 0,040000 | 3 / 25 | 0,8800 |

Detalle de acoplamientos:

- `k=3`: se adoptó `eta_inicial=0.3` con `decaimiento_eta=exponencial`
  (3,109693 contra 3,127882). El acoplamiento de radio no se adoptó porque
  devolvió la misma QE.
- `k=4`: `eta_inicial` y `decaimiento_eta` no se adoptaron. Se adoptó
  `radio_inicial_relativo=1.5` con `decaimiento_radio=exponencial`
  (2,806541 contra 2,808934).
- `k=5`: no se adoptó ninguno de los dos acoplamientos.

Observaciones técnicas:

- Semilla representativa: `0` para `k=3` y `2` para `k=4` y `k=5`.
- Unidades muertas por semilla: `k=3` 0/0/0; `k=4` 2/0/0; `k=5` 6/4/3.
- Ocupación por semilla: `k=3` 1,00/1,00/1,00; `k=4` 0,875/1,00/1,00;
  `k=5` 0,76/0,84/0,88.
- `k=5` tiene la menor QE y el menor TE, pero también la mayor cantidad de
  unidades muertas.
- `radio_inicial_relativo=1.5` para `k=4` es el máximo de la lista y
  `radio_inicial_relativo=0.3` para `k=5` es el mínimo. El resto de los valores
  elegidos queda dentro de sus listas.
- La QE está en unidades de score de PCA y no se compara con la QE de la corrida
  archivada, que usa 27 variables estandarizadas.
- Las figuras A y B del barrido y las tres figuras finales por tamaño se
  regeneraron con esta entrada.

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
- Corrida con 10 componentes principales: completada; barrido, corridas finales
  y figuras regeneradas en `resultados/ej1/ej1.2-pca10/`.
- Pendiente: interpretación de resultados con el grupo, fuera de los archivos
  del proyecto.

## Ejercicio 1.3 — Índice de Oja

### Alcance

- El plan completo está en `docs/plan/ej1-3.md`.
- El script es `src/ej1/oja.py` y la configuración `src/ej1/oja.toml`.
- Los tests están en `tests/ej1/test_oja.py`.
- Las salidas de la corrida final están en `resultados/ej1/oja/epocas_5/`.
- Las salidas de la corrida de 500 épocas están archivadas en
  `resultados/ej1/oja/epocas_500/`.
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
- La corrida final usa `epocas = 5` (250 actualizaciones), para que la
  convergencia sea visible en las figuras. La corrida previa de 500 épocas
  (25.000 actualizaciones) se conserva en `resultados/ej1/oja/epocas_500/` y no
  se vuelve a ejecutar.
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
- 3 semillas y las mismas 5 épocas de la corrida final. La corrida archivada de
  500 épocas conserva su propio análisis de sensibilidad.
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
- En `epocas_500/` está guardada `config_regla_taylor_0.05.toml`, porque las
  tres semillas de esa corrida convergieron.
- En `epocas_5/` no se guardó ninguna configuración de Taylor porque con 5
  épocas ninguna variante de Taylor convergió.

### Salidas

- `resultados/ej1/oja/epocas_5/`: corrida final con 5 épocas.
- `resultados/ej1/oja/epocas_500/`: corrida archivada con 500 épocas.
- Cada carpeta contiene:
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

### Resultados de la corrida de 500 épocas (archivada)

- Esta corrida no se vuelve a ejecutar; sus resultados están en
  `resultados/ej1/oja/epocas_500/`.
- Corrida: regla exacta, `eta_inicial = 0.5`, decaimiento `inversa`,
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
- Salidas completas en `resultados/ej1/oja/epocas_500/`: `config_usada.toml`,
  `config_regla_taylor_0.05.toml`, `normalizacion.csv`, `pesos.csv`,
  `indice.csv`, `comparacion_pc1.csv`, `correlaciones_indice.csv`,
  `convergencia.csv`, `sensibilidad.csv`, `sensibilidad_resumen.csv`,
  `comparacion_reglas.csv`, `comparacion_reglas_resumen.csv`, `resumen.txt` y
  las siete figuras.

### Resultados de la corrida final de 5 épocas

- Corrida: regla exacta, `eta_inicial = 0.5`, decaimiento `inversa`,
  `epocas = 5` (250 actualizaciones) y semillas `[0, 1, 2]`.
- Las tres semillas convergieron sin divergencia.
- Razón de varianza por semilla: 0.999323, 0.999211 y 0.998505.
- Razón de varianza: `0.999013 ± 0.000443`.
- Pearson con PC1 por semilla: 0.999966, 0.999970 y 0.999903.
- Pearson con PC1: `0.999946 ± 0.000038`.
- Norma final de los pesos: `1.000000`.
- Comparación con PC1:
  - Pearson entre el índice y PC1: `0.9999612314`.
  - Razón de varianza: `0.9989829570`.
  - Varianza del índice: `13.4072180710`.
  - Varianza de PC1: `13.4208680017`.
  - `lambda1`: `13.4208676700`.
- El índice ya no coincide exactamente con PC1: la razón de varianza es
  0.998983 y el ranking tiene seis creadores que cambian una posición.
- Los seis cambios de ranking son tres pares que intercambian posiciones
  adyacentes: `@artequeinspira`/`@entrelienzos`, `@brochasymates`/
  `@cosmeticaconsciente` y `@guiadehuellas`/`@clubdelasmascotas`.
- El top 5 se mantiene: `@futbolsinvueltas`, `@galeriacotidiana`,
  `@brillourbano`, `@crearenreels` y `@golconmate`.
- La figura de convergencia muestra la subida desde valores iniciales cercanos
  a 0.97 hasta 0.999 en cinco épocas, con eje x lineal.
- Comparación de reglas con 5 épocas:
  - exacta: razón de varianza `0.999013`, Pearson `0.999946`, sin divergencias;
  - Taylor `0.05`: razón de varianza `0.809878`, Pearson `0.985778`; no
    converge con este presupuesto;
  - Taylor `0.5`: dos semillas divergen y una converge; no se guarda
    configuración.
- Sensibilidad con 5 épocas: solo algunas combinaciones alcanzan el criterio
  (`eta_inicial = 0.05` con `raiz`, `0.2` con `inversa` y `0.5` con `inversa`);
  el resto no converge con tan pocas épocas. El análisis completo de
  sensibilidad con 500 épocas queda en `epocas_500/sensibilidad_resumen.csv`.
- Salidas en `resultados/ej1/oja/epocas_5/`. No se guardó configuración de
  Taylor porque ninguna variante convergió con 5 épocas.

### Pendientes de 1.3

- Fase 1: completada; plan y decisiones registrados.
- Fase 2: completada; `tests/ej1/test_oja.py` con 23 tests.
- Fase 3: completada; `src/ej1/oja.py` y `src/ej1/oja.toml`.
- Fase 4: completada; corrida final de 5 épocas y archivo de 500 épocas.
- Pendiente: interpretación de resultados con el grupo, fuera de los archivos
  del proyecto.

## Ejercicio 2.1 — Recuperación de códigos QR deteriorados

### Alcance

- El plan completo está en `docs/plan/ej2-1.md`.
- Los modelos de ruido, niveles y decisiones están en `docs/noise.md`.
- Los scripts van en `src/ej2/`: `hopfield.py` (motor, que ya tiene la regla
  de la pseudo-inversa), `hopfield-runner.py`, `hopfield-noise.py`, `ruido.py`
  y `hopfield-figures.py`.
- Las configuraciones son `src/ej2/hopfield-qr-hebb.toml`,
  `src/ej2/hopfield-qr-pseudoinversa.toml` y `src/ej2/hopfield-noise.toml`.
- Los tests son `tests/ej2/test_hopfield.py`,
  `tests/ej2/test_hopfield_runner.py`, `tests/ej2/test_ruido.py`,
  `tests/ej2/test_hopfield_noise.py` y `tests/ej2/test_hopfield_figures.py`.
- Las salidas definitivas van en `resultados/ej2/hopfield/qr_37x37/`, con una
  carpeta por regla de pesos, una carpeta por modelo de ruido y una carpeta
  consolidada.
- La corrida preliminar queda archivada en `qr_hebb/`, `qr_pseudoinversa/` y
  `analisis_qr/` (ver «Corrida preliminar (archivada)»).

### Datos

- `datos/hopfield/qr_37x37.csv` contiene los cinco QR: `beverages_menu_qr`,
  `dessert_menu_qr`, `food_menu_qr`, `specials_menu_qr` y `wine_menu_qr`.
- Cada patrón es una matriz de 37 × 37: `N = 1369`.
- `1` representa negro y `-1` representa blanco.
- El archivo es de solo lectura; ningún script de entrenamiento lo modifica.
- `p/N = 0,003652` y el límite teórico `0,15N` es `205,35`. La capacidad no es
  el problema del caso de uso.
- El solapamiento absoluto medio entre pares es `0,670`; el máximo es `0,860`
  para `food`–`wine`. La distancia de Hamming mínima es de 96 bits.
- 955 de los 1369 píxeles (69,8%) valen lo mismo en los cinco QR. Son QR de
  versión 5 sin zona de silencio: los tres finders de 7 × 7 están en las
  esquinas `(0, 0)`, `(0, 30)` y `(30, 0)` y son idénticos en los cinco.

### Pre-chequeo de las reglas de pesos

Se ejecutó una verificación de solo lectura, en memoria, sin escribir archivos
del proyecto.

Con la regla de Hebb (`W = (1/N) Xᵀ X`, diagonal nula):

- ninguno de los cinco patrones es punto fijo;
- los cinco convergen en dos barridos al mismo estado espurio, que coincide con
  el voto mayoritario `sign(Σ ξᵘ)`, con energía `−2172,8795`;
- cualquier par de QR se almacena correctamente;
- cualquier terna o conjunto mayor falla.

Con la regla de la pseudo-inversa (`W = Xᵀ (X Xᵀ)⁻¹ X`, diagonal nula y matriz
simetrizada):

- los cinco patrones son puntos fijos;
- con ruido aleatorio y 3 semillas × 5 repeticiones × 5 patrones, la exactitud
  es 100% hasta el 40% de bits invertidos, 74,7% al 45% y 0% al 50%.

El problema es el crosstalk entre patrones correlacionados, no la capacidad. La
pseudo-inversa es una extensión documentada de la regla de Hebb: se usa para
mostrar una recuperación viable y para comparar, no para reemplazar la línea de
base teórica.

### Corrida preliminar (archivada)

Antes de este plan se hizo una versión previa de 2.1. Sus carpetas no se borran
ni se regeneran.

- Redes: `resultados/ej2/hopfield/qr_hebb/` y `qr_pseudoinversa/`, entrenadas
  con `hopfield.py --regla`. Sus `config_usada.toml` tienen rutas absolutas de
  la máquina en que se corrieron.
- Consultas: deterioros preliminares del runner, 3 semillas × 5 repeticiones ×
  5 QR por nivel. `ruido` de 0,05 a 0,50 en pasos de 0,05; `esquina`
  preliminar (cuadrado de área `nivel · N`) en 0,05; 0,10; 0,15; 0,20; 0,25;
  0,30; 0,40 y 0,50; `manchas` preliminares (bloques de 3 × 3 invertidos) de
  0,05 a 0,50. Las carpetas `detalle_*` guardan la energía por actualización de
  cuatro casos. Son 88 corridas por regla.
- Análisis: `analisis_qr/`, generado con `src/ej2/hopfield-qr-analisis.py`.
- Hebb: 0/5 puntos fijos; todas las consultas de todos los niveles terminan en
  el estado espurio (exactitud 0).
- Pseudo-inversa: 5/5 puntos fijos. `ruido`: 1,000 hasta 0,40,
  0,747 ± 0,101 en 0,45 y 0 en 0,50. `esquina` preliminar: 1,000 hasta 0,40 y
  0,880 ± 0,069 en 0,50. `manchas` preliminares: 1,000 hasta 0,30,
  0,973 ± 0,023 en 0,35, 0,853 ± 0,046 en 0,40, 0,573 ± 0,101 en 0,45 y 0 en
  0,50.
- Capacidad por subconjuntos: con Hebb se almacenan los 5 QR sueltos y los 10
  pares, pero ninguna terna, ninguna cuaterna ni el conjunto completo; con la
  pseudo-inversa se almacenan todos.
- `ruido` es el mismo modelo que `aleatorio`. La `esquina` preliminar y las
  `manchas` preliminares no son los modelos de `docs/noise.md` (ver su sección
  4.8).
- Se generó con el motor anterior al arreglo de empates; ver «Empates».

### Modelo

- Red de Hopfield discreta con estados `{-1, +1}`.
- Campo local `h_i = Σ_j W_ij S_j`; un empate `|h_i| ≤ 1e-9` conserva el
  estado.
- Energía `E(S) = -1/2 Sᵀ W S`.
- Dinámica asincrónica secuencial por defecto; sincrónica disponible, pero no se
  usa en esta corrida.
- `max_barridos = 100`.
- No se aplica normalización: los patrones ya son binarios. No hay split de
  entrenamiento, validación y test; el ejercicio es una memoria asociativa. Los
  cinco QR son los patrones almacenados y las versiones corruptas son la
  evaluación. Es una excepción documentada al punto 2 de `CLAUDE.md`.
- La regla del 15% es una guía para patrones poco correlacionados, no una
  condición suficiente. La comparación entre Hebb y pseudo-inversa separa
  capacidad de ortogonalidad.

### Empates

- **Problema.** `recuperar` comparaba el campo local con 0 en punto flotante.
  Donde la aritmética exacta da 0, el redondeo deja residuos del orden de
  `1e-17` cuyo signo depende del orden de suma, es decir, de la plataforma. En
  las consultas guardadas de 2.2 (pesos múltiplos de 0,04), 22 de 300 no
  respetaban la regla «el empate conserva el estado», y en Windows 3 de 300
  estados finales diferían de los generados en Linux.
- **Decisión.** `|h_i| ≤ 1e-9` es un empate y conserva el estado, en los dos
  modos (`TOLERANCIA_EMPATE` en `hopfield.py`). El análisis usa el mismo
  criterio: un patrón es punto fijo si todos sus márgenes `ξ_i h_i` son
  `≥ −1e-9`, una neurona está en empate si `|ξ_i h_i| ≤ 1e-9` y, en
  `capacidad_por_subconjuntos`, un margen nulo cuenta como estable (antes se
  exigía margen `> 0`).
- **Por qué `1e-9`.** Con Hebb, `h_i` es múltiplo de `1/N`: un campo no nulo
  vale al menos `0,04` en letras y `1/1369 ≈ 7,3e-4` en QR. La tolerancia nunca
  oculta un campo real. Con la pseudo-inversa no apareció ningún `|h_i| < 1e-9`
  en las consultas revisadas.
- **Tests.** Empate numérico (`0,1 + 0,2 − 0,3`) en los dos modos, punto fijo
  con margen nulo en el análisis y en la capacidad, y regresión de las 16
  corridas de 2.2 contra los CSV regenerados.
- **2.2.** Se regeneraron `consultas/limpias/` y los 15 barridos de ruido de
  `resultados/ej2/hopfield/letras_5x5/`. Los estados iniciales y el
  entrenamiento no cambian; 14 de las 300 consultas ruidosas cambiaron de tipo
  de resultado y las 304 coinciden ahora con la dinámica en aritmética exacta.
  El ejemplo de estado espurio no cambia.

  | Ruido | Accuracy antes | Accuracy después | C/I/E antes | C/I/E después |
  |---:|---:|---:|---:|---:|
  | 0.1 | 0.983333 ± 0.028868 | 0.983333 ± 0.028868 | 59/0/1 | 59/0/1 |
  | 0.2 | 0.783333 ± 0.125831 | 0.783333 ± 0.104083 | 47/4/9 | 47/3/10 |
  | 0.3 | 0.600000 ± 0.132288 | 0.566667 ± 0.160728 | 36/11/13 | 34/10/16 |
  | 0.4 | 0.250000 ± 0.050000 | 0.233333 ± 0.028868 | 15/18/27 | 14/17/29 |
  | 0.5 | 0.083333 ± 0.028868 | 0.050000 ± 0.000000 | 5/28/27 | 3/26/31 |

  C/I/E: consultas correctas, incorrectas y espurias sobre 60.
- **QR preliminar.** Se repitieron en el scratchpad, sin pisar nada, el
  entrenamiento y las 88 corridas de cada regla con el motor corregido. Con la
  pseudo-inversa todo es idéntico. Con Hebb, los estados finales y los tipos de
  resultado son idénticos en las 88 corridas; solo 8 de las 2120 consultas, en 7
  corridas, cambian la cantidad de barridos (de 2 a 3 o de 3 a 2), y
  `estabilidad.csv` difiere por redondeo (`1e-12`). Las conclusiones de la
  corrida preliminar no cambian.

### Regla de la pseudo-inversa (Personnaz)

- `W = Xᵀ (X Xᵀ)⁻¹ X`, la regla de proyección de Personnaz, Guyon y Dreyfus
  (1985). El código la llama `pseudoinversa` y usa `numpy.linalg.pinv(X Xᵀ)`,
  que coincide con la inversa porque `X Xᵀ` tiene rango completo.
- Diagonal nula y matriz simetrizada.
- Es una regla cerrada, igual que Hebb; no hay aprendizaje iterativo.
- Para este dataset, `X Xᵀ` tiene rango 5 y número de condición 26,39.
- La diagonal nula es una aproximación: se verifica que los cinco patrones sigan
  siendo puntos fijos después de anularla.
- El límite práctico no es `0,15N`; el área de atracción se reduce al aumentar
  la cantidad de patrones.
- La configuración usa `modelo.regla = "hebb" | "pseudoinversa"`. El valor por
  defecto es `hebb`. La CLI tiene `--regla`. Las configuraciones viejas sin esa
  clave se interpretan como `hebb`, como la de 2.2.

### Ruido

Se ejecutan seis modelos individuales y una escalera de combinaciones. El
detalle está en `docs/noise.md`.

| # | Modelo | Parámetro nominal | Modo |
|---|---|---|---|
| 1 | `aleatorio` | fracción de bits invertidos | invierte |
| 2 | `esquina` | lado del cuadrado en una esquina | borra a `-1` |
| 3 | `finder` | cantidad de finders de 7 × 7 (1, 2 o 3) | borra a `-1` |
| 4 | `rayones` | cantidad de líneas completas | borra a `-1` |
| 5 | `manchas` | cantidad y radio de discos | borra a `-1` |
| 6 | `bloques` | fracción de los 36 bloques completos de 6 × 6 | borra a `-1` |

Escalera de combinaciones:

| Paso | Componentes activos | Nivel fijo |
|---|---|---|
| 1 | `aleatorio` | 5% de bits |
| 2 | + `esquina` | 9 × 9 |
| 3 | + `finder` | un finder de 7 × 7 |
| 4 | + `rayones` | 2 líneas |
| 5 | + `manchas` | 1 disco de radio 5 |
| 6 | + `bloques` | 10% de bloques (4 de 36) |

- Se usan las semillas `[0, 1, 2]`, cinco repeticiones por patrón y los cinco
  patrones.
- Los niveles se fijan a priori y no se ajustan para mejorar la exactitud.
- La consulta ruidosa es la misma para Hebb y pseudo-inversa.
- `fraccion_afectada` es la fracción de bits que efectivamente cambian respecto
  del original, no el área nominal.
- En las combinaciones, el borrado estructural tiene prioridad sobre un bit
  invertido en la misma posición.
- Flujos aleatorios: `aleatorio` usa `default_rng(semilla)`, como el runner, y
  reproduce la corrida preliminar de `ruido`; cada modelo estructural usa
  `default_rng([semilla, código])`. En las combinaciones cada componente
  conserva su flujo.
- El nivel «1 finder de 9 × 9» se descartó porque coincide con `esquina` con
  `L = 9`. Los bloques se eligen solo entre los 36 completos de 6 × 6.
- Los deterioros preliminares del runner (`--deterioro ruido|esquina|manchas`)
  se conservan sin cambios; los modelos definitivos se ejecutan solo con
  `hopfield-noise.py --modelo`.

### Métricas

El runner ya guarda `barridos`, `actualizaciones`, `energia_inicial`,
`energia_final`, `convergio`, `ciclo`, `motivo`, `hamming_inicial` y
`energia.csv` por barrido. Se agregan, sin renombrar columnas existentes:

- `hamming_final_origen` y `hamming_final_normalizado`;
- `fraccion_afectada` (`hamming_inicial / N`);
- `cambios_totales`;
- `delta_energia` y `delta_energia_por_neurona`;
- `monotona`.

`distancia_hamming` se conserva como la distancia al patrón almacenado más
cercano. Las columnas nuevas miden el error de reconstrucción respecto del
origen. Se agrega `metricas.csv` por corrida con exactitud, tasas de
incorrectos, espurios, ciclos y no convergencia, medias y desvíos de barridos,
cambios y energía por neurona, y Hamming normalizado.

La composición de errores distingue correcto, incorrecto, espurio, ciclo y no
converge. La comparación entre Hebb y pseudo-inversa usa exactitud, Hamming y
barridos. La energía absoluta no se compara entre reglas porque la matriz de
pesos cambia; se analiza dentro de cada regla con `E/N` y `ΔE/N`. Para Hebb, el
estado espurio tiene menor energía que los cinco patrones almacenados, lo que
explica la convergencia al voto mayoritario.

### Figuras

Todas las figuras se guardan y se regeneran desde los CSV con
`src/ej2/hopfield-figures.py`, que solo lee resultados. Las principales son:

- comparación de entrenamiento: solapamiento, Hamming, estado espurio, energía,
  distribución de pesos y capacidad por subconjuntos;
- por modelo: exactitud, composición de errores, iteraciones, energía, Hamming,
  confusión, exactitud por patrón, boxplot de barridos y afectación real;
- combinaciones: exactitud, iteraciones, energía, Hamming y mapa de calor por
  paso;
- globales: exactitud, iteraciones, energía, Hamming, comparación
  Hebb/pseudo-inversa, mapa de calor de exactitud, ejemplos cualitativos de QR,
  cuenca de atracción y trayectorias de energía.

### Reproducibilidad y reglas

- Cada corrida usa semillas explícitas y guarda su configuración resuelta.
- Cada resultado presentado es media ± desvío sobre tres semillas.
- Las figuras se regeneran desde los CSV guardados.
- La energía por barrido se guarda para todas las consultas; los estados
  completos, solo para las consultas de ejemplo (`ejemplos.npz`).
- No se usan bibliotecas de machine learning; solo NumPy, pandas, matplotlib y
  la biblioteca estándar.
- Los scripts de entrenamiento escriben; los scripts de análisis y figuras solo
  leen.
- No se escribe en `datos/`.
- No se commitea ni se pushea sin permiso explícito.

### Resultados de entrenamiento (fase 1)

- Redes: `resultados/ej2/hopfield/qr_37x37/hebb/` y `pseudoinversa/`,
  entrenadas con `src/ej2/hopfield-qr-hebb.toml` y
  `src/ej2/hopfield-qr-pseudoinversa.toml`.
- `capacidad.csv`: `p/N = 0,003652`, límite `0,15N = 205,35`; no se excede.
- `condicion.csv`: `X Xᵀ` tiene rango 5 y número de condición 26,3904.
- Hebb: 0/5 puntos fijos. Las cinco consultas limpias convergen en 2 barridos
  al estado mezcla `sign(Σ ξᵘ)`, que es punto fijo (margen mínimo 0,4251).
- Energías con Hebb (`energia_estados.csv`): los patrones tienen entre
  −2077,45 (`food`) y −1789,71 (`beverages`), es decir, `E/N` entre −1,5175 y
  −1,3073. La mezcla tiene −2172,88 (`E/N = −1,5872`), menos que cualquiera de
  ellos: por eso atrae todas las consultas.
- Pseudo-inversa: 5/5 puntos fijos con margen mínimo 0,9808; cada patrón tiene
  energía −682,00 (`E/N = −0,4982`), que es `−(N − p)/2`. Las cinco consultas
  limpias se recuperan en 1 barrido. Con esta regla la mezcla no es punto fijo
  (margen mínimo −0,3247) y tiene más energía (−607,89).
- `capacidad_subconjuntos.csv` reproduce la corrida preliminar: Hebb sostiene
  los 10 pares y ninguna terna; la pseudo-inversa sostiene todos los
  subconjuntos.

### Métricas del runner (fase 2)

- `consultas.csv` agrega al final `hamming_final_origen`,
  `hamming_final_normalizado`, `fraccion_afectada`, `cambios_totales`,
  `delta_energia`, `delta_energia_por_neurona` y `monotona`; las columnas
  viejas no cambian de nombre, orden ni valor (la regresión de 2.2 pasa).
- Cada corrida guarda `metricas.csv` y un bloque «Métricas agregadas» en
  `resumen.txt`.
- Verificación sobre las redes QR (consultas limpias en
  `qr_37x37/<regla>/consultas/limpias/`; la ruidosa, con ruido 0,4 y semilla 0,
  se corrió fuera del repo para no guardar estados completos). En las cuatro
  corridas las métricas coinciden con los estados guardados y con
  `energia.csv`, y la energía es monotónica en todas las consultas.
- Hebb, limpias: exactitud 0, 2 barridos, 143 cambios en promedio,
  `ΔE/N = −0,1711` y `H/N = 0,1045`. Con ruido 0,4: exactitud 0,
  2,28 barridos, 580 cambios, `ΔE/N = −1,5308` y el mismo `H/N = 0,1045`,
  porque siempre termina en el estado mezcla.
- Pseudo-inversa, limpias: exactitud 1, 1 barrido y 0 cambios. Con ruido 0,4:
  exactitud 1, 2,12 barridos, 548,2 cambios (casi exactamente los 548 bits
  invertidos) y `ΔE/N = −0,4789`.

### Barrido de ruido (fases 3 a 9)

- Orquestador: `src/ej2/hopfield-noise.py` con `src/ej2/hopfield-noise.toml`.
  Cada invocación genera cada consulta una vez y la recupera con las dos redes.
  Salida por modelo en `resultados/ej2/hopfield/qr_37x37/ruido/<modelo>/`:
  `config_usada.toml`, `consultas.csv` (una fila por consulta y regla, con los
  parámetros efectivos en JSON), `energia.csv` (energía por barrido de todas las
  consultas), `metricas_por_semilla.csv`, `resumen_niveles.csv`,
  `comparacion_reglas.csv`, `resumen_por_patron.csv`, `resumen_energia.csv`,
  `ejemplos.npz`, `resumen.txt` y `figuras/`.
- `resumen_energia.csv` promedia `E` por barrido; una consulta que ya se
  estabilizó conserva su energía final en los barridos siguientes.
- Figuras: `src/ej2/hopfield-figures.py`, que solo lee resultados; las nueve
  figuras por modelo se regeneran byte a byte.
- Todos los resultados son media ± desvío muestral sobre las semillas 0, 1 y 2,
  con 25 consultas por semilla y regla (5 QR × 5 repeticiones).

#### Ruido aleatorio (fase 3)

- 11 niveles, de 0,02 a 0,50. Tardó 46 s.
- La consulta es idéntica para las dos reglas, y en los niveles comunes
  (0,05 a 0,50) el modelo reproduce consulta por consulta la corrida
  preliminar repetida con el motor corregido (750 consultas por regla, sin
  diferencias). Contra la corrida preliminar original solo cambian 3 conteos
  de barridos de Hebb, por el arreglo de empates.
- Pseudo-inversa: exactitud 1,000 ± 0,000 desde 0,02 hasta 0,40,
  0,747 ± 0,101 en 0,45 (21,3% incorrectos, 4,0% espurios) y 0,000 en 0,50
  (41,3% incorrectos, 58,7% espurios). Hasta 0,35 recupera en 2 barridos
  exactos y cambia exactamente los bits invertidos (27, 68, …, 479); en 0,40
  usa 2,09 barridos y en 0,45, 2,60.
- Hebb: exactitud 0 en todos los niveles; todas las consultas terminan en el
  estado mezcla (`H/N = 0,1045`) salvo en 0,50, donde `H/N = 0,5456`. Necesita
  entre 2,03 y 2,35 barridos (2,91 en 0,50) y cambia más bits que los invertidos
  (164 con 2% de ruido) porque también reescribe los píxeles en que el patrón
  difiere de la mezcla.
- `ΔE/N` pasa de −0,039 a −0,495 con la pseudo-inversa y de −0,280 a −1,588
  con Hebb; la energía es monotónica en todas las consultas.
- Con 50% de bits invertidos la consulta queda a la misma distancia del origen
  que de su negativo: el 56,0% (Hebb) y el 54,7% (pseudo-inversa) de las
  consultas terminan a más de `N/2` del origen, coherente con atractores
  negados (`−ξ`, `−mezcla`), que son estables por la simetría `S → −S`.
- Nivel de referencia de la confusión: 0,45. Las confusiones de la
  pseudo-inversa van sobre todo de `dessert` a `beverages` (4 de 15).

#### Esquina rota (fase 4)

- Lados 7, 9, 11, 15, 19 y 23; 18 s. La esquina se registra en `parametros`;
  como el flujo no depende del lado, cada consulta usa la misma esquina en
  todos los niveles. En las 75 consultas de cada nivel salieron 24 inferior
  derecha, 23 superior izquierda, 16 superior derecha y 12 inferior izquierda.
- Fracción afectada real media: 2,25%, 3,22%, 4,71%, 8,67%, 13,47% y 19,66%,
  alrededor de la mitad de la zona borrada (3,58% a 38,64%), porque solo
  cambian los píxeles negros. Ninguna consulta tiene daño nulo (mínimo 1,83%).
  Con `L = 7`, las tres esquinas con finder pierden exactamente 33 píxeles y la
  inferior derecha, 26,3 en promedio.
- Pseudo-inversa: exactitud 1,000 ± 0,000 en todos los lados; 2 barridos
  (2,08 con `L = 23`).
- Hebb: exactitud 0 en todos los lados; siempre termina en el estado mezcla
  (`H/N = 0,1045`), con 2,31 a 2,44 barridos.

#### Finder borrado (fase 5)

- 1, 2 y 3 finders de 7 × 7; 13 s. Los finders borrados se registran en
  `parametros`; con un finder salieron 28 superior izquierda, 28 superior
  derecha y 19 inferior izquierda.
- La fracción afectada es exacta en todas las consultas: 33, 66 y 99 píxeles
  (2,41%, 4,82% y 7,23%), porque los tres finders son idénticos en los cinco QR.
- Pseudo-inversa: exactitud 1,000 ± 0,000 con 1, 2 y 3 finders, en 2 barridos.
- Hebb: exactitud 0; termina en el estado mezcla, y los barridos suben con la
  cantidad de finders (2,40; 2,60 y 2,80).
- Como los finders son comunes a los cinco patrones, la red los reconstruye
  igual que cualquier otro píxel compartido: no usa la estructura del QR.

#### Rayones (fase 6)

- 1, 2, 4, 8 y 12 líneas; 17 s. Cada línea registra su orientación y el píxel
  por el que pasa. Sobre las 2025 líneas trazadas salieron 551 verticales, 509
  diagonales ascendentes, 486 descendentes y 479 horizontales.
- Fracción afectada real media: 1,15%, 2,26%, 4,57%, 8,49% y 12,38%. La zona
  dañada va de 2,16% a 23,75%. El mínimo (4 píxeles con una línea) es una
  diagonal corta cerca de una esquina, como prevé la definición.
- Pseudo-inversa: exactitud 1,000 ± 0,000 en todos los niveles, en 2 barridos.
- Hebb: exactitud 0; termina en el estado mezcla, con 2,05 a 2,21 barridos.

#### Manchas (fase 7)

- 1 × r3, 1 × r5, 1 × r7, 1 × r10, 2 × r5 y 3 × r5; 24 s. Se registran los
  centros y el radio.
- Fracción afectada real media: 1,02%, 2,66%, 4,74%, 9,33%, 5,30% y 7,75%
  (máximo 12,42% con 1 × r10). La zona dañada media va de 1,97% a 18,08%;
  queda por debajo del área del disco cuando la mancha se recorta en el borde.
- Pseudo-inversa: exactitud 1,000 ± 0,000 en todos los niveles (2,01 barridos
  con 1 × r10).
- Hebb: exactitud 0; termina en el estado mezcla, con 2,08 a 2,36 barridos.
- A diferencia de las `manchas` preliminares (bloques de 3 × 3 invertidos, que
  con la pseudo-inversa caían desde 35% de píxeles invertidos), las manchas
  borran a blanco y su daño real no pasa del 12,4%.

#### Bloques borrados (fase 8)

- 5%, 10%, 20%, 30% y 40% de los 36 bloques completos (2, 4, 7, 11 y 14
  bloques); 19 s. Se registran los índices de los bloques; la última fila y la
  última columna nunca se borran.
- Zona dañada exacta: 5,26%, 10,52%, 18,41%, 28,93% y 36,82%. Fracción
  afectada real media: 2,68%, 5,35%, 9,47%, 14,64% y 18,80% (máximo 20,23%).
- Pseudo-inversa: exactitud 1,000 ± 0,000 en todos los niveles (2,01 barridos
  con 40%).
- Hebb: exactitud 0; termina en el estado mezcla, con 2,13 a 2,24 barridos.
- Con los niveles fijados a priori, ningún modelo estructural supera el 20,2%
  de píxeles cambiados. La pseudo-inversa los recupera todos, igual que al
  ruido aleatorio de hasta 40%. Los niveles no se ajustan para buscar el punto
  de quiebre.

#### Combinaciones (fase 9)

- Escalera de 6 pasos, uno por invocación (`--combinaciones --paso k`), en
  `ruido/combinaciones/paso_<k>/`; consolidada en
  `ruido/combinaciones/consolidado/`.
- Cada paso contiene exactamente sus componentes y cada componente usa el
  mismo flujo que su modelo individual: la esquina del paso 6 es idéntica a la
  del modelo `esquina` con `L = 9` (test).
- Fracción afectada real media por paso: 4,97%, 7,90%, 9,59%, 11,40%, 13,57%
  y 17,48% (máximo 20,23%). La zona dañada crece de 4,97% a 29,54%. En el
  paso 3, 18 de las 75 consultas borran un finder que ya estaba dentro de la
  esquina del paso 2 y no suman daño.
- Pseudo-inversa: exactitud 1,000 ± 0,000 en los seis pasos, en 2 barridos; los
  bits cambiados crecen de 68 a 239,3 y `ΔE/N` de −0,094 a −0,285.
- Hebb: exactitud 0 en los seis pasos (estado mezcla), con 2,03 a 2,59
  barridos.
- La escalera no hace caer a la pseudo-inversa en ningún paso: queda saturada
  en 1. Se reporta así; los niveles fijados a priori no se cambian.

### Figuras y resultados globales (fase 10)

- `src/ej2/hopfield-figures.py` genera 129 figuras: 5 de entrenamiento en
  `qr_37x37/comparacion_entrenamiento/` (con `comparacion_entrenamiento.csv`),
  9 por cada modelo y cada paso de la escalera, 5 de la escalera consolidada y
  11 globales en `ruido/consolidado/figuras/`. Una segunda corrida produce los
  129 PNG idénticos byte a byte, y el script no modifica ningún CSV ni NPZ.
- Consolidado: `ruido/consolidado/` une los seis modelos (5400 consultas, 2700
  por regla) y agrega `exactitud_por_intervalo.csv` con intervalos de 5% de
  fracción afectada real, de 0 a 50%. El nivel 0,50 del ruido aleatorio
  invierte 684 bits (`round(684,5)` redondea al par), es decir 49,96%.
- Todas las consultas de los modelos y de la escalera (900) convergen, ninguna
  llega a `max_barridos` (máximo 5 barridos) y la energía es monotónica en
  todas.
- Pesos: con Hebb toman 6 valores distintos, entre `±5/1369 = ±0,00365`; con la
  pseudo-inversa, valores entre `±0,0192`.
- Daño real contra zona dañada en los modelos estructurales: entre 0,51 y 0,67
  (media 0,54), porque solo cambian los píxeles negros (alrededor del 52% del
  QR). El daño estructural máximo es 19,66% (esquina con `L = 23`).
- `fig_comparacion_reglas.png` compara las reglas en el nivel de cada modelo
  más cercano a 10% de píxeles cambiados (decisión de presentación): aleatorio
  0,1 (10,0%), esquina 15 (8,7%), finder 3 (7,2%), rayones 8 (8,5%), manchas
  1 × r10 (9,3%) y bloques 0,2 (9,5%). La pseudo-inversa da 1 en todos y Hebb, 0.
- `fig_mapa_calor_exactitud.png`: la única celda por debajo de 1 con la
  pseudo-inversa está en el ruido aleatorio, 40–45% (0,873, mezcla de los
  niveles 0,40 y 0,45) y 45–50% (0). Los modelos estructurales no llegan más
  allá del intervalo 20–25%.
- `fig_ejemplos_qr.png` usa el nivel de referencia de cada modelo (el mismo
  criterio que la confusión): con ruido 0,45 el ejemplo de `food` termina en
  `wine` (Hamming 96, la distancia mínima entre QR); en los demás modelos la
  pseudo-inversa lo recupera y Hebb cae al estado mezcla (Hamming 71).
- Tamaño de `qr_37x37/`: 304 archivos, 9,5 MB de datos y 8,6 MB de figuras. Solo
  `consultas/limpias/` guarda estados completos (5 filas por red); el resto,
  solo las consultas de ejemplo en `ejemplos.npz`.

### Estado

- Fase 0: completada; plan, modelos de ruido y decisiones registrados, y
  ajustados después del diagnóstico de la corrida preliminar.
- Antes de la fase 1: tests de Windows corregidos y arreglo de empates con la
  regeneración de 2.2.
- Fase 1: completada; TOML de QR, `condicion.csv`, `energia_estados.csv`,
  redes entrenadas y consultas limpias.
- Fase 2: completada; métricas por consulta, `metricas.csv` y resumen.
- Fase 3: completada; orquestador, figuras por modelo y ruido aleatorio.
- Fase 4: completada; generador y corrida de `esquina`.
- Fase 5: completada; generador y corrida de `finder`.
- Fase 6: completada; generador y corrida de `rayones`.
- Fase 7: completada; generador y corrida de `manchas`.
- Fase 8: completada; generador y corrida de `bloques`.
- Fase 9: completada; escalera de combinaciones, consolidación y figuras.
- Fase 10: completada; figuras de entrenamiento y globales, consolidación,
  documentación y `check-cicd.py`.
- Pendiente final: interpretación de resultados con el grupo, fuera de los
  archivos del proyecto.

## Ejercicio 2.2 — Recomendación (Hopfield de 5 × 5)

### Alcance

- El plan completo está en `docs/plan/ej2-2.md`.
- Los scripts van en `src/ej2/`: `qr-to-matrix.py`, `hopfield.py` y
  `hopfield-runner.py`.
- La configuración por defecto es `src/ej2/hopfield.toml`; la CLI pisa el TOML.
- Las salidas van en `resultados/ej2/hopfield/<dataset>/`.
- El análisis de 2.1 se planifica por separado en `docs/plan/ej2-1.md` y sus
  modelos de ruido en `docs/noise.md`.

### Datos

- `datos/hopfield/abecedario_5x5.csv`: 26 letras A-Z, mayúsculas, sin Ñ.
- `datos/hopfield/letras_5x5.csv`: cuatro patrones para 2.2, `J`, `O`, `T` y
  `A`.
- La `J` se copia exactamente de la consigna.
- Formato CSV: `nombre,filas,columnas,p0000,...,pNNNN`, con valores `1` y `-1`.
- `datos/hopfield/qr_37x37.csv` se genera con `qr-to-matrix.py` a partir de los
  PNG. Es una excepción documentada a la regla de solo lectura de `datos/`.
- El conversor usa umbral 128: negro → `1`, blanco → `-1`, sin invertir.
- Se agrega `Pillow` a `requirements.txt`.

### Modelo

- Hopfield discreto con estados `{-1, +1}`.
- Pesos `W = (1/N) Xᵀ X`, diagonal nula y matriz simétrica.
- Campo local `h_i = Σ_j W_ij S_j`; un empate `|h_i| ≤ 1e-9` conserva el estado,
  en los dos modos. La tolerancia absorbe los residuos de redondeo (del orden de
  `1e-17`) que aparecen donde la aritmética exacta da 0: con letras, los pesos
  son múltiplos de `0.04` y los empates exactos son frecuentes.
- Energía `E(S) = -1/2 Sᵀ W S`.
- Actualización asincrónica secuencial por defecto; sincrónica opcional.
- El almacenamiento es Hebbiano en forma cerrada; la iteración hasta el estado
  estable es la recuperación.
- Se guardan `pesos.npz`, `patrones.csv`, `config_usada.toml`, estabilidad,
  solapamiento, Hamming, capacidad, energía y resumen.

### Runner y análisis

- El runner lee una red entrenada, valida dimensiones y clasifica cada consulta
  como almacenada correcta, almacenada incorrecta, espuria, ciclo o no
  convergente.
- `--ruido` y `--repeticiones` generan consultas ruidosas reproducibles.
- `--graphic` renderiza `1` como `*` y `-1` como espacio.
- El análisis de 2.2 revisa ortogonalidad, capacidad, estabilidad, energía y al
  menos un estado espurio si aparece.

### Corrida final

- Dataset: `datos/hopfield/letras_5x5.csv`.
- Patrones: `J`, `O`, `T` y `A`.
- `N = 25` neuronas y `p = 4` patrones.
- Configuración: `asincronica`, orden `secuencial`, semilla 0 y máximo 100
  barridos.
- Salida: `resultados/ej2/hopfield/letras_5x5/`.
- Los cuatro patrones son puntos fijos de la red.

### Ortogonalidad y capacidad

- Carga `p/N = 0.16`; el límite teórico es `0.15 N = 3.75`, así que la red
  excede el límite.
- Solapamiento absoluto medio: `0.173333`; máximo: `0.36` (par `O`-`A`).
- Distancia de Hamming mínima: `8` (par `O`-`A`); media: `11.833333`.
- No hay pares ortogonales.

| | J | O | T | A |
|---|---:|---:|---:|---:|
| J | 1.00 | 0.04 | 0.28 | -0.12 |
| O | 0.04 | 1.00 | -0.04 | 0.36 |
| T | 0.28 | -0.04 | 1.00 | -0.20 |
| A | -0.12 | 0.36 | -0.20 | 1.00 |

### Estabilidad

| Patrón | Punto fijo | Margen mínimo | Barridos | Energía |
|---|---|---:|---:|---:|
| J | Sí | 0.40 | 1 | -11.68 |
| O | Sí | 0.40 | 1 | -12.16 |
| T | Sí | 0.32 | 1 | -12.00 |
| A | Sí | 0.16 | 1 | -12.80 |

- Ningún patrón tiene neuronas inestables ni empates.
- `A` tiene el menor margen y el mayor solapamiento con `O`.
- La regla del 15 por ciento es una guía estadística: con cuatro patrones se
  supera el límite, pero los cuatro quedan estables.

### Consultas limpias

- Las cuatro consultas limpias se recuperan correctamente.
- Distancia de Hamming 0, un barrido y energía final igual a la inicial.
- Salida: `resultados/ej2/hopfield/letras_5x5/consultas/limpias/`.

### Robustez al ruido

Se corrieron 5 niveles de ruido × 3 semillas × 5 repeticiones × 4 letras = 60
consultas por nivel. La accuracy se reporta como media ± desvío muestral sobre
las 3 semillas.

| Ruido | Bits invertidos | Accuracy media ± std | Correctos | Incorrectos | Espurios |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 2 | 0.983333 ± 0.028868 | 59 | 0 | 1 |
| 0.2 | 5 | 0.783333 ± 0.104083 | 47 | 3 | 10 |
| 0.3 | 8 | 0.566667 ± 0.160728 | 34 | 10 | 16 |
| 0.4 | 10 | 0.233333 ± 0.028868 | 14 | 17 | 29 |
| 0.5 | 12 | 0.050000 ± 0.000000 | 3 | 26 | 31 |

Accuracy por letra:

| Ruido | A | J | O | T |
|---:|---:|---:|---:|---:|
| 0.1 | 0.933333 | 1.000000 | 1.000000 | 1.000000 |
| 0.2 | 0.733333 | 0.800000 | 0.733333 | 0.866667 |
| 0.3 | 0.533333 | 0.400000 | 0.666667 | 0.666667 |
| 0.4 | 0.266667 | 0.266667 | 0.333333 | 0.066667 |
| 0.5 | 0.133333 | 0.000000 | 0.066667 | 0.000000 |

- La recuperación cae rápido con el ruido. A partir de 0.4 dominan los estados
  espurios y las asociaciones incorrectas.
- `T` es la letra más robusta en 0.2 y empata con `O` en 0.3, pero es la que
  más cae en 0.4. `A` es la única que falla con ruido 0.1 y `J` es la más
  frágil en 0.3.
- Las consultas se regeneraron con la tolerancia de empate `|h_i| ≤ 1e-9`. Los
  estados iniciales no cambiaron; 14 de las 300 consultas ruidosas cambiaron de
  tipo de resultado. Las 304 consultas (limpias y ruidosas) coinciden ahora, en
  estado final y barridos, con la dinámica en aritmética exacta. La versión
  anterior de esta tabla queda en el historial de git.

### Estado espurio

- Caso: ruido `0.2`, semilla `0`, repetición `1`, patrón de origen `J`.
- El estado final es estable: 2 barridos y sin cambios en el último.
- No coincide con ninguna letra almacenada. El más cercano es `T`, con
  distancia de Hamming `4`; energía final `-11.68`.
- Matriz final (`·` representa `-1`):

```text
*****
··**·
·····
··**·
·**··
```

- En el mismo nivel aparecen otros espurios; con ruido `0.1` ya hay uno en la
  semilla `2`.
- El caso no cambia al regenerar las consultas con la tolerancia de empate:
  mismo estado final, mismos barridos y misma energía.

### Reproducibilidad

- Una segunda corrida de entrenamiento en `/tmp/opencode/hopfield_repro`
  produjo pesos, patrones y tablas idénticos.
- Una segunda corrida del runner para ruido `0.2` y semilla `0` produjo un
  `consultas.csv` idéntico.
- La corrida final y las consultas se pueden regenerar con los comandos de
  `docs/structure.md`.

### Salidas

- `resultados/ej2/hopfield/letras_5x5/`: `config_usada.toml`, `patrones.csv`,
  `pesos.npz`, `estabilidad.csv`, `estabilidad_neuronas.csv`,
  `solapamiento.csv`, `hamming.csv`, `ortogonalidad_resumen.csv`,
  `capacidad.csv`, `energia_entrenamiento.csv` y `resumen.txt`.
- `consultas/limpias/`.
- `consultas/ruido_<nivel>_semilla_<semilla>/` para los 15 barridos de ruido.
- No se generan figuras en esta etapa.

### Estado

- Fase 1: completada; plan y decisiones registrados.
- Fases 2 a 6: completadas.
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
- `python3 -m pytest -q`: 240 tests en verde (también en Windows, después de
  corregir la lectura de archivos y las rutas en dos tests).
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
- Interpretar con el grupo los resultados de 2.2, incluida la aparición de
  estados espurios y la caída de accuracy con el ruido.
- Interpretar con el grupo las limitaciones de 2.1: correlación entre QR,
  estados espurios y preprocesamiento de imágenes reales.
