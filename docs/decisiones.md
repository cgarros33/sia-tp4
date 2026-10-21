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
- Etapa actual: 1.2 implementado; barrido completo con resultados provisorios.
  Queda pendiente revisar los valores que quedaron en el borde de sus rangos y
  correr la configuración final (fase F).
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

Orden del barrido secuencial:

1. `grilla`
2. `topologia`
3. `distancia`
4. `inicializacion`
5. `radio_inicial_relativo`
6. `decaimiento_radio`
7. `eta_inicial`
8. `decaimiento_eta`
9. `iteraciones`

| Hiperparámetro | Valores a probar | Neutro |
|---|---|---|
| `grilla` | `6, 7, 8, 9, 10, 11, 12, 13, 14` | `6` |
| `topologia` | `rectangular, hexagonal` | `rectangular` |
| `distancia` | `euclidiana, exponencial` | `euclidiana` |
| `inicializacion` | `uniforme, muestras` | `muestras` |
| `radio_inicial_relativo` | `0.3k, 0.4k, 0.5k, 0.75k, 1.0k, 1.5k` | `0.5k` |
| `decaimiento_radio` | `constante, lineal, exponencial, inversa` | `lineal` |
| `eta_inicial` | `0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9` | `0.5` |
| `decaimiento_eta` | `lineal, exponencial, inversa` | `lineal` |
| `iteraciones` | `5000, 10000, 20000, 40000, 80000` | `20000` |

- `radio_final = 1.0` y `eta_final = 0.001` quedan fijos por la teoría.
- Las listas de valores viven en `src/ej1/kohonen-sweep.toml`, no en el código.
  Si un óptimo queda en el borde de una lista, se amplía la lista y se vuelve a
  correr ese paso sin tocar el código.
- Las iteraciones se dejan como están por ahora (`20000` neutro y candidatos de
  `5000` a `80000`). Quedan fácilmente modificables en la configuración.

### Selección de hiperparámetros

- Para todos los hiperparámetros, salvo `grilla`, se elige el menor QE medio
  sobre las semillas.
- Excepción para `grilla`: se elige el menor `k` cuyo QE medio esté dentro del
  5% del QE medio mínimo. Se configura con
  `criterio_grilla = "codo"` y `tolerancia_codo = 0.05`.
- La excepción y el codo del 5% quedan registrados acá.
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

- Se generan las opciones A y B, cada una en su carpeta:
  - `resultados/ej1/kohonen_sweep/figuras/opcion_a/`: una figura por paso más
    `fig_progreso.png`. El valor elegido se marca subrayando su etiqueta del
    eje x, sin línea vertical.
  - `resultados/ej1/kohonen_sweep/figuras/opcion_b/`: dashboard enfocado en
    legibilidad, con los 9 pasos secuenciales y otro dashboard para los
    acoplamientos.
- La opción C (boxplots por paso) queda desactivada por legibilidad. Su código
  se conserva, pero no se genera en el barrido real.
- No se genera la opción D (heatmap global), porque cada paso secuencial tiene
  una configuración base distinta y las corridas no son directamente
  comparables en una única matriz.
- Los CSV quedan en la raíz de `kohonen_sweep/`. Las dos carpetas de figuras
  permiten comparar y elegir cuál va a la presentación.

### Resultados finales

- La corrida final usa `config_optima.toml` y guarda en
  `resultados/ej1/kohonen/`.
- Se generan tres figuras:
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

### Salidas

- Primer barrido archivado: `resultados/ej1/kohonen_sweep/sweep1/`.
- Próximo barrido: raíz de `resultados/ej1/kohonen_sweep/`.
- Corrida final: `resultados/ej1/kohonen/`.
- Corridas neutras manuales: `resultados/ej1/kohonen_base/`.
- El barrido guarda `config_barrido.toml`, `config_optima.toml`, `registro.txt`,
  `resumen_pasos.csv`, los CSV por paso, `acoplamientos/` y las dos carpetas de
  figuras habilitadas (A y B).
- La corrida final guarda `config_usada.toml`, `normalizacion.csv`, `pesos.csv`,
  `neuronas.csv`, `asignaciones.csv`, `matriz_u.csv`, `metricas.csv`,
  `resumen.txt` y las tres figuras.

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
  `0.3` y `0.4` a `radio_inicial_relativo`. El próximo barrido escribe en la
  raíz de `resultados/ej1/kohonen_sweep/`; `sweep1/` queda como archivo.
- `eta_inicial` e `iteraciones` se dejan como están por ahora; el próximo
  barrido vuelve a decidirlos.
- `config_optima.toml` de `sweep1/` queda como configuración provisoria del
  primer barrido y no se usa todavía para la corrida final.

### Pendientes de 1.2

- Fase B: completada.
- Fase C: completada.
- Fase D: completada.
- Fase E.1: resultados del primer barrido archivados en `sweep1/`; completada.
- Fases E.2 a E.6: pendientes (nuevo rango de grilla y radio, subrayado del
  valor elegido, barra de progreso, opción C desactivada y nuevo barrido).
- Fase F: pendiente de correr la configuración final.

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
- `python3 -m pytest -q`: 50 tests en verde.
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
