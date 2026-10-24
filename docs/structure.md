# Estructura del proyecto

Este archivo describe cómo está organizado el repositorio y para qué sirve cada
carpeta. No reemplaza la consigna ni las notas de `docs/context/`: solo explica
dónde está cada cosa.

## Árbol

```text
sia-tp4/
├── CLAUDE.md
├── README.md
├── check-cicd.py
├── requirements.txt
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── ci.yml
├── datos/
│   ├── creadores.csv
│   └── hopfield/
│       ├── abecedario_5x5.csv
│       ├── letras_5x5.csv
│       ├── qr_37x37.csv
│       ├── beverages_menu_qr.png
│       ├── dessert_menu_qr.png
│       ├── food_menu_qr.png
│       ├── specials_menu_qr.png
│       └── wine_menu_qr.png
├── docs/
│   ├── structure.md
│   ├── decisiones.md
│   ├── sweep.md
│   ├── plan/
│   │   ├── ej1-1.md
│   │   ├── ej1-2.md
│   │   ├── ej1-3.md
│   │   └── ej2-2.md
│   └── context/
│       ├── gusslop.md
│       └── consigna/
│           ├── SIATP42026.pdf
│           ├── artificializando.pdf
│           └── artif-eng.pdf
├── resultados/
│   ├── ej1/
│   │   ├── eda/
│   │   ├── pca_<umbral>/
│   │   ├── ej1.2-full/
│   │   │   ├── kohonen_sweep/
│   │   │   │   ├── sweep1/
│   │   │   │   ├── sweep2/
│   │   │   │   ├── sweep3/
│   │   │   │   ├── sweep4/
│   │   │   │   └── sweep5/
│   │   │   ├── kohonen_k3/
│   │   │   ├── kohonen_k4/
│   │   │   └── kohonen_k5/
│   │   ├── ej1.2-pca10/
│   │   │   ├── kohonen_sweep/
│   │   │   │   ├── sweep3/
│   │   │   │   ├── sweep4/
│   │   │   │   └── sweep5/
│   │   │   ├── kohonen_sweep_euclidiana/
│   │   │   │   └── sweep4/
│   │   │   ├── kohonen_k3/
│   │   │   ├── kohonen_k4/
│   │   │   ├── kohonen_k4_euclidiana/
│   │   │   └── kohonen_k5/
│   │   ├── ej1.2-comparacion/
│   │   └── oja/
│   │       ├── epocas_5/
│   │       └── epocas_500/
│   └── ej2/
│       └── hopfield/
│           └── letras_5x5/
│               ├── consultas/
│               │   ├── limpias/
│               │   └── ruido_<nivel>_semilla_<semilla>/
│               ├── config_usada.toml
│               ├── patrones.csv
│               ├── pesos.npz
│               ├── estabilidad.csv
│               ├── estabilidad_neuronas.csv
│               ├── solapamiento.csv
│               ├── hamming.csv
│               ├── ortogonalidad_resumen.csv
│               ├── capacidad.csv
│               ├── energia_entrenamiento.csv
│               └── resumen.txt
├── src/
│   ├── ej1/
│   │   ├── eda-pca.py
│   │   ├── kohonen.py
│   │   ├── kohonen-sweep.py
│   │   ├── kohonen.toml
│   │   ├── kohonen-sweep.toml
│   │   ├── kohonen-pca10.toml
│   │   ├── kohonen-sweep-pca10.toml
│   │   ├── kohonen-sweep-pca10-euclidiana.toml
│   │   ├── kohonen-comparacion.py
│   │   ├── oja.py
│   │   └── oja.toml
│   └── ej2/
│       ├── hopfield.py
│       ├── hopfield.toml
│       ├── hopfield-runner.py
│       └── qr-to-matrix.py
└── tests/
    ├── ej1/
    │   ├── test_eda_pca.py
    │   ├── test_kohonen.py
    │   ├── test_kohonen_sweep.py
    │   ├── test_kohonen_comparacion.py
    │   └── test_oja.py
    └── ej2/
        ├── test_hopfield.py
        ├── test_hopfield_runner.py
        └── test_qr_to_matrix.py
```

## Convenciones

- `datos/` contiene las entradas del trabajo. Es de **solo lectura**: ningún
  script debe modificarlas.
- `src/ejN/` contiene los scripts de cada ejercicio. Los nombres de archivo
  pueden usar guiones, como `eda-pca.py`.
- `tests/ejN/` contiene los tests de cada ejercicio. Se ejecutan con `pytest`.
- `resultados/ejN/` contiene las salidas generadas por los scripts: tablas,
  figuras y resúmenes. Se pueden regenerar; no son datos de entrada.
- `check-cicd.py` ejecuta localmente los mismos chequeos que el CI.
- `pyproject.toml` configura `pytest` y `ruff`.
- `.github/workflows/` contiene los workflows de GitHub Actions.
- `docs/` contiene documentación del proyecto:
  - `docs/plan/`: planes de trabajo aprobados.
  - `docs/decisiones.md`: registro simple de decisiones y estado actual.
  - `docs/structure.md`: este archivo.
  - `docs/sweep.md`: análisis del barrido de Kohonen (empates de TE, régimen
    de capacidad, iteraciones bajas, suelo de k-means y normalización min–max).
  - `docs/context/`: consigna, teoría y notas personales. Es la fuente de
    verdad para las reglas del trabajo.

## Ejercicio 1.1 — EDA y PCA

- Script: `src/ej1/eda-pca.py`.
- Plan: `docs/plan/ej1-1.md`.
- Tests: `tests/ej1/test_eda_pca.py`.
- Salidas de EDA: `resultados/ej1/eda/`.
- Salidas de PCA: `resultados/ej1/pca_<umbral>/`.
- Corridas generadas: `pca_0.90/`, `pca_0.95/` y `pca_0.99/`.

### Ejecución

```bash
python src/ej1/eda-pca.py --varianza 0.90
python src/ej1/eda-pca.py --varianza 0.95
python src/ej1/eda-pca.py --varianza 0.99
```

`--varianza` es una fracción entre 0 y 1. Por ejemplo, `0.95` significa
conservar al menos el 95% de la varianza total. El valor por defecto es `0.95`.

## Ejercicio 1.2 — Red de Kohonen

- Motor: `src/ej1/kohonen.py`.
- Barrido: `src/ej1/kohonen-sweep.py`.
- Configuración actual (10 componentes): `src/ej1/kohonen-pca10.toml`.
- Configuración del barrido actual: `src/ej1/kohonen-sweep-pca10.toml`.
- Configuración archivada con las 27 variables: `src/ej1/kohonen.toml` y
  `src/ej1/kohonen-sweep.toml`.
- Plan: `docs/plan/ej1-2.md`.
- Tests: `tests/ej1/test_kohonen.py` y `tests/ej1/test_kohonen_sweep.py`.
- Corrida actual: `resultados/ej1/ej1.2-pca10/`.
- Corrida archivada con todas las variables: `resultados/ej1/ej1.2-full/`.

La corrida actual usa `PC1` a `PC10` de
`resultados/ej1/pca_0.90/scores.csv` (10 componentes, 91,53% de varianza
acumulada). El PCA no se recalcula. Los scores se usan tal como salen de 1.1,
sin reestandarizar (`datos.estandarizar = false`), para conservar el peso
relativo de cada componente. Por eso la QE queda en unidades de score de PCA y
no es comparable con la de la corrida archivada.

La grilla se fija por corrida con `grillas_fijas = [3, 4, 5]`; no se barre
`grilla`. Para cada tamaño se corre el mismo barrido secuencial y después se
prueban los acoplamientos fuertes.

### Ejecución

```bash
# Barrido actual con PC1..PC10 para k = 3, 4 y 5
python3 src/ej1/kohonen-sweep.py --config src/ej1/kohonen-sweep-pca10.toml

# Corridas finales con el mejor config_optima de cada tamaño
python3 src/ej1/kohonen.py \
  --config resultados/ej1/ej1.2-pca10/kohonen_sweep/sweep3/config_optima.toml
python3 src/ej1/kohonen.py \
  --config resultados/ej1/ej1.2-pca10/kohonen_sweep/sweep4/config_optima.toml
python3 src/ej1/kohonen.py \
  --config resultados/ej1/ej1.2-pca10/kohonen_sweep/sweep5/config_optima.toml
```

Los hiperparámetros y las rutas se pueden pisar por línea de comandos. La
precedencia es: CLI > archivo TOML > valores por defecto.

Cada corrida final guarda, además de los archivos de la semilla
representativa, `pesos_semillas.csv` y `asignaciones_semillas.csv` con los
resultados de todas las semillas.

### Control de k = 4 con distancia euclidiana

```bash
python3 src/ej1/kohonen-sweep.py \
  --config src/ej1/kohonen-sweep-pca10-euclidiana.toml
python3 src/ej1/kohonen.py \
  --config resultados/ej1/ej1.2-pca10/kohonen_sweep_euclidiana/sweep4/config_optima.toml \
  --salida resultados/ej1/ej1.2-pca10/kohonen_k4_euclidiana
```

### Comparación 27 variables contra PC1..PC10

- Script: `src/ej1/kohonen-comparacion.py` (solo lee corridas finales).
- Tests: `tests/ej1/test_kohonen_comparacion.py`.
- Salida: `resultados/ej1/ej1.2-comparacion/` con `metricas_por_semilla.csv`,
  `metricas_resumen.csv`, `concordancia.csv`, `resumen.txt`,
  `fig_metricas.png`, `fig_concordancia.png` y `fig_topografia.png`.

```bash
python3 src/ej1/kohonen-comparacion.py
```

### Tests y lint

La forma más simple de correr todo como en el CI:

```bash
python3 check-cicd.py
```

El script crea o reutiliza `.venv`, instala o actualiza `requirements.txt` si
hace falta y ejecuta `pip check`, `ruff check .`, `ruff format --check .` y
`pytest -q`.

Los comandos individuales son:

```bash
python3 -m pytest -q
ruff check .
ruff format --check .
```

Para correr solo los tests del ejercicio 1:

```bash
python3 -m pytest -q tests/ej1
```

### Dependencias

- Python 3.11 o superior.
- `numpy`, `pandas`, `matplotlib` para el script.
- `pytest` para los tests.
- `ruff` para lint y formato.
- Las versiones usadas están en `requirements.txt`.
- No se usa scikit-learn ni otra librería de machine learning.

### Integración continua

El workflow `.github/workflows/ci.yml` corre en cada push y pull request:

1. Instala `requirements.txt`.
2. Ejecuta `python -m pip check`.
3. Ejecuta `ruff check .`.
4. Ejecuta `ruff format --check .`.
5. Ejecuta `python -m pytest -q`.

`python3 check-cicd.py` replica estos pasos en local, creando o actualizando
`.venv` si hace falta.

Para exigir que el chequeo pase antes de mergear, activar la protección de la
rama principal en GitHub: **Settings → Branches → Require status checks to pass
before merging** y seleccionar `Calidad`.

## Ejercicio 1.3 — Índice de Oja

- Script: `src/ej1/oja.py`.
- Configuración: `src/ej1/oja.toml`.
- Plan: `docs/plan/ej1-3.md`.
- Tests: `tests/ej1/test_oja.py`.
- Salida final: `resultados/ej1/oja/epocas_5/`.
- Corrida archivada: `resultados/ej1/oja/epocas_500/`.

El script lee los resultados de PCA y correlaciones de 1.1 desde
`resultados/ej1/pca_0.95/` y `resultados/ej1/eda/correlaciones.csv`; no
recalcula PC1. La corrida final usa la regla exacta de Oja con 5 épocas
(250 actualizaciones), para que la convergencia sea visible y el índice no
coincida exactamente con PC1. La corrida anterior de 500 épocas quedó archivada
en `epocas_500/` y no se vuelve a ejecutar. Las corridas de Taylor con
`eta_inicial = 0.5` y `0.05` se usan solo como comparación, y sus
configuraciones se guardan únicamente si convergen.

### Ejecución

```bash
python src/ej1/oja.py --config src/ej1/oja.toml
```

Los parámetros y las rutas se pueden pisar por línea de comandos. La
precedencia es: CLI > archivo TOML > valores por defecto. Ejemplo:

```bash
python src/ej1/oja.py --config src/ej1/oja.toml \
  --regla taylor --eta-inicial 0.05 \
  --salida /tmp/opencode/oja_taylor_005
```

### Salidas principales

En `resultados/ej1/oja/epocas_5/` (corrida final):

- `config_usada.toml` y, si convergen, `config_regla_taylor_0.5.toml` y
  `config_regla_taylor_0.05.toml`.
- `normalizacion.csv`, `pesos.csv`, `indice.csv`.
- `comparacion_pc1.csv`, `correlaciones_indice.csv`.
- `convergencia.csv`, `sensibilidad.csv`, `sensibilidad_resumen.csv`.
- `comparacion_reglas.csv`, `comparacion_reglas_resumen.csv`.
- `resumen.txt`.
- Figuras `fig_convergencia.png`, `fig_pesos.png`, `fig_indice.png`,
  `fig_comparacion_pc1.png`, `fig_indice_vs_negocio.png`,
  `fig_sensibilidad.png` y `fig_comparacion_reglas.png`.

En `resultados/ej1/oja/epocas_500/` quedó archivada la corrida anterior con 500
épocas, con los mismos tipos de archivo y sus configuraciones resueltas.

## Ejercicio 2.2 — Recomendación (Hopfield de 5 × 5)

- Motor: `src/ej2/hopfield.py`.
- Runner: `src/ej2/hopfield-runner.py`.
- Conversor de QR: `src/ej2/qr-to-matrix.py`.
- Configuración: `src/ej2/hopfield.toml`.
- Plan: `docs/plan/ej2-2.md`.
- Tests: `tests/ej2/test_hopfield.py`,
  `tests/ej2/test_hopfield_runner.py` y `tests/ej2/test_qr_to_matrix.py`.
- Datos de letras: `datos/hopfield/abecedario_5x5.csv` y
  `datos/hopfield/letras_5x5.csv`.
- Matrices de QR: `datos/hopfield/qr_37x37.csv`.
- Salida de la corrida final: `resultados/ej2/hopfield/letras_5x5/`.

El motor implementa una red de Hopfield discreta con estados `-1`/`1`,
almacenamiento Hebbiano en forma cerrada y recuperación asincrónica secuencial
por defecto; también admite el modo sincrónico. El dataset de 2.2 contiene las
letras `J`, `O`, `T` y `A` en matrices de 5 × 5. El conversor transforma los
PNG de 37 × 37 en filas del CSV, con negro → `1` y blanco → `-1`.

### Ejecución

```bash
# Entrenamiento y tablas de la red
python src/ej2/hopfield.py --config src/ej2/hopfield.toml

# Consultas limpias
python src/ej2/hopfield-runner.py \
  --red resultados/ej2/hopfield/letras_5x5 \
  --salida resultados/ej2/hopfield/letras_5x5/consultas/limpias

# Barrido de ruido: 5 niveles × 3 semillas × 5 repeticiones
for nivel in 0.1 0.2 0.3 0.4 0.5; do
  for semilla in 0 1 2; do
    python src/ej2/hopfield-runner.py \
      --red resultados/ej2/hopfield/letras_5x5 \
      --salida "resultados/ej2/hopfield/letras_5x5/consultas/ruido_${nivel}_semilla_${semilla}" \
      --ruido "$nivel" --semilla "$semilla" --repeticiones 5
  done
done

# Ejemplo con salida gráfica
python src/ej2/hopfield-runner.py \
  --red resultados/ej2/hopfield/letras_5x5 \
  --ruido 0.2 --semilla 0 --repeticiones 2 --graphic
```

### Salidas principales

- `config_usada.toml`, `patrones.csv` y `pesos.npz`.
- `estabilidad.csv`, `estabilidad_neuronas.csv`, `solapamiento.csv`,
  `hamming.csv`, `ortogonalidad_resumen.csv`, `capacidad.csv`,
  `energia_entrenamiento.csv` y `resumen.txt`.
- `consultas/limpias/` con `consultas.csv`, `energia.csv`,
  `estados_iniciales.csv`, `estados_finales.csv` y `resumen.txt`.
- `consultas/ruido_<nivel>_semilla_<semilla>/` para los 15 barridos de ruido,
  con los mismos archivos.
- No se generan figuras en esta etapa.

Los resultados numéricos y la interpretación técnica están en
`docs/decisiones.md`.

## Estado

| Ejercicio | Subejercicio | Script | Estado |
|---|---|---|---|
| 1 | 1. Exploración y representación | `src/ej1/eda-pca.py` | Implementado |
| 1 | 2. Red de Kohonen | `src/ej1/kohonen.py` + `src/ej1/kohonen-sweep.py` | Implementado |
| 1 | 3. Índice de Oja | `src/ej1/oja.py` | Implementado |
| 1 | 4. Decisión para las campañas | — | Pendiente |
| 2 | 1. Recuperación de QR deteriorados | — | Pendiente |
| 2 | 2. Recomendación (Hopfield 5×5) | `src/ej2/hopfield.py` + `src/ej2/hopfield-runner.py` | Implementado |
