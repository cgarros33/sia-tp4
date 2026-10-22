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
├── docs/
│   ├── structure.md
│   ├── decisiones.md
│   ├── plan/
│   │   ├── ej1-1.md
│   │   ├── ej1-2.md
│   │   └── ej1-3.md
│   └── context/
│       ├── gusslop.md
│       └── consigna/
│           ├── SIATP42026.pdf
│           ├── artificializando.pdf
│           └── artif-eng.pdf
├── resultados/
│   └── ej1/
│       ├── eda/
│       ├── pca_<umbral>/
│       ├── kohonen_sweep/
│       │   ├── sweep1/
│       │   ├── sweep2/
│       │   ├── sweep3/
│       │   ├── sweep4/
│       │   └── sweep5/
│       ├── kohonen_k3/
│       ├── kohonen_k4/
│       ├── kohonen_k5/
│       └── oja/
├── src/
│   └── ej1/
│       ├── eda-pca.py
│       ├── kohonen.py
│       ├── kohonen-sweep.py
│       ├── kohonen.toml
│       ├── kohonen-sweep.toml
│       ├── oja.py
│       └── oja.toml
└── tests/
    └── ej1/
        ├── test_eda_pca.py
        ├── test_kohonen.py
        ├── test_kohonen_sweep.py
        └── test_oja.py
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
- Configuración neutra: `src/ej1/kohonen.toml`.
- Configuración del barrido: `src/ej1/kohonen-sweep.toml`.
- Plan: `docs/plan/ej1-2.md`.
- Tests: `tests/ej1/test_kohonen.py` y `tests/ej1/test_kohonen_sweep.py`.
- Barridos archivados: `resultados/ej1/kohonen_sweep/sweep1/` y `sweep2/`.
- Barridos de grilla fija: `sweep3/` (`k=3`), `sweep4/` (`k=4`) y
  `sweep5/` (`k=5`).
- Corridas finales: `resultados/ej1/kohonen_k3/`, `kohonen_k4/` y
  `kohonen_k5/`.

La grilla se fija por corrida con `grillas_fijas = [3, 4, 5]`; no se barre
`grilla`. Para cada tamaño se corre el mismo barrido secuencial y después se
prueban los acoplamientos fuertes.

### Ejecución

```bash
# Barridos completos para k = 3, 4 y 5
python src/ej1/kohonen-sweep.py --config src/ej1/kohonen-sweep.toml

# Corridas finales con el mejor config_optima de cada tamaño
python src/ej1/kohonen.py \
  --config resultados/ej1/kohonen_sweep/sweep3/config_optima.toml
python src/ej1/kohonen.py \
  --config resultados/ej1/kohonen_sweep/sweep4/config_optima.toml
python src/ej1/kohonen.py \
  --config resultados/ej1/kohonen_sweep/sweep5/config_optima.toml
```

Los hiperparámetros y las rutas se pueden pisar por línea de comandos. La
precedencia es: CLI > archivo TOML > valores por defecto.

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
- Salidas: `resultados/ej1/oja/`.

El script lee los resultados de PCA y correlaciones de 1.1 desde
`resultados/ej1/pca_0.95/` y `resultados/ej1/eda/correlaciones.csv`; no
recalcula PC1. La corrida final usa la regla exacta de Oja. Las corridas de
Taylor con `eta_inicial = 0.5` y `0.05` se usan solo como comparación, y sus
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

## Estado

| Ejercicio | Subejercicio | Script | Estado |
|---|---|---|---|
| 1 | 1. Exploración y representación | `src/ej1/eda-pca.py` | Implementado |
| 1 | 2. Red de Kohonen | `src/ej1/kohonen.py` + `src/ej1/kohonen-sweep.py` | Implementado |
| 1 | 3. Índice de Oja | `src/ej1/oja.py` | Implementado |
| 1 | 4. Decisión para las campañas | — | Pendiente |
| 2 | Memoria asociativa (Hopfield) | — | Pendiente |
