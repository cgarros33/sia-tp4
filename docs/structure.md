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
│   │   └── ej1-1.md
│   └── context/
│       ├── gusslop.md
│       └── consigna/
│           ├── SIATP42026.pdf
│           ├── artificializando.pdf
│           └── artif-eng.pdf
├── resultados/
│   └── ej1/
│       ├── eda/
│       └── pca_<umbral>/
├── src/
│   └── ej1/
│       └── eda-pca.py
└── tests/
    └── ej1/
        └── test_eda_pca.py
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

Para correr solo los tests del ejercicio 1.1:

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

## Estado

| Ejercicio | Subejercicio | Script | Estado |
|---|---|---|---|
| 1 | 1. Exploración y representación | `src/ej1/eda-pca.py` | Implementado |
| 1 | 2. Red de Kohonen | — | Pendiente |
| 1 | 3. Índice de Oja | — | Pendiente |
| 1 | 4. Decisión para las campañas | — | Pendiente |
| 2 | Memoria asociativa (Hopfield) | — | Pendiente |
