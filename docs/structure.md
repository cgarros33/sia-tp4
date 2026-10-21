# Estructura del proyecto

Este archivo describe cómo está organizado el repositorio y para qué sirve cada
carpeta. No reemplaza la consigna ni las notas de `docs/context/`: solo explica
dónde está cada cosa.

## Árbol

```text
sia-tp4/
├── CLAUDE.md
├── README.md
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

### Tests

```bash
python3 -m pytest -q tests/ej1
```

### Dependencias

- Python 3.11 o superior.
- `numpy`, `pandas`, `matplotlib` para el script.
- `pytest` para los tests.
- No se usa scikit-learn ni otra librería de machine learning.

## Estado

| Ejercicio | Subejercicio | Script | Estado |
|---|---|---|---|
| 1 | 1. Exploración y representación | `src/ej1/eda-pca.py` | Implementado |
| 1 | 2. Red de Kohonen | — | Pendiente |
| 1 | 3. Índice de Oja | — | Pendiente |
| 1 | 4. Decisión para las campañas | — | Pendiente |
| 2 | Memoria asociativa (Hopfield) | — | Pendiente |
