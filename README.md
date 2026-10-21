**Sistemas de Inteligencia Artificial — ITBA — 2º cuatrimestre 2026**

| Integrante | Legajo |
|---|---|
| Celestino Garrós | 64375 |
| Leo Weitz | 64365 |
| Federico Ignacio Ruckauf | 64356 |
| Matías Romanato | 62072 |

## Requisitos

- Python 3.11 o superior.
- Las dependencias están en `requirements.txt`.

## Chequeo local del CI

La forma más simple de comprobar que el proyecto va a pasar el CI es:

```bash
python3 check-cicd.py
```

En Windows:

```powershell
python check-cicd.py
```

El script:

- crea `.venv` si no existe;
- se vuelve a ejecutar dentro del entorno virtual;
- actualiza pip y reinstala `requirements.txt` si cambió;
- corre `pip check`, `ruff check .`, `ruff format --check .` y `pytest -q`.

No hace falta activar el entorno virtual antes. Si las dependencias ya están al
día, no reinstala nada.

## Entorno virtual (manual)

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verificación manual

Los mismos chequeos se pueden correr a mano con el entorno virtual activado:

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

Para correr solo los tests del ejercicio 1.1:

```bash
python -m pytest -q tests/ej1
```

`-m` es una opción de Python para ejecutar un módulo; no es una opción de
pytest para indicar la carpeta. La forma correcta es `python -m pytest` seguido
de la ruta de los tests.

Para confirmar que se está usando el Python del entorno virtual:

```bash
python -c "import sys; print(sys.executable)"
python -c "import numpy; print(numpy.__version__)"
python -m pytest --version
```

El ejecutable debe estar dentro de `.venv`, `numpy` debe ser `2.4.6` y pytest
debe ser `8.3.5`. Si `pytest` muestra otra versión (por ejemplo, 9.x), se está
ejecutando un `pytest` global instalado con `pipx` o en `/usr/bin`, que
pertenece a otro Python y no ve las dependencias del entorno virtual. En ese
caso, usá siempre `python -m pytest` o llamá directamente a
`.venv/bin/python -m pytest`. También podés revisar si hay un alias o función
que esté tapando el comando:

```bash
type -a pytest
alias pytest
hash -r
```

## Lint y formato

El proyecto usa `ruff` para lint y formato. La configuración está en
`pyproject.toml`.

```bash
ruff check .
ruff format --check .
```

Si `ruff` no es el del entorno virtual, usá la forma de módulo:

```bash
python -m ruff check .
python -m ruff format --check .
```

Para aplicar correcciones automáticas:

```bash
ruff check --fix .
ruff format .
```

## Integración continua

El workflow de GitHub Actions está en `.github/workflows/ci.yml`. En cada push
y en cada pull request instala `requirements.txt` y ejecuta:

1. `python -m pip check`
2. `ruff check .`
3. `ruff format --check .`
4. `python -m pytest -q`

`python3 check-cicd.py` ejecuta localmente esos mismos pasos. Si el repositorio
está en GitHub, el workflow se ejecuta automáticamente. Para
exigir que pase antes de mergear, activá la protección de la rama principal en
**Settings → Branches** y seleccioná el chequeo `Calidad` como obligatorio.
