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

## Entorno virtual

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

## Verificación

Con el entorno virtual activado:

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
