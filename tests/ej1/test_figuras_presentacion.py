"""Tests del script de figuras de presentación del ejercicio 1."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "figuras-presentacion.py"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script como módulo."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location(
        "figuras_presentacion", RUTA_SCRIPT
    )
    assert especificacion is not None and especificacion.loader is not None
    cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = cargado
    especificacion.loader.exec_module(cargado)
    return cargado


def test_etiqueta_legible(modulo: ModuleType) -> None:
    """Los nombres de variables se acortan para el gráfico."""
    assert modulo.etiqueta("average_reach") == "reach"
    assert modulo.etiqueta("saves_per_1000") == "saves /1000"
    assert modulo.etiqueta("topic_art_pct") == "topic art %"


def test_orden_por_carga(modulo: ModuleType) -> None:
    """Las variables se ordenan de mayor a menor carga en la componente."""
    cargas = pd.DataFrame({"PC1": [0.1, -0.5, 0.7]}, index=["a", "b", "c"])
    assert modulo.orden_por_carga(cargas) == ["c", "a", "b"]


def test_formato_valor(modulo: ModuleType) -> None:
    """Los valores del barrido se muestran como en la presentación."""
    assert modulo.formato_valor("radio_inicial_relativo", 0.3) == "0,3·k"
    assert modulo.formato_valor("iteraciones", 10000.0) == "10.000"
    assert modulo.formato_valor("eta_inicial", 0.05) == "0,05"
    assert modulo.formato_valor("topologia", "rectangular") == "rectangular"


def test_mismo_valor(modulo: ModuleType) -> None:
    """Compara valores del barrido guardados como texto o número."""
    assert modulo.mismo_valor("0.3", 0.30000000)
    assert modulo.mismo_valor("lineal", "lineal")
    assert not modulo.mismo_valor("0.3", 0.4)


def test_corrida_completa(modulo: ModuleType, tmp_path: Path) -> None:
    """Con los resultados guardados genera las dos figuras."""
    assert modulo.main(["--salida", str(tmp_path)]) == 0
    assert (tmp_path / "fig_correlaciones_ordenadas.png").exists()
    assert (tmp_path / "fig_barrido.png").exists()
