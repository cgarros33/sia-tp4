"""Tests del script de figuras de Hopfield para la presentación."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "hopfield-figuras.py"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script de figuras como módulo."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location(
        "hopfield_figuras", RUTA_SCRIPT
    )
    assert especificacion is not None and especificacion.loader is not None
    cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = cargado
    especificacion.loader.exec_module(cargado)
    return cargado


def _consultas(tipos: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "consulta": [f"c{i}" for i in range(len(tipos))],
            "tipo_resultado": tipos,
        }
    )


def test_composicion_por_semilla(modulo: ModuleType, tmp_path: Path) -> None:
    """Fracción de cada tipo de resultado por nivel de ruido y semilla."""
    consultas = tmp_path / "consultas"
    casos = {
        "ruido_0.1_semilla_0": ["almacenado_correcto"] * 3 + ["espurio"],
        "ruido_0.1_semilla_1": ["almacenado_correcto"] * 2
        + ["almacenado_incorrecto"] * 2,
        "limpias": ["almacenado_correcto"] * 4,
    }
    for carpeta, tipos in casos.items():
        (consultas / carpeta).mkdir(parents=True)
        _consultas(tipos).to_csv(consultas / carpeta / "consultas.csv", index=False)

    tabla = modulo.composicion_por_semilla(consultas).set_index("semilla")
    assert list(tabla.index) == [0, 1]
    assert tabla.loc[0, "almacenado_correcto"] == pytest.approx(0.75)
    assert tabla.loc[0, "espurio"] == pytest.approx(0.25)
    assert tabla.loc[1, "almacenado_incorrecto"] == pytest.approx(0.5)
    assert tabla.loc[1, "espurio"] == pytest.approx(0.0)

    resumen = modulo.resumir_composicion(tabla.reset_index())
    assert resumen.loc[0.1, ("almacenado_correcto", "media")] == pytest.approx(0.625)
    assert resumen.loc[0.1, ("almacenado_correcto", "desvio")] == pytest.approx(
        pd.Series([0.75, 0.5]).std(ddof=1)
    )


def test_primer_caso(modulo: ModuleType) -> None:
    """Devuelve la primera consulta del tipo pedido."""
    consultas = _consultas(["almacenado_correcto", "espurio", "espurio"])
    assert modulo.primer_caso(consultas, "espurio")["consulta"] == "c1"
    with pytest.raises(ValueError):
        modulo.primer_caso(consultas, "almacenado_incorrecto")


def test_corrida_completa(modulo: ModuleType, tmp_path: Path) -> None:
    """Con los resultados guardados genera las tres figuras."""
    assert modulo.main(["--salida", str(tmp_path)]) == 0
    for nombre in [
        "fig_letras_ortogonalidad.png",
        "fig_letras_ruido.png",
        "fig_qr_fallas.png",
        "letras_composicion.csv",
    ]:
        assert (tmp_path / nombre).exists(), nombre
