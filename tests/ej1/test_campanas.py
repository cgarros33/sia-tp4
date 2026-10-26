"""Tests del análisis de decisión para las campañas (ejercicio 1.4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "campanas.py"
RUTA_CONFIG = RAIZ / "src" / "ej1" / "campanas.toml"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script de campañas como módulo."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("campanas", RUTA_SCRIPT)
    assert especificacion is not None and especificacion.loader is not None
    cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = cargado
    especificacion.loader.exec_module(cargado)
    return cargado


def _creadores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_usd": [100.0, 1000.0, 50.0],
            "estimated_argentina_reach": [5000.0, 20000.0, 1000.0],
            "topic_football_pct": [80.0, 25.0, 50.0],
            "saves_per_1000": [10.0, 2.0, 30.0],
            "profile_visits_per_1000": [10.0, 3.0, 20.0],
            "argentina_audience_pct": [50.0, 80.0, 90.0],
        },
        index=["A", "B", "C"],
    )


def test_metricas_del_cliente(modulo: ModuleType) -> None:
    """Argentinos alcanzados en el tópico, su costo e intención."""
    datos = modulo.agregar_metricas(_creadores(), "topic_football_pct")
    np.testing.assert_allclose(datos["argentinos_topico"], [4000.0, 5000.0, 500.0])
    np.testing.assert_allclose(datos["usd_por_mil_argentinos"], [25.0, 200.0, 100.0])
    np.testing.assert_allclose(datos["intencion"], [20.0, 5.0, 50.0])


def test_metricas_de_una_seleccion(modulo: ModuleType) -> None:
    """Costo, alcance útil e intención de una selección."""
    datos = modulo.agregar_metricas(_creadores(), "topic_football_pct")
    metricas = modulo.metricas_seleccion(datos, ["A", "C"], "topic_football_pct")
    assert metricas["costo_total"] == pytest.approx(150.0)
    assert metricas["argentinos_topico"] == pytest.approx(4500.0)
    assert metricas["usd_por_mil_argentinos"] == pytest.approx(150 / 4500 * 1000)
    # Intención: 20/1000·4000 + 50/1000·500 = 105 acciones por 150 USD.
    assert metricas["intencion_cada_100_usd"] == pytest.approx(105 / 150 * 100)
    assert metricas["afinidad_media"] == pytest.approx(65.0)


def test_puntaje_respeta_signo_de_los_pesos(modulo: ModuleType) -> None:
    """Un peso negativo sobre el costo invierte el orden."""
    datos = modulo.agregar_metricas(_creadores(), "topic_football_pct")
    caro = modulo.puntaje(datos, {"usd_por_mil_argentinos": 1.0})
    barato = modulo.puntaje(datos, {"usd_por_mil_argentinos": -1.0})
    assert caro.idxmax() == "B"
    assert barato.idxmax() == "A"
    np.testing.assert_allclose(caro, -barato)


def test_topico_dominante_por_neurona(modulo: ModuleType) -> None:
    """Cada neurona se etiqueta con el tópico de mayor promedio."""
    datos = pd.DataFrame(
        {
            "creator_id": ["A", "B", "C"],
            "topic_x": [80.0, 10.0, 30.0],
            "topic_y": [20.0, 90.0, 70.0],
        }
    )
    asignaciones = pd.DataFrame({"creator_id": ["A", "B", "C"], "neurona": [0, 1, 1]})
    regiones = modulo.topico_dominante(datos, asignaciones, ["topic_x", "topic_y"])
    assert regiones.to_dict() == {0: "topic_x", 1: "topic_y"}


def test_seleccionar_toma_los_mejores_del_grupo(modulo: ModuleType) -> None:
    """Se eligen los N de mayor puntaje entre los candidatos."""
    puntajes = pd.Series([0.1, 0.9, 0.5, 0.7], index=["A", "B", "C", "D"])
    assert modulo.seleccionar(puntajes, ["A", "C", "D"], 2) == ["D", "C"]
    assert modulo.seleccionar(puntajes, ["A"], 3) == ["A"]


def test_seleccionar_con_presupuesto(modulo: ModuleType) -> None:
    """Se agregan en orden los que entran en el presupuesto."""
    precios = pd.Series({"A": 60.0, "B": 50.0, "C": 30.0, "D": 10.0})
    assert modulo.seleccionar_con_presupuesto(["A", "B", "C", "D"], precios, 100.0) == [
        "A",
        "C",
        "D",
    ]
    assert modulo.seleccionar_con_presupuesto(["D"], precios, 100.0) == ["D"]


def test_jaccard(modulo: ModuleType) -> None:
    """Coincidencia entre dos selecciones."""
    assert modulo.jaccard(["A", "B"], ["B", "C"]) == pytest.approx(1 / 3)
    assert modulo.jaccard(["A"], ["A"]) == 1.0


def test_ubicar_creadores(modulo: ModuleType) -> None:
    """La neurona ganadora es la de pesos más cercanos."""
    pesos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    puntos = np.array([[1.0, 1.0], [9.0, 2.0], [0.0, 7.0]])
    np.testing.assert_array_equal(modulo.ubicar(puntos, pesos, "euclidiana"), [0, 1, 2])


def test_perturbar_mantiene_rangos(modulo: ModuleType) -> None:
    """Sin ruido no cambia nada; con ruido los porcentajes quedan en [0, 100]."""
    datos = pd.DataFrame({"followers": [10.0, 20.0], "algo_pct": [99.0, 1.0]})
    generador = np.random.default_rng(0)
    igual = modulo.perturbar(datos, 0.0, generador)
    pd.testing.assert_frame_equal(igual, datos)
    ruidoso = modulo.perturbar(datos, 5.0, generador)
    assert (ruidoso >= 0).all().all()
    assert (ruidoso["algo_pct"] <= 100).all()


def test_corrida_completa(modulo: ModuleType, tmp_path: Path) -> None:
    """Con los datos reales genera tablas y figuras y reproduce el mapa."""
    codigo = modulo.main(
        ["--config", str(RUTA_CONFIG), "--salida", str(tmp_path), "--repeticiones", "2"]
    )
    assert codigo == 0
    for nombre in [
        "selecciones.csv",
        "comparacion.csv",
        "creadores_por_cliente.csv",
        "actualizacion.csv",
        "resumen.txt",
        "fig_mapa_regiones.png",
        "fig_enfoques.png",
        "fig_followers.png",
        "fig_comparacion.png",
        "fig_actualizacion.png",
    ]:
        assert (tmp_path / nombre).exists(), nombre
    comparacion = pd.read_csv(
        tmp_path / "comparacion.csv", header=[0, 1], index_col=[0, 1]
    )
    assert set(comparacion.index.get_level_values(1)) == {"followers", "propuesta"}
    actualizacion = pd.read_csv(tmp_path / "actualizacion.csv")
    assert actualizacion["cambia_neurona_media"].between(0, 1).all()
