"""Tests del ejercicio 1.1: EDA y PCA sobre creadores de contenido."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "eda-pca.py"
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script como módulo para poder probar sus funciones."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("eda_pca", RUTA_SCRIPT)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


def test_descomposicion_ejemplo_teoria(modulo: ModuleType) -> None:
    """Verifica el ejemplo de la teoría: autovalores 6, 3 y 2."""
    matriz = np.array(
        [
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 1.0, 5.0],
        ]
    )

    autovalores, autovectores = modulo.descomponer_matriz(matriz)

    assert np.allclose(autovalores, [6.0, 3.0, 2.0])
    assert np.allclose(autovalores / autovalores.sum(), [6 / 11, 3 / 11, 2 / 11])
    assert np.allclose(autovectores @ np.diag(autovalores) @ autovectores.T, matriz)


def test_estandarizacion(modulo: ModuleType) -> None:
    """Cada columna estandarizada debe tener media 0 y desvío 1."""
    datos = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 10.0]])

    estandarizados, medias, desvios = modulo.estandarizar(datos)

    assert np.allclose(estandarizados.mean(axis=0), 0.0)
    assert np.allclose(estandarizados.std(axis=0, ddof=1), 1.0)
    assert np.allclose(medias, datos.mean(axis=0))
    assert np.allclose(desvios, datos.std(axis=0, ddof=1))


@pytest.mark.parametrize(
    ("umbral", "esperado"),
    [
        (0.50, 1),
        (0.80, 2),
        (0.99, 3),
    ],
)
def test_seleccionar_q(modulo: ModuleType, umbral: float, esperado: int) -> None:
    """La cantidad de componentes debe ser la primera que alcanza el umbral."""
    autovalores = np.array([5.0, 3.0, 2.0])

    assert modulo.seleccionar_q(autovalores, umbral) == esperado


def test_deteccion_de_redundancias(modulo: ModuleType) -> None:
    """Detecta duplicados, derivadas, sumas y complementos exactos."""
    datos = pd.DataFrame(
        {
            "creator_id": ["a", "b"],
            "name": ["@a", "@b"],
            "followers": [100, 200],
            "price_usd": [10.0, 20.0],
            "price_ars": [15000, 30000],
            "average_reach": [100.0, 200.0],
            "argentina_audience_pct": [50.0, 50.0],
            "estimated_argentina_reach": [50, 100],
            "likes_per_1000": [1.0, 2.0],
            "comments_per_1000": [1.0, 2.0],
            "saves_per_1000": [1.0, 2.0],
            "shares_per_1000": [1.0, 2.0],
            "total_engagement_per_1000": [4.0, 8.0],
            "sponsored_posts_pct": [10.0, 20.0],
            "organic_posts_pct": [90.0, 80.0],
            "topic_football_pct": [10.0, 20.0],
            "topic_makeup_pct": [20.0, 30.0],
            "topic_dogs_pct": [30.0, 20.0],
            "topic_art_pct": [40.0, 30.0],
            "content_tutorial_pct": [10.0, 20.0],
            "content_review_pct": [20.0, 30.0],
            "content_entertainment_pct": [30.0, 20.0],
            "content_showcase_pct": [40.0, 30.0],
        }
    )

    tabla, excluidas = modulo.detectar_redundancias(datos)

    assert set(excluidas) == {
        "price_ars",
        "estimated_argentina_reach",
        "total_engagement_per_1000",
        "organic_posts_pct",
    }
    assert "topic_art_pct" not in excluidas
    assert "content_showcase_pct" not in excluidas
    assert {"precio", "derivada", "suma", "complemento"} <= set(tabla["tipo"])


def test_integracion_dataset_real(modulo: ModuleType) -> None:
    """Comprueba dimensiones, rango y umbrales sobre el dataset real."""
    datos = modulo.cargar_datos(RUTA_DATOS)
    columnas = modulo.columnas_para_pca(datos)

    assert len(columnas) == 27
    assert "topic_art_pct" in columnas
    assert "content_showcase_pct" in columnas

    for umbral, esperado in [(0.90, 10), (0.95, 13), (0.99, 18)]:
        resultado = modulo.ajustar_pca(datos, columnas, umbral)
        assert resultado.q == esperado
        assert len(resultado.autovalores_pos) == 25
        assert np.isclose(resultado.autovalores_pos.sum(), 27.0)
        assert np.isclose(resultado.acumulada[-1], 1.0)


def test_varianza_invalida() -> None:
    """Un umbral fuera de (0, 1] debe terminar con error."""
    resultado = subprocess.run(
        [sys.executable, str(RUTA_SCRIPT), "--varianza", "1.5"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert resultado.returncode != 0
    assert "varianza" in (resultado.stderr + resultado.stdout).lower()
