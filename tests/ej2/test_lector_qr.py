"""Tests del ejercicio 2.1: lectura de estados con un lector de QR."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "lector_qr.py"
RUTA_QR = RAIZ / "datos" / "hopfield" / "qr_37x37.csv"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el lector de QR."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("lector_qr", RUTA_SCRIPT)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


@pytest.fixture(scope="module")
def patrones() -> pd.DataFrame:
    datos = pd.read_csv(RUTA_QR).set_index("nombre")
    return datos[[c for c in datos.columns if c.startswith("p")]]


def test_imagen(modulo: ModuleType) -> None:
    """Zona de silencio de 4 módulos, ×10 con vecino más cercano y 1 en negro."""
    estado = -np.ones(9, dtype=np.int8)
    estado[4] = 1
    imagen = modulo.a_imagen(estado, 3, 3)
    assert imagen.shape == ((3 + 8) * 10, (3 + 8) * 10)
    assert imagen.dtype == np.uint8
    assert set(np.unique(imagen)) == {0, 255}
    negros = np.argwhere(imagen == 0)
    assert negros.min(axis=0).tolist() == [50, 50]
    assert negros.max(axis=0).tolist() == [59, 59]


def test_lee_los_originales(modulo: ModuleType, patrones: pd.DataFrame) -> None:
    """Los cinco QR originales se leen y cada uno da una URL distinta."""
    textos = [modulo.leer(fila.to_numpy(), 37, 37) for _, fila in patrones.iterrows()]
    assert all(texto.startswith("https://") for texto in textos)
    assert len(set(textos)) == 5
    for nombre, texto in zip(patrones.index, textos, strict=True):
        assert nombre.replace("_qr", "") in texto


def test_no_lee_estados_sin_qr(modulo: ModuleType) -> None:
    """Un estado en blanco o al azar no se lee."""
    assert modulo.leer(-np.ones(37 * 37, dtype=np.int8), 37, 37) == ""
    azar = np.random.default_rng(0).choice([-1, 1], size=37 * 37)
    assert modulo.leer(azar, 37, 37) == ""
