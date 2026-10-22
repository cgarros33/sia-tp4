"""Tests del ejercicio 2.2: conversor de PNG a matriz binaria."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "qr-to-matrix.py"
RUTA_QR = RAIZ / "datos" / "hopfield" / "food_menu_qr.png"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script como módulo para probar sus funciones."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("qr_to_matrix", RUTA_SCRIPT)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


def _guardar_png(
    ruta: Path,
    pixeles: list[list[int]],
    modo: str = "L",
) -> None:
    """Guarda una imagen sintética con los valores indicados."""
    alto = len(pixeles)
    ancho = len(pixeles[0])
    imagen = Image.new(modo, (ancho, alto))
    for y, fila in enumerate(pixeles):
        for x, valor in enumerate(fila):
            if modo == "L":
                imagen.putpixel((x, y), valor)
            elif modo == "RGB":
                imagen.putpixel((x, y), (valor, valor, valor))
            elif modo == "RGBA":
                imagen.putpixel((x, y), (valor, valor, valor, 255))
            else:
                raise ValueError(f"Modo no soportado en el test: {modo}")
    imagen.save(ruta)


def _png_base(tmp_path: Path, nombre: str = "patron.png") -> Path:
    """Crea un PNG de 2x2 con dos píxeles negros y dos blancos."""
    ruta = tmp_path / nombre
    _guardar_png(ruta, [[0, 255], [255, 0]])
    return ruta


def test_leer_matriz_umbral(modulo: ModuleType, tmp_path: Path) -> None:
    """El umbral debe ser 128: 127 es negro y 128 es blanco."""
    ruta = tmp_path / "umbral.png"
    _guardar_png(ruta, [[0, 127], [128, 255]])

    matriz = modulo.leer_matriz(ruta)

    assert matriz.shape == (2, 2)
    assert np.issubdtype(matriz.dtype, np.integer)
    assert matriz.tolist() == [[1, 1], [-1, -1]]


@pytest.mark.parametrize("modo", ["RGB", "RGBA"])
def test_leer_matriz_colores(
    modulo: ModuleType,
    tmp_path: Path,
    modo: str,
) -> None:
    """Las imágenes RGB y RGBA se convierten a escala de grises."""
    ruta = tmp_path / f"color_{modo}.png"
    _guardar_png(ruta, [[0, 255], [255, 0]], modo=modo)

    matriz = modulo.leer_matriz(ruta)

    assert matriz.tolist() == [[1, -1], [-1, 1]]


def test_construir_tabla(modulo: ModuleType, tmp_path: Path) -> None:
    """La tabla debe tener una fila por PNG y los píxeles aplanados."""
    ruta_a = _png_base(tmp_path, "a.png")
    ruta_b = _png_base(tmp_path, "b.png")

    tabla = modulo.construir_tabla([ruta_a, ruta_b])

    assert list(tabla.columns) == [
        "nombre",
        "filas",
        "columnas",
        "p0000",
        "p0001",
        "p0002",
        "p0003",
    ]
    assert tabla["nombre"].tolist() == ["a", "b"]
    assert tabla["filas"].tolist() == [2, 2]
    assert tabla["columnas"].tolist() == [2, 2]
    assert tabla.loc[0, "p0000":"p0003"].tolist() == [1, -1, -1, 1]
    assert tabla.loc[1, "p0000":"p0003"].tolist() == [1, -1, -1, 1]


def test_construir_tabla_vacia(modulo: ModuleType) -> None:
    """No se puede construir un dataset sin imágenes."""
    with pytest.raises(ValueError, match="al menos"):
        modulo.construir_tabla([])


def test_stems_duplicados(modulo: ModuleType, tmp_path: Path) -> None:
    """Dos imágenes con el mismo nombre no pueden convivir en un dataset."""
    carpeta_a = tmp_path / "uno"
    carpeta_b = tmp_path / "dos"
    carpeta_a.mkdir()
    carpeta_b.mkdir()
    ruta_a = _png_base(carpeta_a, "misma.png")
    ruta_b = _png_base(carpeta_b, "misma.png")

    with pytest.raises(ValueError, match="duplicad"):
        modulo.construir_tabla([ruta_a, ruta_b])


def test_tamanos_distintos(modulo: ModuleType, tmp_path: Path) -> None:
    """Todas las imágenes de un dataset deben tener el mismo tamaño."""
    ruta_a = _png_base(tmp_path, "chica.png")
    ruta_b = tmp_path / "grande.png"
    _guardar_png(ruta_b, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    with pytest.raises(ValueError, match="mismo tama"):
        modulo.construir_tabla([ruta_a, ruta_b])


def test_main_stdout_un_png(
    modulo: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Con un PNG y sin --salida, el CSV se imprime por stdout."""
    ruta = tmp_path / "patron.png"
    _guardar_png(ruta, [[0, 127], [128, 255]])

    codigo = modulo.main([str(ruta)])
    salida = capsys.readouterr().out

    assert codigo == 0
    assert "nombre,filas,columnas,p0000,p0001,p0002,p0003" in salida
    assert "patron,2,2,1,1,-1,-1" in salida


def test_main_escribe_archivo(
    modulo: ModuleType,
    tmp_path: Path,
) -> None:
    """Con --salida, el CSV se guarda y se crean las carpetas necesarias."""
    ruta_a = _png_base(tmp_path, "a.png")
    ruta_b = _png_base(tmp_path, "b.png")
    salida = tmp_path / "sub" / "dataset.csv"

    codigo = modulo.main([str(ruta_a), str(ruta_b), "--salida", str(salida)])

    assert codigo == 0
    assert salida.exists()
    tabla = pd.read_csv(salida)
    assert list(tabla.columns) == [
        "nombre",
        "filas",
        "columnas",
        "p0000",
        "p0001",
        "p0002",
        "p0003",
    ]
    assert tabla["nombre"].tolist() == ["a", "b"]


def test_main_varios_sin_salida(
    modulo: ModuleType,
    tmp_path: Path,
) -> None:
    """Con varios PNG, --salida es obligatorio."""
    ruta_a = _png_base(tmp_path, "a.png")
    ruta_b = _png_base(tmp_path, "b.png")

    with pytest.raises(SystemExit):
        modulo.main([str(ruta_a), str(ruta_b)])


def test_qr_real(modulo: ModuleType) -> None:
    """El QR real debe conservar su forma de 37x37 y sus valores binarios."""
    assert RUTA_QR.exists(), f"No existe el QR: {RUTA_QR}"

    matriz = modulo.leer_matriz(RUTA_QR)

    assert matriz.shape == (37, 37)
    assert set(np.unique(matriz).tolist()) == {-1, 1}
    assert matriz[0, 0] == 1
    assert matriz[0, 7] == -1
