"""Tests del ejercicio 2.1: generadores de ruido estructural."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "ruido.py"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga los generadores de ruido."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("ruido", RUTA_SCRIPT)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


def _patron(semilla: int = 0, lado: int = 37) -> np.ndarray:
    return (
        np.random.default_rng(semilla).choice([-1, 1], size=lado * lado).astype(np.int8)
    )


def test_borrar_no_modifica_el_original(modulo: ModuleType) -> None:
    """Borrar pone en -1 la máscara y devuelve una copia."""
    patron = np.array([1, -1, 1, 1], dtype=np.int8)
    copia = patron.copy()
    mascara = np.array([[True, True], [False, False]])
    resultado = modulo.borrar(patron, mascara)
    assert resultado.tolist() == [-1, -1, 1, 1]
    assert np.array_equal(patron, copia)


NIVELES_PRUEBA = {
    "esquina": 9,
    "finder": 2,
    "rayones": 3,
    "manchas": [2, 5],
    "bloques": 0.2,
}


@pytest.mark.parametrize("modelo", list(NIVELES_PRUEBA))
def test_generador_comun(modulo: ModuleType, modelo: str) -> None:
    """Cada generador devuelve una máscara de la forma de la imagen y es reproducible."""
    generador = modulo.GENERADORES[modelo]
    nivel = NIVELES_PRUEBA[modelo]
    mascara, parametros = generador(nivel, 37, 37, np.random.default_rng(3))
    otra, otros = generador(nivel, 37, 37, np.random.default_rng(3))

    assert mascara.shape == (37, 37)
    assert mascara.dtype == bool
    assert mascara.any()
    assert np.array_equal(mascara, otra)
    assert parametros == otros
    patron = _patron()
    consulta = modulo.borrar(patron, mascara)
    assert np.isin(consulta, [-1, 1]).all()
    # Solo cambian los píxeles negros de la zona borrada.
    cambiados = consulta != patron
    assert np.array_equal(cambiados, mascara.reshape(-1) & (patron == 1))


def test_esquina(modulo: ModuleType) -> None:
    """Borra un cuadrado de L × L pegado a la esquina elegida y la registra."""
    esquinas = {
        "superior_izquierda": (0, 0),
        "superior_derecha": (0, 28),
        "inferior_izquierda": (28, 0),
        "inferior_derecha": (28, 28),
    }
    vistas = set()
    for semilla in range(40):
        mascara, parametros = modulo.mascara_esquina(
            9, 37, 37, np.random.default_rng(semilla)
        )
        assert mascara.sum() == 81
        fila, columna = esquinas[parametros["esquina"]]
        assert mascara[fila : fila + 9, columna : columna + 9].all()
        assert parametros["lado"] == 9
        vistas.add(parametros["esquina"])
    assert vistas == set(esquinas)


def test_esquina_lado_invalido(modulo: ModuleType) -> None:
    """El lado debe estar entre 1 y el tamaño de la imagen."""
    for lado in (0, 38):
        with pytest.raises(ValueError):
            modulo.mascara_esquina(lado, 37, 37, np.random.default_rng(0))


def test_finder(modulo: ModuleType) -> None:
    """Borra 1, 2 o 3 finders de 7 × 7 distintos y los registra."""
    origenes = {
        "superior_izquierda": (0, 0),
        "superior_derecha": (0, 30),
        "inferior_izquierda": (30, 0),
    }
    vistos = set()
    for cantidad in (1, 2, 3):
        for semilla in range(10):
            mascara, parametros = modulo.mascara_finder(
                cantidad, 37, 37, np.random.default_rng(semilla)
            )
            assert parametros["cantidad"] == cantidad
            assert len(set(parametros["finders"])) == cantidad
            assert mascara.sum() == 49 * cantidad
            for finder in parametros["finders"]:
                fila, columna = origenes[finder]
                assert mascara[fila : fila + 7, columna : columna + 7].all()
            vistos.update(parametros["finders"])
    assert vistos == set(origenes)


def test_finder_en_los_qr_reales(modulo: ModuleType) -> None:
    """En los cinco QR, cada finder borrado cambia exactamente 33 píxeles negros."""
    import pandas as pd

    datos = pd.read_csv(RAIZ / "datos" / "hopfield" / "qr_37x37.csv")
    patrones = datos[[c for c in datos.columns if c.startswith("p")]].to_numpy()
    for cantidad in (1, 2, 3):
        mascara, _ = modulo.mascara_finder(cantidad, 37, 37, np.random.default_rng(0))
        for patron in patrones:
            cambiados = modulo.borrar(patron, mascara) != patron
            assert cambiados.sum() == 33 * cantidad


def test_finder_cantidad_invalida(modulo: ModuleType) -> None:
    """La cantidad de finders va de 1 a 3."""
    for cantidad in (0, 4):
        with pytest.raises(ValueError):
            modulo.mascara_finder(cantidad, 37, 37, np.random.default_rng(0))


def _linea(orientacion: str, fila: int, columna: int, lado: int = 37) -> np.ndarray:
    """Línea esperada de borde a borde que pasa por (fila, columna)."""
    i, j = np.indices((lado, lado))
    if orientacion == "horizontal":
        return i == fila
    if orientacion == "vertical":
        return j == columna
    if orientacion == "diagonal_descendente":
        return i - j == fila - columna
    return i + j == fila + columna


def test_rayones_una_linea(modulo: ModuleType) -> None:
    """Cada línea pasa por el píxel registrado y va de borde a borde."""
    vistas = set()
    for semilla in range(40):
        mascara, parametros = modulo.mascara_rayones(
            1, 37, 37, np.random.default_rng(semilla)
        )
        (linea,) = parametros["lineas"]
        esperada = _linea(linea["orientacion"], linea["fila"], linea["columna"])
        assert np.array_equal(mascara, esperada)
        assert mascara[linea["fila"], linea["columna"]]
        vistas.add(linea["orientacion"])
    assert vistas == set(modulo.ORIENTACIONES)


def test_rayones_varias_lineas(modulo: ModuleType) -> None:
    """Varias líneas se unen y pueden cruzarse."""
    mascara, parametros = modulo.mascara_rayones(8, 37, 37, np.random.default_rng(5))
    assert parametros["cantidad"] == 8
    union = np.zeros((37, 37), dtype=bool)
    for linea in parametros["lineas"]:
        union |= _linea(linea["orientacion"], linea["fila"], linea["columna"])
    assert np.array_equal(mascara, union)
    assert mascara.sum() <= 8 * 37


def test_rayones_cantidad_invalida(modulo: ModuleType) -> None:
    """Hace falta al menos una línea."""
    with pytest.raises(ValueError):
        modulo.mascara_rayones(0, 37, 37, np.random.default_rng(0))


def _disco(fila: int, columna: int, radio: int, lado: int = 37) -> np.ndarray:
    i, j = np.indices((lado, lado))
    return (i - fila) ** 2 + (j - columna) ** 2 <= radio**2


@pytest.mark.parametrize(("radio", "area"), [(3, 29), (5, 81), (7, 149), (10, 317)])
def test_manchas_un_disco(modulo: ModuleType, radio: int, area: int) -> None:
    """El disco borrado es el del centro registrado, recortado en el borde."""
    enteros = 0
    for semilla in range(60):
        mascara, parametros = modulo.mascara_manchas(
            [1, radio], 37, 37, np.random.default_rng(semilla)
        )
        assert parametros["radio"] == radio
        ((fila, columna),) = parametros["centros"]
        assert np.array_equal(mascara, _disco(fila, columna, radio))
        if radio <= fila <= 36 - radio and radio <= columna <= 36 - radio:
            assert mascara.sum() == area
            enteros += 1
    assert enteros > 0


def test_manchas_varias(modulo: ModuleType) -> None:
    """Varias manchas son independientes y se unen."""
    mascara, parametros = modulo.mascara_manchas(
        [3, 5], 37, 37, np.random.default_rng(2)
    )
    assert parametros["cantidad"] == 3
    assert len(parametros["centros"]) == 3
    union = np.zeros((37, 37), dtype=bool)
    for fila, columna in parametros["centros"]:
        union |= _disco(fila, columna, 5)
    assert np.array_equal(mascara, union)


def test_manchas_invalidas(modulo: ModuleType) -> None:
    """Hace falta al menos una mancha y un radio no negativo."""
    for nivel in ([0, 5], [1, -1]):
        with pytest.raises(ValueError):
            modulo.mascara_manchas(nivel, 37, 37, np.random.default_rng(0))


@pytest.mark.parametrize(
    ("fraccion", "cantidad"), [(0.05, 2), (0.10, 4), (0.20, 7), (0.30, 11), (0.40, 14)]
)
def test_bloques(modulo: ModuleType, fraccion: float, cantidad: int) -> None:
    """Borra round(f · 36) bloques completos de 6 × 6, distintos y registrados."""
    for semilla in range(5):
        mascara, parametros = modulo.mascara_bloques(
            fraccion, 37, 37, np.random.default_rng(semilla)
        )
        bloques = parametros["bloques"]
        assert parametros["cantidad"] == cantidad == len(set(bloques))
        assert all(0 <= bloque < 36 for bloque in bloques)
        esperada = np.zeros((37, 37), dtype=bool)
        for bloque in bloques:
            fila, columna = divmod(bloque, 6)
            esperada[6 * fila : 6 * fila + 6, 6 * columna : 6 * columna + 6] = True
        assert np.array_equal(mascara, esperada)
        assert mascara.sum() == 36 * cantidad
        assert not mascara[36, :].any()
        assert not mascara[:, 36].any()


def test_bloques_fraccion_invalida(modulo: ModuleType) -> None:
    """La fracción debe dar al menos un bloque y no superar 1."""
    for fraccion in (0.0, 0.01, 1.5):
        with pytest.raises(ValueError):
            modulo.mascara_bloques(fraccion, 37, 37, np.random.default_rng(0))
