"""Tests del ejercicio 1.2: red de Kohonen."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "kohonen.py"
RUTA_CONFIG = RAIZ / "src" / "ej1" / "kohonen.toml"
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script como módulo para poder probar sus funciones."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("kohonen", RUTA_SCRIPT)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


def modelo_base() -> dict:
    """Devuelve un modelo chico válido para las pruebas de entrenamiento."""
    return {
        "grilla": 3,
        "topologia": "rectangular",
        "distancia": "euclidiana",
        "inicializacion": "muestras",
        "radio_inicial_relativo": 0.5,
        "radio_final": 1.0,
        "decaimiento_radio": "lineal",
        "eta_inicial": 0.5,
        "eta_final": 0.001,
        "decaimiento_eta": "lineal",
        "iteraciones": 100,
    }


def test_cargar_datos_y_columnas(modulo: ModuleType) -> None:
    """El dataset conserva las 50 filas y usa 27 variables de entrada."""
    datos = modulo.cargar_datos(RUTA_DATOS)
    config = modulo.cargar_config(RUTA_CONFIG)
    columnas = modulo.columnas_entrada(datos, config["datos"]["columnas_excluidas"])

    assert datos.shape == (50, 33)
    assert len(columnas) == 27
    assert set(columnas).issubset(datos.columns)
    for columna in config["datos"]["columnas_excluidas"]:
        assert columna not in columnas


def test_estandarizacion(modulo: ModuleType) -> None:
    """Cada columna estandarizada debe tener media 0 y desvío muestral 1."""
    matriz = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 10.0]])

    estandarizados, medias, desvios = modulo.estandarizar(matriz)

    assert np.allclose(estandarizados.mean(axis=0), 0.0)
    assert np.allclose(estandarizados.std(axis=0, ddof=1), 1.0)
    assert np.allclose(medias, matriz.mean(axis=0))
    assert np.allclose(desvios, matriz.std(axis=0, ddof=1))


def test_distancias_grilla_rectangular(modulo: ModuleType) -> None:
    """R=1 debe dar 4 vecinos y R=sqrt(2) debe dar los 8 vecinos."""
    distancias = modulo.distancias_grilla(3, "rectangular")
    centro = 4

    assert distancias.shape == (9, 9)
    assert np.allclose(distancias, distancias.T)
    assert np.allclose(np.diag(distancias), 0.0)
    assert np.isclose(distancias[0, 4], np.sqrt(2.0))
    assert np.count_nonzero(distancias[centro] <= 1.0) - 1 == 4
    assert np.count_nonzero(distancias[centro] <= np.sqrt(2.0)) - 1 == 8


def test_distancias_grilla_hexagonal(modulo: ModuleType) -> None:
    """En hexagonal, R=1 debe dar los 6 vecinos inmediatos."""
    distancias = modulo.distancias_grilla(3, "hexagonal")
    centro = 4

    assert distancias.shape == (9, 9)
    assert np.allclose(distancias, distancias.T)
    assert np.allclose(np.diag(distancias), 0.0)
    assert np.count_nonzero(distancias[centro] <= 1.0) - 1 == 6


def test_bmu_euclidiana_y_exponencial(modulo: ModuleType) -> None:
    """La BMU euclidiana y la exponencial normalizada pueden diferir."""
    pesos = np.array([[2.0, 0.0], [0.6, 0.6]])
    muestra = np.array([1.0, 0.0])

    assert modulo.seleccionar_bmu(pesos, muestra, "euclidiana") == 1
    assert modulo.seleccionar_bmu(pesos, muestra, "exponencial") == 0


def test_seleccionar_bmus_vectorizado(modulo: ModuleType) -> None:
    """La selección de BMUs debe devolver un índice por muestra."""
    pesos = np.array([[2.0, 0.0], [0.6, 0.6]])
    muestras = np.array([[1.0, 0.0], [0.0, 1.0]])

    bmus = modulo.seleccionar_bmus(pesos, muestras, "euclidiana")

    assert bmus.shape == (2,)
    assert np.array_equal(bmus, np.array([1, 1]))


def test_actualizacion_pesos_acerca_al_patron(modulo: ModuleType) -> None:
    """La regla de Kohonen debe acercar la BMU al patrón."""
    pesos = np.array([[0.0, 0.0], [10.0, 10.0]])
    muestra = np.array([2.0, 2.0])
    mascara = np.array([True, False])

    nuevos = modulo.actualizar_pesos(pesos, muestra, 0, mascara, 0.5)

    assert np.allclose(nuevos[0], np.array([1.0, 1.0]))
    assert np.allclose(nuevos[1], pesos[1])


@pytest.mark.parametrize("esquema", ["lineal", "exponencial", "inversa"])
def test_calcular_radios(modulo: ModuleType, esquema: str) -> None:
    """El radio debe ir de R0 a R_f y no aumentar."""
    radios = modulo.calcular_radios(5, 3.0, 1.0, esquema)

    assert len(radios) == 5
    assert np.isclose(radios[0], 3.0)
    assert np.isclose(radios[-1], 1.0)
    assert np.all(np.diff(radios) <= 1e-12)


def test_calcular_radios_constante(modulo: ModuleType) -> None:
    """El esquema constante mantiene R0 durante toda la corrida."""
    radios = modulo.calcular_radios(5, 3.0, 1.0, "constante")

    assert np.allclose(radios, 3.0)


def test_calcular_radios_t_uno(modulo: ModuleType) -> None:
    """Con una sola iteración se usa el valor inicial."""
    radios = modulo.calcular_radios(1, 3.0, 1.0, "lineal")

    assert np.allclose(radios, np.array([3.0]))


def test_calcular_radios_inversa_sin_cambio(modulo: ModuleType) -> None:
    """Si R0 es igual a R_f, el esquema inverso no debe dividir por cero."""
    radios = modulo.calcular_radios(5, 1.0, 1.0, "inversa")

    assert np.allclose(radios, 1.0)


@pytest.mark.parametrize("esquema", ["lineal", "exponencial", "inversa"])
def test_calcular_tasas(modulo: ModuleType, esquema: str) -> None:
    """La tasa debe ir de eta0 a eta_f y no aumentar."""
    tasas = modulo.calcular_tasas(5, 0.5, 0.001, esquema)

    assert len(tasas) == 5
    assert np.isclose(tasas[0], 0.5)
    assert np.isclose(tasas[-1], 0.001)
    assert np.all(np.diff(tasas) <= 1e-12)


def test_calcular_tasas_t_uno(modulo: ModuleType) -> None:
    """Con una sola iteración se usa la tasa inicial."""
    tasas = modulo.calcular_tasas(1, 0.5, 0.001, "lineal")

    assert np.allclose(tasas, np.array([0.5]))


@pytest.mark.parametrize("topologia", ["rectangular", "hexagonal"])
def test_matriz_u(modulo: ModuleType, topologia: str) -> None:
    """La matriz U debe promediar las distancias a los vecinos inmediatos."""
    pesos = np.arange(9, dtype=float).reshape(9, 1)
    distancias = modulo.distancias_grilla(3, topologia)
    vecinos = np.flatnonzero(distancias[4] == 1.0)
    esperado = np.mean(np.abs(pesos[vecinos, 0] - pesos[4, 0]))

    matriz = modulo.matriz_u(pesos, 3, topologia)

    assert matriz.shape == (3, 3)
    assert np.isclose(matriz[1, 1], esperado)


def test_calcular_qe(modulo: ModuleType) -> None:
    """La QE es el promedio de las distancias mínimas."""
    distancias = np.array([[1.0, 2.0], [3.0, 0.5]])

    assert np.isclose(modulo.calcular_qe(distancias), 0.75)


def test_calcular_te(modulo: ModuleType) -> None:
    """El TE cuenta las muestras cuya segunda BMU no es vecina."""
    grilla = modulo.distancias_grilla(2, "rectangular")

    no_vecinas = np.array([[1.0, 2.0, 2.0, 1.5]])
    vecinas = np.array([[1.0, 1.5, 2.0, 2.0]])

    assert np.isclose(modulo.calcular_te(no_vecinas, grilla), 1.0)
    assert np.isclose(modulo.calcular_te(vecinas, grilla), 0.0)


def test_unidades_muertas_y_ocupacion(modulo: ModuleType) -> None:
    """Las unidades muertas son las que nunca ganan."""
    bmus = np.array([0, 0, 2])

    assert modulo.contar_unidades_muertas(bmus, 4) == 2
    assert np.isclose(modulo.calcular_ocupacion(bmus, 4), 0.5)


def test_entrenar_som_reproducible(modulo: ModuleType) -> None:
    """La misma semilla debe producir los mismos pesos y métricas."""
    matriz = np.random.default_rng(0).normal(size=(12, 3))
    modelo = modelo_base()

    primero = modulo.entrenar_som(matriz, modelo, 7)
    segundo = modulo.entrenar_som(matriz, modelo, 7)

    assert np.allclose(primero.pesos, segundo.pesos)
    assert np.isclose(primero.qe, segundo.qe)
    assert primero.bmu.shape == (12,)
    assert primero.unidades_muertas >= 0
    assert 0.0 <= primero.ocupacion <= 1.0


def test_config_cli_precedencia(modulo: ModuleType) -> None:
    """La CLI debe pisar los valores del archivo de configuración."""
    config = modulo.cargar_config(RUTA_CONFIG)
    args = modulo.parse_args(["--grilla", "7", "--semillas", "1,2"])

    resuelta = modulo.resolver_config(config, args)

    assert resuelta["modelo"]["grilla"] == 7
    assert resuelta["reproducibilidad"]["semillas"] == [1, 2]


def test_cargar_config_completa_defaults(modulo: ModuleType, tmp_path: Path) -> None:
    """Las claves ausentes deben completarse con los valores por defecto."""
    ruta = tmp_path / "kohonen.toml"
    ruta.write_text("[modelo]\ngrilla = 4\n", encoding="utf-8")

    config = modulo.cargar_config(ruta)

    assert config["modelo"]["grilla"] == 4
    assert config["modelo"]["topologia"] == "rectangular"


def test_validar_config_rechaza_grilla_invalida(modulo: ModuleType) -> None:
    """Una grilla de 1 neurona no permite calcular la matriz U."""
    config = modulo.cargar_config(RUTA_CONFIG)
    config["modelo"]["grilla"] = 1

    with pytest.raises(ValueError):
        modulo.validar_config(config)


def test_validar_config_rechaza_eta_inicial(modulo: ModuleType) -> None:
    """La tasa inicial debe estar en (0, 1)."""
    config = modulo.cargar_config(RUTA_CONFIG)
    config["modelo"]["eta_inicial"] = 1.5

    with pytest.raises(ValueError):
        modulo.validar_config(config)


def test_validar_config_rechaza_radio_inicial_menor(modulo: ModuleType) -> None:
    """El radio inicial debe ser al menos el radio final."""
    config = modulo.cargar_config(RUTA_CONFIG)
    config["modelo"]["grilla"] = 3
    config["modelo"]["radio_inicial_relativo"] = 0.1

    with pytest.raises(ValueError):
        modulo.validar_config(config)


def test_validar_config_rechaza_config_vacia(modulo: ModuleType) -> None:
    """Una configuración sin secciones obligatorias debe fallar."""
    with pytest.raises(ValueError):
        modulo.validar_config({})


def test_integracion_smoke(modulo: ModuleType, tmp_path: Path) -> None:
    """Una corrida chica debe generar todas las tablas y figuras."""
    salida = tmp_path / "kohonen"

    codigo = modulo.main(
        [
            "--config",
            str(RUTA_CONFIG),
            "--salida",
            str(salida),
            "--grilla",
            "3",
            "--iteraciones",
            "200",
            "--semillas",
            "0",
        ]
    )

    assert codigo == 0
    assert (salida / "config_usada.toml").exists()
    assert (salida / "resumen.txt").exists()
    assert (salida / "fig_neuronas.png").exists()
    assert (salida / "fig_distribucion.png").exists()
    assert (salida / "fig_matriz_u.png").exists()

    asignaciones = pd.read_csv(salida / "asignaciones.csv")
    pesos = pd.read_csv(salida / "pesos.csv")
    neuronas = pd.read_csv(salida / "neuronas.csv")
    normalizacion = pd.read_csv(salida / "normalizacion.csv")
    metricas = pd.read_csv(salida / "metricas.csv")
    matriz_u = pd.read_csv(salida / "matriz_u.csv", index_col=0)

    assert len(asignaciones) == 50
    assert len(pesos) == 9
    assert len(neuronas) == 9
    assert len(normalizacion) == 27
    assert len(metricas) == 1
    assert matriz_u.shape == (3, 3)
