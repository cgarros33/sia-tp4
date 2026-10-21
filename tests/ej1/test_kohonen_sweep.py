"""Tests del barrido de hiperparámetros del ejercicio 1.2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_KOHONEN = RAIZ / "src" / "ej1" / "kohonen.py"
RUTA_SWEEP = RAIZ / "src" / "ej1" / "kohonen-sweep.py"
RUTA_CONFIG_KOHONEN = RAIZ / "src" / "ej1" / "kohonen.toml"
RUTA_CONFIG_SWEEP = RAIZ / "src" / "ej1" / "kohonen-sweep.toml"


def cargar_modulo(ruta: Path, nombre: str) -> ModuleType:
    """Carga un script como módulo para poder probar sus funciones."""
    assert ruta.exists(), f"No existe el script: {ruta}"
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


@pytest.fixture(scope="module")
def kohonen() -> ModuleType:
    """Carga el motor de la red de Kohonen."""
    return cargar_modulo(RUTA_KOHONEN, "kohonen_para_barrido")


@pytest.fixture(scope="module")
def barrido() -> ModuleType:
    """Carga el script de barrido."""
    return cargar_modulo(RUTA_SWEEP, "kohonen_sweep")


def datos_pequenos() -> np.ndarray:
    """Matriz sintética para las pruebas de orquestación."""
    return np.random.default_rng(3).normal(size=(10, 3))


def config_barrido_minimo() -> dict:
    """Configuración de barrido reducida para las pruebas."""
    return {
        "barrido": {
            "grillas_fijas": [3, 4, 5],
            "orden": ["topologia"],
            "semillas": [0, 1],
        },
        "valores": {
            "topologia": ["rectangular", "hexagonal"],
        },
        "acoplamientos": {
            "pares": [],
            "adoptar_solo_si_mejora": True,
        },
        "figuras": {"opciones": ["a", "b", "c"]},
    }


def entrenador_falso(matriz: np.ndarray, modelo: dict, semilla: int) -> dict:
    """Entrenador determinista para probar la orquestación sin entrenar."""
    del matriz
    qe = 10.0
    if modelo["grilla"] == 3:
        qe -= 1.0
    if modelo["topologia"] == "hexagonal":
        qe -= 0.5
    if modelo["eta_inicial"] == 0.2:
        qe -= 0.3
    if modelo["decaimiento_eta"] == "exponencial":
        qe -= 0.2
    return {
        "qe": qe + 0.01 * semilla,
        "te": 0.1,
        "unidades_muertas": 1,
        "ocupacion": 0.5,
    }


def entrenador_peor_acoplamiento(
    matriz: np.ndarray,
    modelo: dict,
    semilla: int,
) -> dict:
    """Entrenador donde el acoplamiento siempre empeora la QE."""
    del matriz, semilla
    qe = 10.0
    if modelo["eta_inicial"] == 0.1:
        qe += 0.5
    if modelo["eta_inicial"] == 0.2:
        qe += 1.0
    return {
        "qe": qe,
        "te": 0.1,
        "unidades_muertas": 1,
        "ocupacion": 0.5,
    }


def test_agregar_resultados_media_desvio(barrido: ModuleType) -> None:
    """La agregación debe calcular media y desvío muestral por valor."""
    resultados = pd.DataFrame(
        {
            "parametro": ["eta_inicial"] * 4,
            "valor": [0.1, 0.1, 0.2, 0.2],
            "semilla": [0, 1, 0, 1],
            "qe": [1.0, 1.2, 0.8, 0.9],
            "te": [0.1, 0.2, 0.1, 0.1],
            "unidades_muertas": [1, 1, 0, 0],
            "ocupacion": [0.9, 0.9, 1.0, 1.0],
        }
    )

    resumen = barrido.agregar_resultados(resultados)
    fila = resumen[
        (resumen["parametro"] == "eta_inicial") & (resumen["valor"] == 0.1)
    ].iloc[0]

    assert np.isclose(fila["qe_media"], 1.1)
    assert np.isclose(fila["qe_desvio"], np.std([1.0, 1.2], ddof=1))


def test_seleccionar_mejor_minimo(barrido: ModuleType) -> None:
    """La selección normal debe elegir el menor QE medio."""
    resumen = pd.DataFrame(
        {
            "valor": [0.1, 0.2, 0.3],
            "qe_media": [1.0, 0.8, 0.9],
        }
    )

    elegido = barrido.seleccionar_mejor(resumen)

    assert np.isclose(elegido, 0.2)


def test_actualizar_config_copia(kohonen: ModuleType, barrido: ModuleType) -> None:
    """Actualizar un parámetro no debe modificar el diccionario original."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)

    nueva = barrido.actualizar_config(config, "grilla", 3)

    assert nueva["modelo"]["grilla"] == 3
    assert config["modelo"]["grilla"] == 6


def test_evaluar_paso_incluye_neutro(
    kohonen: ModuleType,
    barrido: ModuleType,
) -> None:
    """Cada paso debe incluir el valor neutro vigente entre los candidatos."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)

    resultado = barrido.evaluar_paso(
        "grilla",
        [2, 3],
        config,
        datos_pequenos(),
        [0, 1],
        entrenador_falso,
    )

    assert set(resultado.resumen["valor"].astype(int)) == {2, 3, 6}
    assert resultado.mejor_valor == 3


def test_ejecutar_barrido_secuencial(
    kohonen: ModuleType,
    barrido: ModuleType,
    tmp_path: Path,
) -> None:
    """El barrido debe actualizar el neutro después de cada paso."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)
    config["modelo"]["grilla"] = 3
    salida = tmp_path / "salida"

    final, resumen = barrido.ejecutar_barrido(
        config,
        datos_pequenos(),
        config_barrido_minimo(),
        entrenador_falso,
        salida,
    )

    assert final["modelo"]["grilla"] == 3
    assert final["modelo"]["topologia"] == "hexagonal"
    assert set(resumen["parametro"]) == {"topologia"}


def test_acoplamiento_cartesiano(
    kohonen: ModuleType,
    barrido: ModuleType,
) -> None:
    """El acoplamiento debe probar el producto cartesiano de los dos pares."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)

    resultado = barrido.evaluar_acoplamiento(
        "eta_inicial",
        [0.1, 0.2],
        "decaimiento_eta",
        ["lineal", "exponencial"],
        config,
        datos_pequenos(),
        [0, 1],
        entrenador_falso,
    )

    assert resultado.mejor_valor == (0.2, "exponencial")
    assert {0.1, 0.2}.issubset(set(resultado.resumen["valor_a"]))
    assert {"lineal", "exponencial"}.issubset(set(resultado.resumen["valor_b"]))


def test_debe_adoptar(barrido: ModuleType) -> None:
    """Un acoplamiento se adopta solo si baja la QE media."""
    assert barrido.debe_adoptar(8.0, 9.0)
    assert not barrido.debe_adoptar(9.0, 9.0)
    assert not barrido.debe_adoptar(10.0, 9.0)


def test_barra_progreso(barrido: ModuleType) -> None:
    """La barra debe cubrir el ancho indicado según el porcentaje."""
    assert barrido._barra_progreso(0.0, 10) == "[----------]"
    assert barrido._barra_progreso(50.0, 10) == "[#####-----]"
    assert barrido._barra_progreso(100.0, 10) == "[##########]"


def test_radios_invalidos_se_filtran(barrido: ModuleType) -> None:
    """Con k=3, el radio 0.3 queda por debajo de R_f=1 y se descarta."""
    validos = barrido._valores_radio_validos([0.3, 0.4, 0.5], 3, 1.0)

    assert validos == [0.4, 0.5]


def test_ejecutar_barrido_no_adopta_acoplamiento_peor(
    kohonen: ModuleType,
    barrido: ModuleType,
    tmp_path: Path,
) -> None:
    """Si el acoplamiento no mejora, se conserva el valor secuencial."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)
    config_barrido = config_barrido_minimo()
    config_barrido["barrido"]["orden"] = ["topologia"]
    config_barrido["valores"] = {
        "topologia": ["rectangular", "hexagonal"],
        "eta_inicial": [0.1, 0.2],
        "decaimiento_eta": ["lineal"],
    }
    config_barrido["acoplamientos"]["pares"] = [["eta_inicial", "decaimiento_eta"]]

    final, _ = barrido.ejecutar_barrido(
        config,
        datos_pequenos(),
        config_barrido,
        entrenador_peor_acoplamiento,
        tmp_path / "salida",
    )

    assert final["modelo"]["eta_inicial"] == 0.5


def test_config_optima_roundtrip(
    kohonen: ModuleType,
    barrido: ModuleType,
    tmp_path: Path,
) -> None:
    """La configuración óptima debe poder leerse con el motor."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)
    salida = tmp_path / "salida"

    final, _ = barrido.ejecutar_barrido(
        config,
        datos_pequenos(),
        config_barrido_minimo(),
        entrenador_falso,
        salida,
    )
    recargada = kohonen.cargar_config(salida / "config_optima.toml")

    assert recargada["modelo"]["grilla"] == final["modelo"]["grilla"]
    assert recargada["modelo"]["topologia"] == final["modelo"]["topologia"]


def test_salidas_reducidas(
    kohonen: ModuleType,
    barrido: ModuleType,
    tmp_path: Path,
) -> None:
    """El barrido reducido debe generar CSV y las opciones de figuras configuradas."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)
    salida = tmp_path / "salida"

    barrido.ejecutar_barrido(
        config,
        datos_pequenos(),
        config_barrido_minimo(),
        entrenador_falso,
        salida,
    )

    assert (salida / "config_optima.toml").exists()
    assert (salida / "resumen_pasos.csv").exists()
    assert (salida / "paso_01_topologia.csv").exists()

    for opcion in ["opcion_a", "opcion_b", "opcion_c"]:
        carpeta = salida / "figuras" / opcion
        assert carpeta.is_dir()
        assert list(carpeta.glob("*.png"))


def test_opciones_por_defecto_sin_c(
    kohonen: ModuleType,
    barrido: ModuleType,
    tmp_path: Path,
) -> None:
    """Sin sección de figuras se generan A y B, pero no C."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)
    config_barrido = config_barrido_minimo()
    del config_barrido["figuras"]
    salida = tmp_path / "salida"

    barrido.ejecutar_barrido(
        config,
        datos_pequenos(),
        config_barrido,
        entrenador_falso,
        salida,
    )

    assert (salida / "figuras" / "opcion_a").is_dir()
    assert (salida / "figuras" / "opcion_b").is_dir()
    assert not (salida / "figuras" / "opcion_c").exists()


def test_ejecutar_barridos_crea_tres_tamanos(
    kohonen: ModuleType,
    barrido: ModuleType,
    tmp_path: Path,
) -> None:
    """Un solo llamado debe crear sweep3, sweep4 y sweep5."""
    config = kohonen.cargar_config(RUTA_CONFIG_KOHONEN)
    config_barrido = config_barrido_minimo()
    config_barrido["salida"] = {
        "directorio": str(tmp_path / "barridos"),
        "directorio_final_base": str(tmp_path / "finales" / "kohonen"),
    }

    resultados = barrido.ejecutar_barridos(
        config,
        datos_pequenos(),
        config_barrido,
        entrenador_falso,
        tmp_path / "barridos",
    )

    assert set(resultados) == {3, 4, 5}
    for grilla in (3, 4, 5):
        carpeta = tmp_path / "barridos" / f"sweep{grilla}"
        assert (carpeta / "config_optima.toml").exists()
        recargada = kohonen.cargar_config(carpeta / "config_optima.toml")
        assert recargada["modelo"]["grilla"] == grilla
        assert (carpeta / "figuras" / "opcion_a").is_dir()
        assert (carpeta / "figuras" / "opcion_b").is_dir()


def test_config_barrido_valida(barrido: ModuleType) -> None:
    """La configuración completa debe cargar y validar."""
    config_barrido = barrido.cargar_config_barrido(RUTA_CONFIG_SWEEP)

    assert config_barrido["barrido"]["grillas_fijas"] == [3, 4, 5]
    assert config_barrido["barrido"]["orden"][0] == "topologia"
    assert "grilla" not in config_barrido["barrido"]["orden"]
    assert "grilla" not in config_barrido["valores"]
    assert config_barrido["valores"]["iteraciones"] == [
        5000,
        10000,
        15000,
        25000,
    ]
    assert config_barrido["valores"]["distancia"] == ["euclidiana", "exponencial"]
    assert config_barrido["figuras"]["opciones"] == ["a", "b"]

    with pytest.raises(ValueError):
        barrido.validar_config_barrido(
            {
                "barrido": {
                    "grillas_fijas": [3],
                    "orden": ["inexistente"],
                    "semillas": [0],
                },
                "valores": {},
            }
        )

    with pytest.raises(ValueError):
        barrido.validar_config_barrido(
            {
                "barrido": {
                    "grillas_fijas": [3],
                    "orden": ["grilla"],
                    "semillas": [0],
                },
                "valores": {"grilla": [3]},
            }
        )
