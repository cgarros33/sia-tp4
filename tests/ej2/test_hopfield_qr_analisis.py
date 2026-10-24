"""Tests del análisis de recuperación de QR con Hopfield (ejercicio 2.1)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "hopfield-qr-analisis.py"
RUTA_MOTOR = RAIZ / "src" / "ej2" / "hopfield.py"
RUTA_RUNNER = RAIZ / "src" / "ej2" / "hopfield-runner.py"


def _cargar(ruta: Path, nombre: str) -> ModuleType:
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script de análisis como módulo."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    return _cargar(RUTA_SCRIPT, "hopfield_qr_analisis")


def test_tasas_por_tipo(modulo: ModuleType) -> None:
    """Cuenta la fracción de cada tipo de resultado."""
    consultas = pd.DataFrame(
        {
            "tipo_resultado": [
                "almacenado_correcto",
                "almacenado_correcto",
                "espurio",
                "almacenado_incorrecto",
            ],
            "barridos": [1, 2, 3, 4],
            "hamming_inicial": [0, 2, 4, 6],
            "distancia_hamming": [0, 0, 5, 0],
        }
    )
    tasas = modulo.tasas(consultas)
    assert tasas["correcto"] == pytest.approx(0.5)
    assert tasas["espurio"] == pytest.approx(0.25)
    assert tasas["incorrecto"] == pytest.approx(0.25)
    assert tasas["otro"] == pytest.approx(0.0)
    assert tasas["barridos"] == pytest.approx(2.5)
    assert tasas["hamming_inicial"] == pytest.approx(3.0)


def test_leer_carpeta_consulta(modulo: ModuleType) -> None:
    """Interpreta el nombre de las carpetas del barrido."""
    assert modulo.leer_carpeta("ruido_0.25_semilla_2") == ("ruido", 0.25, 2)
    assert modulo.leer_carpeta("esquina_0.1_semilla_0") == ("esquina", 0.1, 0)
    assert modulo.leer_carpeta("detalle_ruido_0.2") is None
    assert modulo.leer_carpeta("limpias") is None


def test_estado_mezcla_y_pixeles_comunes(modulo: ModuleType) -> None:
    """El voto mayoritario y los píxeles iguales en todos los patrones."""
    patrones = np.array([[1, 1, -1, 1], [1, -1, -1, 1], [1, 1, -1, -1]])
    np.testing.assert_array_equal(modulo.estado_mezcla(patrones), [1, 1, -1, 1])
    np.testing.assert_array_equal(
        modulo.pixeles_comunes(patrones), [True, False, True, False]
    )


def test_integracion_smoke(modulo: ModuleType, tmp_path: Path) -> None:
    """Analiza dos redes chicas con consultas y detalle de energía."""
    motor = _cargar(RUTA_MOTOR, "hopfield_motor_analisis")
    runner = _cargar(RUTA_RUNNER, "hopfield_runner_analisis")

    generador = np.random.default_rng(0)
    base = generador.choice([-1, 1], size=36)
    patrones = np.tile(base, (3, 1)).astype(np.int8)
    for indice in range(3):
        patrones[indice, indice * 6 : indice * 6 + 6] *= -1
    dataset = motor.Dataset(
        nombres=("a", "b", "c"), patrones=patrones, filas=6, columnas=6
    )
    ruta_datos = tmp_path / "datos.csv"
    motor.escribir_dataset(dataset, ruta_datos)

    redes = []
    for regla in ("hebb", "pseudoinversa"):
        red_dir = tmp_path / f"qr_{regla}"
        argumentos = ["--datos", str(ruta_datos), "--regla", regla]
        assert motor.main(argumentos + ["--salida", str(red_dir)]) == 0
        for deterioro in ("ruido", "esquina", "manchas"):
            for nivel in ("0.1", "0.3"):
                for semilla in ("0", "1"):
                    nombre = f"{deterioro}_{nivel}_semilla_{semilla}"
                    argumentos = ["--red", str(red_dir), "--deterioro", deterioro]
                    argumentos += ["--nivel", nivel, "--semilla", semilla]
                    argumentos += ["--repeticiones", "2"]
                    argumentos += ["--salida", str(red_dir / "consultas" / nombre)]
                    assert runner.main(argumentos) == 0
            detalle = red_dir / "consultas" / f"detalle_{deterioro}_0.3"
            argumentos = ["--red", str(red_dir), "--deterioro", deterioro]
            argumentos += ["--nivel", "0.3", "--energia-detallada"]
            argumentos += ["--salida", str(detalle)]
            assert runner.main(argumentos) == 0
        redes.append(f"{regla}={red_dir}")

    salida = tmp_path / "analisis"
    argumentos = []
    for red in redes:
        argumentos += ["--red", red]
    assert modulo.main(argumentos + ["--salida", str(salida)]) == 0

    por_semilla = pd.read_csv(salida / "recuperacion_por_semilla.csv")
    resumen = pd.read_csv(salida / "recuperacion.csv")
    assert len(por_semilla) == 2 * 3 * 2 * 2
    assert len(resumen) == 2 * 3 * 2
    assert resumen["correcto_media"].between(0, 1).all()
    for figura in (
        "fig_correlacion.png",
        "fig_recuperacion.png",
        "fig_barridos.png",
        "fig_energia.png",
        "fig_ejemplos.png",
        "fig_capacidad.png",
    ):
        assert (salida / figura).exists(), figura
    assert (salida / "correlacion.csv").exists()
    capacidad = pd.read_csv(salida / "capacidad.csv")
    assert set(capacidad["regla"]) == {"hebb", "pseudoinversa"}
    assert capacidad["cantidad"].tolist() == [1, 2, 3, 1, 2, 3]
    assert (salida / "resumen.txt").exists()
