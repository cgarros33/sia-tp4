"""Smoke tests del ejercicio 2.1: figuras generadas desde los CSV guardados."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "hopfield-figures.py"
RUTA_NOISE = RAIZ / "src" / "ej2" / "hopfield-noise.py"
RUTA_MOTOR = RAIZ / "src" / "ej2" / "hopfield.py"
RUTA_RUNNER = RAIZ / "src" / "ej2" / "hopfield-runner.py"
FIGURAS_MODELO = (
    "fig_exactitud.png",
    "fig_composicion_errores.png",
    "fig_iteraciones.png",
    "fig_energia.png",
    "fig_hamming.png",
    "fig_confusion.png",
    "fig_exactitud_por_patron.png",
    "fig_boxplot_barridos.png",
    "fig_afectacion_real.png",
)


def _cargar(ruta: Path, nombre: str) -> ModuleType:
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
def modulo() -> ModuleType:
    """Carga el script de figuras."""
    return _cargar(RUTA_SCRIPT, "hopfield_figures")


@pytest.fixture(scope="module")
def raiz(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Árbol de resultados chico: dos redes, consultas limpias y un modelo."""
    motor = _cargar(RUTA_MOTOR, "hopfield_motor_test_figuras")
    runner = _cargar(RUTA_RUNNER, "hopfield_runner_test_figuras")
    noise = _cargar(RUTA_NOISE, "hopfield_noise_test_figuras")
    base = tmp_path_factory.mktemp("figuras")
    generador = np.random.default_rng(1)
    patron = generador.choice([-1, 1], size=144)
    patrones = np.tile(patron, (3, 1)).astype(np.int8)
    for indice in range(3):
        patrones[indice, indice * 12 : indice * 12 + 12] *= -1
    dataset = motor.Dataset(
        nombres=("a", "b", "c"), patrones=patrones, filas=12, columnas=12
    )
    ruta_datos = base / "datos.csv"
    motor.escribir_dataset(dataset, ruta_datos)
    for regla in ("hebb", "pseudoinversa"):
        red = base / regla
        argumentos = ["--datos", str(ruta_datos), "--regla", regla]
        assert motor.main(argumentos + ["--salida", str(red)]) == 0
        limpias = red / "consultas" / "limpias"
        assert runner.main(["--red", str(red), "--salida", str(limpias)]) == 0

    config = base / "noise.toml"
    lineas = [
        "[redes]",
        f"hebb = {json.dumps(str(base / 'hebb'))}",
        f"pseudoinversa = {json.dumps(str(base / 'pseudoinversa'))}",
        "[barrido]",
        "semillas = [0, 1]",
        "repeticiones = 2",
        'patron_ejemplo = "a"',
        f"salida = {json.dumps(str(base / 'ruido'))}",
        "[niveles]",
        "aleatorio = [0.1, 0.3, 0.45]",
        "[combinaciones]",
        'componentes = ["aleatorio", "esquina"]',
        "aleatorio = 0.1",
        "esquina = 3",
    ]
    config.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    for accion in (
        ["--modelo", "aleatorio"],
        ["--modelo", "aleatorio", "--legibilidad"],
        ["--combinaciones"],
        ["--consolidar"],
    ):
        assert noise.main(["--config", str(config), *accion]) == 0

    # Barrido exploratorio: esquinas grandes, sin escalera.
    quiebre = base / "noise_quiebre.toml"
    texto = config.read_text(encoding="utf-8").split("[niveles]")[0]
    texto = (
        texto.replace("ruido", "ruido_quiebre") + "[niveles]\nesquina = [8, 10, 12]\n"
    )
    quiebre.write_text(texto, encoding="utf-8")
    for accion in (
        ["--modelo", "esquina"],
        ["--modelo", "esquina", "--legibilidad"],
        ["--consolidar"],
    ):
        assert noise.main(["--config", str(quiebre), *accion]) == 0
    return base


def test_figuras_por_modelo(modulo: ModuleType, raiz: Path) -> None:
    """Las figuras de cada modelo se generan y se regeneran igual."""
    assert modulo.main(["--raiz", str(raiz), "--solo", "modelos"]) == 0
    figuras = raiz / "ruido" / "aleatorio" / "figuras"
    primera = {}
    for nombre in FIGURAS_MODELO:
        assert (figuras / nombre).exists(), nombre
        primera[nombre] = (figuras / nombre).read_bytes()

    assert modulo.main(["--raiz", str(raiz), "--solo", "modelos"]) == 0
    for nombre in FIGURAS_MODELO:
        assert (figuras / nombre).read_bytes() == primera[nombre], nombre


def test_nivel_de_referencia(modulo: ModuleType) -> None:
    """El nivel de referencia es el primero en que la pseudo-inversa cae de 0,95."""
    import pandas as pd

    resumen = pd.DataFrame(
        {
            "orden_nivel": [0, 1, 2, 0, 1, 2],
            "nivel": ["a", "b", "c", "a", "b", "c"],
            "regla": ["hebb"] * 3 + ["pseudoinversa"] * 3,
            "exactitud_media": [0.0, 0.0, 0.0, 1.0, 0.9, 0.2],
        }
    )
    assert modulo.nivel_referencia(resumen) == "b"
    resumen.loc[resumen["regla"] == "pseudoinversa", "exactitud_media"] = 1.0
    assert modulo.nivel_referencia(resumen) == "b"


def test_no_importa_motor_ni_runner() -> None:
    """El script de figuras solo lee resultados."""
    fuente = RUTA_SCRIPT.read_text(encoding="utf-8")
    for prohibido in ("hopfield.py", "hopfield-runner.py", "hopfield-noise.py"):
        assert prohibido not in fuente


FIGURAS_COMBINACIONES = (
    "fig_combinaciones_exactitud.png",
    "fig_combinaciones_iteraciones.png",
    "fig_combinaciones_energia.png",
    "fig_combinaciones_hamming.png",
    "fig_mapa_calor_exactitud_combinaciones.png",
)


def test_figuras_combinaciones(modulo: ModuleType, raiz: Path) -> None:
    """Cada paso tiene sus figuras y la escalera consolidada, las suyas."""
    assert modulo.main(["--raiz", str(raiz), "--solo", "combinaciones"]) == 0
    combinaciones = raiz / "ruido" / "combinaciones"
    for paso in (1, 2):
        for nombre in FIGURAS_MODELO:
            assert (combinaciones / f"paso_{paso}" / "figuras" / nombre).exists()
    figuras = combinaciones / "consolidado" / "figuras"
    primera = {
        nombre: (figuras / nombre).read_bytes() for nombre in FIGURAS_COMBINACIONES
    }
    assert modulo.main(["--raiz", str(raiz), "--solo", "combinaciones"]) == 0
    for nombre in FIGURAS_COMBINACIONES:
        assert (figuras / nombre).read_bytes() == primera[nombre], nombre


FIGURAS_ENTRENAMIENTO = (
    "fig_solapamiento_entrenamiento.png",
    "fig_estado_espurio.png",
    "fig_energia_entrenamiento.png",
    "fig_pesos_histograma.png",
    "fig_capacidad_subconjuntos.png",
)
FIGURAS_GLOBALES = (
    "fig_exactitud_global.png",
    "fig_iteraciones_global.png",
    "fig_energia_global.png",
    "fig_hamming_global.png",
    "fig_comparacion_global.png",
    "fig_comparacion_reglas.png",
    "fig_mapa_calor_exactitud.png",
    "fig_ejemplos_qr.png",
    "fig_basin_hamming.png",
    "fig_trayectorias_energia_ejemplo.png",
    "fig_afectacion_real_global.png",
    "fig_lectura_ejemplos.png",
)


def _regenerar_igual(modulo: ModuleType, raiz: Path, solo: str, carpeta: Path, nombres):
    assert modulo.main(["--raiz", str(raiz), "--solo", solo]) == 0
    primera = {}
    for nombre in nombres:
        assert (carpeta / nombre).exists(), nombre
        primera[nombre] = (carpeta / nombre).read_bytes()
    assert modulo.main(["--raiz", str(raiz), "--solo", solo]) == 0
    for nombre in nombres:
        assert (carpeta / nombre).read_bytes() == primera[nombre], nombre


def test_figuras_entrenamiento(modulo: ModuleType, raiz: Path) -> None:
    """Las figuras y la tabla de la comparación de entrenamiento."""
    import pandas as pd

    carpeta = raiz / "comparacion_entrenamiento"
    _regenerar_igual(modulo, raiz, "entrenamiento", carpeta, FIGURAS_ENTRENAMIENTO)
    tabla = pd.read_csv(carpeta / "comparacion_entrenamiento.csv")
    assert set(tabla["regla"]) == {"hebb", "pseudoinversa"}
    assert tabla["estado"].tolist().count("mezcla") == 2
    assert {"rango", "numero_condicion", "punto_fijo"} <= set(tabla.columns)


def test_figuras_globales(modulo: ModuleType, raiz: Path) -> None:
    """Las figuras globales se generan desde el consolidado."""
    carpeta = raiz / "ruido" / "consolidado" / "figuras"
    _regenerar_igual(modulo, raiz, "global", carpeta, FIGURAS_GLOBALES)


FIGURAS_QUIEBRE = (
    "fig_quiebre_exactitud_vs_afectada.png",
    "fig_quiebre_zonas.png",
    "fig_legibilidad.png",
)


def test_figuras_quiebre(modulo: ModuleType, raiz: Path) -> None:
    """El barrido exploratorio tiene sus figuras por modelo, las comparativas y tablas."""
    import pandas as pd

    quiebre = raiz / "ruido_quiebre"
    consolidado = quiebre / "consolidado"
    _regenerar_igual(modulo, raiz, "quiebre", consolidado / "figuras", FIGURAS_QUIEBRE)
    for nombre in FIGURAS_MODELO:
        assert (quiebre / "esquina" / "figuras" / nombre).exists(), nombre

    por_esquina = pd.read_csv(
        consolidado / "exactitud_por_esquina.csv", dtype={"nivel": str}
    )
    assert set(por_esquina["nivel"]) == {"8", "10", "12"}
    assert por_esquina["exactitud"].between(0, 1).all()
    por_margen = pd.read_csv(consolidado / "exactitud_por_margen.csv")
    assert {"regla", "margen_positivo", "consultas", "exactitud"} <= set(
        por_margen.columns
    )
    consultas = pd.read_csv(consolidado / "consultas.csv")
    assert por_margen["consultas"].sum() == len(consultas)
