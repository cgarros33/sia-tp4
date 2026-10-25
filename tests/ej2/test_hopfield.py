"""Tests del ejercicio 2.2: motor de la red de Hopfield."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "hopfield.py"
RUTA_CONFIG = RAIZ / "src" / "ej2" / "hopfield.toml"
RUTA_LETRAS = RAIZ / "datos" / "hopfield" / "letras_5x5.csv"
RUTA_ABECEDARIO = RAIZ / "datos" / "hopfield" / "abecedario_5x5.csv"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script como módulo para probar sus funciones."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("hopfield", RUTA_SCRIPT)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


def _guardar_dataset(
    ruta: Path,
    nombres: list[str],
    matrices: list[list[list[int]]],
) -> None:
    """Guarda un dataset de prueba en el formato CSV del proyecto."""
    arreglos = [np.asarray(matriz, dtype=int) for matriz in matrices]
    filas, columnas = arreglos[0].shape
    cantidad_pixeles = filas * columnas
    columnas_pixeles = [f"p{i:04d}" for i in range(cantidad_pixeles)]

    registros: list[dict[str, object]] = []
    for nombre, matriz in zip(nombres, arreglos, strict=True):
        assert matriz.shape == (filas, columnas)
        registro: dict[str, object] = {
            "nombre": nombre,
            "filas": filas,
            "columnas": columnas,
        }
        registro.update(
            {
                columna: int(valor)
                for columna, valor in zip(
                    columnas_pixeles, matriz.reshape(-1), strict=True
                )
            }
        )
        registros.append(registro)

    pd.DataFrame(registros).to_csv(ruta, index=False)


def test_cargar_dataset_real(modulo: ModuleType) -> None:
    """El dataset de 2.2 debe tener las cuatro letras y la forma esperada."""
    dataset = modulo.cargar_dataset(RUTA_LETRAS)

    assert dataset.nombres == ("J", "O", "T", "A")
    assert dataset.patrones.shape == (4, 25)
    assert dataset.filas == 5
    assert dataset.columnas == 5
    assert set(np.unique(dataset.patrones).tolist()) == {-1, 1}


def test_cargar_abecedario(modulo: ModuleType) -> None:
    """El abecedario debe contener las 26 letras y valores binarios."""
    dataset = modulo.cargar_dataset(RUTA_ABECEDARIO)

    assert dataset.nombres == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert dataset.patrones.shape == (26, 25)
    assert len(set(dataset.nombres)) == 26
    assert set(np.unique(dataset.patrones).tolist()) == {-1, 1}


def test_dataset_valores_invalidos(modulo: ModuleType, tmp_path: Path) -> None:
    """Los patrones solo pueden contener 1 y -1."""
    ruta = tmp_path / "invalido.csv"
    tabla = pd.DataFrame(
        {
            "nombre": ["A"],
            "filas": [2],
            "columnas": [2],
            "p0000": [1],
            "p0001": [0],
            "p0002": [-1],
            "p0003": [1],
        }
    )
    tabla.to_csv(ruta, index=False)

    with pytest.raises(ValueError, match="1 y -1"):
        modulo.cargar_dataset(ruta)


def test_dataset_nombres_duplicados(modulo: ModuleType, tmp_path: Path) -> None:
    """Los nombres de los patrones deben ser únicos."""
    ruta = tmp_path / "duplicado.csv"
    _guardar_dataset(
        ruta,
        ["A", "A"],
        [
            [[1, 1], [1, -1]],
            [[-1, -1], [-1, 1]],
        ],
    )

    with pytest.raises(ValueError, match="únic"):
        modulo.cargar_dataset(ruta)


def test_dataset_forma_inconsistente(modulo: ModuleType, tmp_path: Path) -> None:
    """Todas las filas deben declarar la misma forma."""
    ruta = tmp_path / "forma.csv"
    tabla = pd.DataFrame(
        {
            "nombre": ["A", "B"],
            "filas": [2, 3],
            "columnas": [2, 3],
            "p0000": [1, 1],
            "p0001": [-1, 1],
            "p0002": [1, -1],
            "p0003": [-1, -1],
        }
    )
    tabla.to_csv(ruta, index=False)

    with pytest.raises(ValueError, match="misma forma"):
        modulo.cargar_dataset(ruta)


def test_dataset_columnas_insuficientes(
    modulo: ModuleType,
    tmp_path: Path,
) -> None:
    """La cantidad de columnas de píxeles debe coincidir con filas x columnas."""
    ruta = tmp_path / "pixeles.csv"
    tabla = pd.DataFrame(
        {
            "nombre": ["A"],
            "filas": [2],
            "columnas": [2],
            "p0000": [1],
            "p0001": [-1],
            "p0002": [1],
        }
    )
    tabla.to_csv(ruta, index=False)

    with pytest.raises(ValueError, match="píxeles"):
        modulo.cargar_dataset(ruta)


def test_pesos_formula_manual(modulo: ModuleType) -> None:
    """La regla de Hebb debe coincidir con un cálculo hecho a mano."""
    patrones = np.array([[1, -1, 1]], dtype=int)
    pesos = modulo.calcular_pesos(patrones)

    esperado = np.array(
        [
            [0.0, -1.0 / 3.0, 1.0 / 3.0],
            [-1.0 / 3.0, 0.0, -1.0 / 3.0],
            [1.0 / 3.0, -1.0 / 3.0, 0.0],
        ]
    )

    assert np.allclose(pesos, esperado)


def test_pesos_simetricos_y_diagonal(modulo: ModuleType) -> None:
    """La matriz de pesos debe ser simétrica y tener diagonal nula."""
    dataset = modulo.cargar_dataset(RUTA_LETRAS)

    pesos = modulo.calcular_pesos(dataset.patrones)

    assert pesos.shape == (25, 25)
    assert np.allclose(pesos, pesos.T)
    assert np.allclose(np.diag(pesos), 0.0)


def test_patrones_ortogonales_puntos_fijos(modulo: ModuleType) -> None:
    """Los patrones ortogonales deben quedar como puntos fijos."""
    patrones = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
        ],
        dtype=int,
    )
    pesos = modulo.calcular_pesos(patrones)

    for patron in patrones:
        resultado = modulo.recuperar(
            pesos,
            patron,
            modo="asincronica",
            orden="secuencial",
            max_barridos=10,
            semilla=0,
        )
        margenes = modulo.calcular_margenes_estabilidad(pesos, patron)

        assert resultado.convergio
        assert np.array_equal(resultado.estado_final, patron)
        assert resultado.barridos == 1
        assert np.all(margenes >= 0)


def test_recuperacion_un_bit(modulo: ModuleType) -> None:
    """Un patrón con un bit invertido debe volver al patrón almacenado."""
    patron = np.ones(5, dtype=int)
    pesos = modulo.calcular_pesos(patron[None, :])
    ruidoso = patron.copy()
    ruidoso[0] = -1

    resultado = modulo.recuperar(
        pesos,
        ruidoso,
        modo="asincronica",
        orden="secuencial",
        max_barridos=10,
        semilla=0,
    )

    assert resultado.convergio
    assert np.array_equal(resultado.estado_final, patron)


def test_energia_asincronica_no_creciente(modulo: ModuleType) -> None:
    """La energía no debe aumentar en la dinámica asincrónica."""
    patrones = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, -1, -1, 1, -1],
        ],
        dtype=int,
    )
    pesos = modulo.calcular_pesos(patrones)
    inicial = np.array([1, -1, 1, -1, 1, -1], dtype=int)

    resultado = modulo.recuperar(
        pesos,
        inicial,
        modo="asincronica",
        orden="secuencial",
        max_barridos=20,
        semilla=0,
    )
    energias = [paso.energia for paso in resultado.historial]

    assert all(
        siguiente <= anterior + 1e-12
        for anterior, siguiente in zip(energias[:-1], energias[1:], strict=True)
    )


def test_empate_conserva_estado(modulo: ModuleType) -> None:
    """Un campo local nulo debe conservar el estado anterior."""
    assert modulo.actualizar_neurona(0.0, 1) == 1
    assert modulo.actualizar_neurona(0.0, -1) == -1
    assert modulo.actualizar_neurona(0.5, -1) == 1
    assert modulo.actualizar_neurona(-0.5, 1) == -1

    pesos = np.zeros((3, 3))
    estado = np.array([1, -1, 1], dtype=int)
    resultado = modulo.recuperar(
        pesos,
        estado,
        modo="asincronica",
        orden="secuencial",
        max_barridos=5,
        semilla=0,
    )

    assert resultado.convergio
    assert np.array_equal(resultado.estado_final, estado)
    assert resultado.barridos == 1
    assert all(paso.energia == 0.0 for paso in resultado.historial)


def _pesos_con_empate() -> tuple[np.ndarray, np.ndarray]:
    """Red de 4 neuronas donde h_0 = 0,1 + 0,2 - 0,3 = 0 en aritmética exacta.

    Las neuronas 1, 2 y 3 están ancladas por acoplamientos fuertes, así que el
    estado es punto fijo solo si el empate de la neurona 0 conserva su valor.
    """
    pesos = np.array(
        [
            [0.0, 0.1, 0.2, -0.3],
            [0.1, 0.0, 1.0, 1.0],
            [0.2, 1.0, 0.0, 1.0],
            [-0.3, 1.0, 1.0, 0.0],
        ]
    )
    return pesos, np.array([-1, 1, 1, 1], dtype=int)


def test_empate_numerico_conserva_estado(modulo: ModuleType) -> None:
    """Un campo nulo con residuo de redondeo cuenta como empate en ambos modos."""
    campo = 0.1 + 0.2 - 0.3
    assert campo != 0.0
    assert modulo.actualizar_neurona(campo, -1) == -1
    assert modulo.actualizar_neurona(-campo, 1) == 1
    assert modulo.actualizar_neurona(10 * modulo.TOLERANCIA_EMPATE, -1) == 1

    pesos, estado = _pesos_con_empate()
    for modo in modulo.MODOS:
        resultado = modulo.recuperar(pesos, estado, modo=modo, max_barridos=5)
        assert resultado.convergio, modo
        assert resultado.barridos == 1, modo
        assert np.array_equal(resultado.estado_final, estado), modo


def test_empate_cuenta_como_punto_fijo(modulo: ModuleType) -> None:
    """Un margen nulo es punto fijo en el análisis y en la capacidad."""
    pesos, estado = _pesos_con_empate()
    dataset = modulo.Dataset(
        nombres=("a",), patrones=estado[None, :].astype(np.int8), filas=2, columnas=2
    )
    analisis = modulo.analizar_red(
        dataset,
        pesos,
        modo="asincronica",
        orden="secuencial",
        max_barridos=5,
        semilla=0,
    )
    fila = analisis.estabilidad.iloc[0]
    assert fila["punto_fijo"]
    assert fila["neuronas_empate"] == 1
    assert fila["neuronas_inestables"] == 0

    # Con Hebb, la cuarta neurona de ambos patrones tiene margen exactamente 0.
    patrones = np.array([[1, 1, 1, 1], [1, 1, 1, -1]])
    tabla = modulo.capacidad_por_subconjuntos(patrones, ("a", "b"), "hebb")
    par = tabla[tabla["cantidad"] == 2].iloc[0]
    assert par["margen_minimo"] == 0.0
    assert par["todos_estables"]


def test_ciclo_sincronico(modulo: ModuleType) -> None:
    """La dinámica sincrónica debe detectar un ciclo de dos estados."""
    pesos = np.array([[0.0, 1.0], [1.0, 0.0]])
    estado = np.array([1, -1], dtype=int)

    resultado = modulo.recuperar(
        pesos,
        estado,
        modo="sincronica",
        orden="secuencial",
        max_barridos=10,
        semilla=0,
    )

    assert resultado.ciclo
    assert not resultado.convergio
    assert resultado.barridos >= 2
    assert np.array_equal(resultado.estado_final, estado)


def test_sincronico_estable(modulo: ModuleType) -> None:
    """La dinámica sincrónica debe detenerse cuando el estado no cambia."""
    pesos = np.zeros((3, 3))
    estado = np.array([1, -1, 1], dtype=int)

    resultado = modulo.recuperar(
        pesos,
        estado,
        modo="sincronica",
        orden="secuencial",
        max_barridos=5,
        semilla=0,
    )

    assert resultado.convergio
    assert not resultado.ciclo
    assert np.array_equal(resultado.estado_final, estado)
    assert resultado.barridos == 1


def test_solapamiento_y_hamming(modulo: ModuleType) -> None:
    """El solapamiento y la distancia de Hamming deben coincidir a mano."""
    patrones = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
        ],
        dtype=int,
    )

    solapamiento = modulo.calcular_solapamiento(patrones)
    hamming = modulo.calcular_hamming(patrones)

    assert np.allclose(solapamiento, [[1.0, 0.0], [0.0, 1.0]])
    assert np.array_equal(hamming, [[0, 2], [2, 0]])


def test_resumen_ortogonalidad(modulo: ModuleType) -> None:
    """El resumen debe calcular medias y máximos sobre los pares."""
    patrones = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
        ],
        dtype=int,
    )
    solapamiento = modulo.calcular_solapamiento(patrones)
    hamming = modulo.calcular_hamming(patrones)

    resumen = modulo.resumir_ortogonalidad(solapamiento, hamming)

    assert resumen["solapamiento_abs_medio"] == 0.0
    assert resumen["solapamiento_abs_max"] == 0.0
    assert resumen["hamming_min"] == 2
    assert resumen["hamming_medio"] == 2.0
    assert resumen["pares_ortogonales"] == 1


def test_capacidad(modulo: ModuleType) -> None:
    """La capacidad debe comparar p con el 15 por ciento de N."""
    capacidad = modulo.calcular_capacidad(25, 4)

    assert capacidad["neuronas"] == 25
    assert capacidad["patrones"] == 4
    assert capacidad["carga"] == pytest.approx(4 / 25)
    assert capacidad["limite_teorico"] == pytest.approx(3.75)
    assert capacidad["excede_limite"]


def test_margenes_estabilidad(modulo: ModuleType) -> None:
    """Con un solo patrón, el margen debe ser (N-1)/N en cada neurona."""
    patron = np.array([1, -1, 1], dtype=int)
    pesos = modulo.calcular_pesos(patron[None, :])

    margenes = modulo.calcular_margenes_estabilidad(pesos, patron)

    assert np.allclose(margenes, [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


def test_clasificar_estado(modulo: ModuleType) -> None:
    """La clasificación debe distinguir correcto, incorrecto y espurio."""
    patrones = np.array(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=int,
    )
    nombres = ("a", "b")

    correcto = modulo.clasificar_estado(
        np.array([1, 1], dtype=int),
        patrones,
        nombres,
        nombre_origen="a",
    )
    incorrecto = modulo.clasificar_estado(
        np.array([1, -1], dtype=int),
        patrones,
        nombres,
        nombre_origen="a",
    )
    espurio = modulo.clasificar_estado(
        np.array([-1, -1], dtype=int),
        patrones,
        nombres,
    )

    assert correcto.tipo == "almacenado_correcto"
    assert correcto.nombre == "a"
    assert correcto.distancia_minima == 0
    assert incorrecto.tipo == "almacenado_incorrecto"
    assert incorrecto.nombre == "b"
    assert espurio.tipo == "espurio"
    assert espurio.nombre is None
    assert espurio.distancia_minima == 1


def test_estado_invalido(modulo: ModuleType) -> None:
    """Un estado con valores fuera de {-1, 1} debe rechazarse."""
    pesos = np.zeros((3, 3))

    with pytest.raises(ValueError, match="1 y -1"):
        modulo.recuperar(
            pesos,
            np.array([1, 0, 1], dtype=int),
            modo="asincronica",
            orden="secuencial",
            max_barridos=5,
            semilla=0,
        )


def test_guardar_cargar_red(modulo: ModuleType, tmp_path: Path) -> None:
    """El archivo pesos.npz debe conservar pesos, patrones y metadatos."""
    dataset = modulo.cargar_dataset(RUTA_LETRAS)
    pesos = modulo.calcular_pesos(dataset.patrones)
    directorio = tmp_path / "red"

    modulo.guardar_red(directorio, dataset, pesos)
    red = modulo.cargar_red(directorio)

    assert (directorio / "pesos.npz").exists()
    assert (directorio / "patrones.csv").exists()
    assert np.allclose(red.pesos, pesos)
    assert np.array_equal(red.patrones, dataset.patrones)
    assert red.nombres == dataset.nombres
    assert red.filas == 5
    assert red.columnas == 5


def test_reproducibilidad_orden_aleatorio(modulo: ModuleType) -> None:
    """La misma semilla debe producir la misma recuperación."""
    dataset = modulo.cargar_dataset(RUTA_LETRAS)
    pesos = modulo.calcular_pesos(dataset.patrones)
    inicial = dataset.patrones[0].copy()
    inicial[0] = -inicial[0]

    primero = modulo.recuperar(
        pesos,
        inicial,
        modo="asincronica",
        orden="aleatorio",
        max_barridos=20,
        semilla=7,
    )
    segundo = modulo.recuperar(
        pesos,
        inicial,
        modo="asincronica",
        orden="aleatorio",
        max_barridos=20,
        semilla=7,
    )

    assert np.array_equal(primero.estado_final, segundo.estado_final)
    assert primero.barridos == segundo.barridos
    assert [paso.energia for paso in primero.historial] == [
        paso.energia for paso in segundo.historial
    ]


@pytest.mark.parametrize(
    ("seccion", "clave", "valor"),
    [
        ("modelo", "modo", "otro"),
        ("modelo", "orden", "otro"),
        ("modelo", "semilla", -1),
        ("modelo", "max_barridos", 0),
    ],
)
def test_validar_config(
    modulo: ModuleType,
    seccion: str,
    clave: str,
    valor: object,
) -> None:
    """La configuración debe rechazar valores fuera de rango."""
    config = deepcopy(modulo.CONFIG_POR_DEFECTO)
    config["datos"]["ruta"] = str(RUTA_LETRAS)
    config[seccion][clave] = valor

    with pytest.raises(ValueError):
        modulo.validar_config(config)


def test_resolver_config_precedencia(
    modulo: ModuleType,
    tmp_path: Path,
) -> None:
    """La CLI debe pisar el TOML y el directorio vacío debe derivarse."""
    config = deepcopy(modulo.CONFIG_POR_DEFECTO)
    config["datos"]["ruta"] = str(RUTA_LETRAS)
    ruta_config = tmp_path / "config.toml"
    modulo.guardar_config(config, ruta_config)

    argumentos = argparse.Namespace(
        config=ruta_config,
        datos=None,
        salida=None,
        modo="sincronica",
        orden="aleatorio",
        semilla=3,
        max_barridos=10,
    )

    resuelta = modulo.resolver_config(argumentos)

    assert resuelta["modelo"]["modo"] == "sincronica"
    assert resuelta["modelo"]["orden"] == "aleatorio"
    assert resuelta["modelo"]["semilla"] == 3
    assert resuelta["modelo"]["max_barridos"] == 10
    directorio = Path(resuelta["salida"]["directorio"]).as_posix()
    assert directorio.endswith("resultados/ej2/hopfield/letras_5x5")


def test_analizar_red(modulo: ModuleType) -> None:
    """El análisis debe devolver las tablas con las formas esperadas."""
    dataset = modulo.cargar_dataset(RUTA_LETRAS)
    pesos = modulo.calcular_pesos(dataset.patrones)

    analisis = modulo.analizar_red(
        dataset,
        pesos,
        modo="asincronica",
        orden="secuencial",
        max_barridos=20,
        semilla=0,
    )

    assert len(analisis.estabilidad) == 4
    assert len(analisis.estabilidad_neuronas) == 4 * 25
    assert len(analisis.energia) >= 4
    assert analisis.solapamiento.shape == (4, 4)
    assert analisis.hamming.shape == (4, 4)
    assert np.allclose(np.diag(analisis.solapamiento), 1.0)
    assert np.array_equal(np.diag(analisis.hamming), np.zeros(4, dtype=int))
    assert set(analisis.estabilidad["nombre"]) == {"J", "O", "T", "A"}
    assert set(analisis.estabilidad["tipo_resultado"]).issubset(
        {
            "almacenado_correcto",
            "almacenado_incorrecto",
            "espurio",
            "ciclo",
            "no_converge",
        }
    )
    assert "carga" in analisis.capacidad
    assert "hamming_min" in analisis.resumen_ortogonalidad


def test_main_integracion(modulo: ModuleType, tmp_path: Path) -> None:
    """El CLI debe generar todas las salidas sin escribir en resultados/."""
    directorio = tmp_path / "red"

    codigo = modulo.main(
        [
            "--datos",
            str(RUTA_LETRAS),
            "--salida",
            str(directorio),
            "--config",
            str(RUTA_CONFIG),
        ]
    )

    assert codigo == 0
    for nombre in (
        "config_usada.toml",
        "patrones.csv",
        "pesos.npz",
        "estabilidad.csv",
        "estabilidad_neuronas.csv",
        "solapamiento.csv",
        "hamming.csv",
        "ortogonalidad_resumen.csv",
        "capacidad.csv",
        "energia_entrenamiento.csv",
        "resumen.txt",
    ):
        assert (directorio / nombre).exists(), nombre

    with np.load(directorio / "pesos.npz", allow_pickle=False) as datos:
        assert datos["pesos"].shape == (25, 25)
        assert datos["patrones"].shape == (4, 25)

    estabilidad = pd.read_csv(directorio / "estabilidad.csv")
    patrones = pd.read_csv(directorio / "patrones.csv")
    assert len(estabilidad) == 4
    assert len(patrones) == 4
    assert (directorio / "resumen.txt").read_text(encoding="utf-8").strip()


def test_pesos_pseudoinversa_estabiliza_patrones_correlacionados(
    modulo: ModuleType,
) -> None:
    """Con patrones muy correlacionados Hebb falla y la pseudo-inversa no."""
    generador = np.random.default_rng(0)
    base = generador.choice([-1, 1], size=60)
    patrones = np.tile(base, (4, 1))
    # Cada patrón difiere de la base en un bloque propio de 6 píxeles.
    for indice in range(4):
        patrones[indice, indice * 6 : indice * 6 + 6] *= -1

    hebb = modulo.calcular_pesos(patrones, regla="hebb")
    pseudo = modulo.calcular_pesos(patrones, regla="pseudoinversa")

    estables_hebb = [
        np.all(modulo.calcular_margenes_estabilidad(hebb, p) > 0) for p in patrones
    ]
    estables_pseudo = [
        np.all(modulo.calcular_margenes_estabilidad(pseudo, p) > 0) for p in patrones
    ]
    assert not all(estables_hebb)
    assert all(estables_pseudo)
    np.testing.assert_allclose(pseudo, pseudo.T, atol=1e-12)
    assert np.all(np.diag(pseudo) == 0.0)


def test_pesos_pseudoinversa_patrones_ortogonales_igual_hebb(
    modulo: ModuleType,
) -> None:
    """Con patrones ortogonales la pseudo-inversa coincide con Hebb."""
    patrones = np.array([[1, 1, 1, 1], [1, -1, 1, -1]])
    np.testing.assert_allclose(
        modulo.calcular_pesos(patrones, regla="pseudoinversa"),
        modulo.calcular_pesos(patrones, regla="hebb"),
        atol=1e-12,
    )


def test_calcular_pesos_regla_invalida(modulo: ModuleType) -> None:
    """Una regla desconocida debe fallar."""
    with pytest.raises(ValueError):
        modulo.calcular_pesos(np.ones((1, 4)), regla="oja")


def test_energia_por_actualizacion(modulo: ModuleType) -> None:
    """La energía incremental coincide con la directa y nunca crece."""
    generador = np.random.default_rng(3)
    patrones = generador.choice([-1, 1], size=(3, 30))
    pesos = modulo.calcular_pesos(patrones)
    inicial = patrones[0].copy()
    inicial[:8] *= -1

    resultado = modulo.recuperar(pesos, inicial, registrar_actualizaciones=True)

    energias = np.asarray(resultado.energia_actualizaciones)
    assert energias.size == 1 + resultado.actualizaciones
    assert energias[0] == pytest.approx(resultado.energia_inicial)
    assert energias[-1] == pytest.approx(resultado.energia_final)
    assert np.all(np.diff(energias) <= 1e-12)
    # Sin pedirlo no se registra nada.
    assert modulo.recuperar(pesos, inicial).energia_actualizaciones is None


def test_config_regla(modulo: ModuleType, tmp_path: Path) -> None:
    """La regla se lee del TOML, la CLI la pisa y una config vieja es Hebb."""
    config = deepcopy(modulo.CONFIG_POR_DEFECTO)
    del config["modelo"]["regla"]
    modulo.validar_config(config)

    config["modelo"]["regla"] = "pseudoinversa"
    modulo.validar_config(config)
    config["modelo"]["regla"] = "otra"
    with pytest.raises(ValueError):
        modulo.validar_config(config)

    cli = argparse.Namespace(config=RUTA_CONFIG, regla="pseudoinversa")
    assert modulo.resolver_config(cli)["modelo"]["regla"] == "pseudoinversa"


def test_capacidad_por_subconjuntos(modulo: ModuleType) -> None:
    """Con patrones muy correlacionados Hebb sostiene dos pero no tres."""
    generador = np.random.default_rng(0)
    base = generador.choice([-1, 1], size=60)
    patrones = np.tile(base, (3, 1))
    # Cada patrón difiere de la base en un bloque propio de 6 píxeles: los
    # solapamientos son 0,6 y en el bloque propio h = ξ(1 - 0,6 - 0,6) < 0.
    for indice in range(3):
        patrones[indice, indice * 6 : indice * 6 + 6] *= -1

    tabla = modulo.capacidad_por_subconjuntos(patrones, ("a", "b", "c"), "hebb")

    assert len(tabla) == 3 + 3 + 1
    assert set(tabla.columns) >= {
        "cantidad",
        "patrones",
        "todos_estables",
        "patrones_estables",
        "margen_minimo",
    }
    por_cantidad = tabla.groupby("cantidad")["todos_estables"].all()
    assert por_cantidad[1] and por_cantidad[2] and not por_cantidad[3]

    pseudo = modulo.capacidad_por_subconjuntos(
        patrones, ("a", "b", "c"), "pseudoinversa"
    )
    assert pseudo["todos_estables"].all()


def test_main_guarda_capacidad_por_subconjuntos(
    modulo: ModuleType, tmp_path: Path
) -> None:
    """La corrida del motor guarda la tabla de capacidad por subconjuntos."""
    salida = tmp_path / "red"
    assert modulo.main(["--datos", str(RUTA_LETRAS), "--salida", str(salida)]) == 0
    tabla = pd.read_csv(salida / "capacidad_subconjuntos.csv")
    assert len(tabla) == 2**4 - 1


RUTA_QR = RAIZ / "datos" / "hopfield" / "qr_37x37.csv"


@pytest.fixture(scope="module")
def dataset_qr(modulo: ModuleType):
    """Los cinco QR reales del ejercicio 2.1."""
    return modulo.cargar_dataset(RUTA_QR)


def test_qr_reales_pseudoinversa_almacena_y_hebb_no(
    modulo: ModuleType, dataset_qr
) -> None:
    """Con los cinco QR, la pseudo-inversa los fija y Hebb cae al voto mayoritario."""
    pseudo = modulo.calcular_pesos(dataset_qr.patrones, "pseudoinversa")
    hebb = modulo.calcular_pesos(dataset_qr.patrones, "hebb")
    mezcla = modulo.estado_mezcla(dataset_qr.patrones)

    for patron in dataset_qr.patrones:
        assert modulo.calcular_margenes_estabilidad(pseudo, patron).min() > 0
        assert modulo.calcular_margenes_estabilidad(hebb, patron).min() < 0
        resultado = modulo.recuperar(hebb, patron)
        assert resultado.convergio
        assert np.array_equal(resultado.estado_final, mezcla)


def test_calcular_condicion(modulo: ModuleType, dataset_qr) -> None:
    """Rango y número de condición de X Xᵀ."""
    ortogonales = np.array([[1, 1, 1, 1], [1, -1, 1, -1]])
    condicion = modulo.calcular_condicion(ortogonales)
    assert condicion == {"patrones": 2, "rango": 2, "numero_condicion": 1.0}

    repetidos = np.array([[1, 1, -1, 1], [1, 1, -1, 1]])
    assert modulo.calcular_condicion(repetidos)["rango"] == 1

    qr = modulo.calcular_condicion(dataset_qr.patrones)
    assert qr["rango"] == 5
    assert qr["numero_condicion"] == pytest.approx(26.39, abs=0.01)


def test_estado_mezcla(modulo: ModuleType) -> None:
    """El estado mezcla es el voto mayoritario; un empate vale 1."""
    patrones = np.array([[1, 1, -1, 1], [1, -1, -1, 1], [1, 1, -1, -1]])
    assert modulo.estado_mezcla(patrones).tolist() == [1, 1, -1, 1]
    assert modulo.estado_mezcla(patrones[:2]).tolist() == [1, 1, -1, 1]


def test_calcular_energia_estados(modulo: ModuleType) -> None:
    """Energía, energía por neurona y estabilidad de los patrones y la mezcla."""
    patrones = np.array([[1, 1, 1, 1], [1, -1, 1, -1]])
    pesos = modulo.calcular_pesos(patrones)
    tabla = modulo.calcular_energia_estados(pesos, patrones, ("a", "b"))

    assert tabla["estado"].tolist() == ["a", "b", "mezcla"]
    for fila, estado in zip(
        tabla.itertuples(), [*patrones, modulo.estado_mezcla(patrones)], strict=True
    ):
        energia = modulo.calcular_energia(pesos, estado)
        assert fila.energia == pytest.approx(energia)
        assert fila.energia_por_neurona == pytest.approx(energia / 4)
    assert tabla["punto_fijo"].tolist()[:2] == [True, True]


def test_main_guarda_condicion_y_energia_estados(
    modulo: ModuleType, tmp_path: Path
) -> None:
    """El entrenamiento guarda la condición de X Xᵀ, las energías y la regla."""
    salida = tmp_path / "red"
    argumentos = ["--datos", str(RUTA_LETRAS), "--salida", str(salida)]
    assert modulo.main(argumentos + ["--regla", "pseudoinversa"]) == 0

    condicion = pd.read_csv(salida / "condicion.csv")
    assert condicion.loc[0, "rango"] == 4
    energias = pd.read_csv(salida / "energia_estados.csv")
    assert energias["estado"].tolist() == ["J", "O", "T", "A", "mezcla"]
    assert energias["punto_fijo"].tolist()[:4] == [True] * 4
    resumen = (salida / "resumen.txt").read_text(encoding="utf-8")
    assert "Regla: pseudoinversa" in resumen


@pytest.mark.parametrize("regla", ["hebb", "pseudoinversa"])
def test_configuraciones_qr(modulo: ModuleType, regla: str) -> None:
    """Los TOML de QR fijan la regla, los datos y la carpeta de salida."""
    cli = argparse.Namespace(config=RAIZ / "src" / "ej2" / f"hopfield-qr-{regla}.toml")
    config = modulo.resolver_config(cli)

    assert config["modelo"]["regla"] == regla
    assert config["modelo"]["modo"] == "asincronica"
    assert config["modelo"]["orden"] == "secuencial"
    assert config["modelo"]["max_barridos"] == 100
    assert Path(config["datos"]["ruta"]) == RUTA_QR
    salida = Path(config["salida"]["directorio"]).as_posix()
    assert salida.endswith(f"resultados/ej2/hopfield/qr_37x37/{regla}")
