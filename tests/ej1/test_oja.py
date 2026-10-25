"""Tests del ejercicio 1.3: índice de Oja."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "oja.py"
RUTA_CONFIG = RAIZ / "src" / "ej1" / "oja.toml"
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"
RUTA_PCA = RAIZ / "resultados" / "ej1" / "pca_0.95"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script como módulo para poder probar sus funciones."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location("oja", RUTA_SCRIPT)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


def datos_sinteticos_con_senal(
    n_muestras: int = 300,
    n_variables: int = 5,
    semilla: int = 1,
) -> np.ndarray:
    """Genera datos estandarizados con una dirección dominante clara."""
    rng = np.random.default_rng(semilla)
    factor = rng.normal(size=n_muestras)
    datos = rng.normal(scale=0.5, size=(n_muestras, n_variables))
    for columna in range(n_variables):
        datos[:, columna] += (4.0 - 0.2 * columna) * factor
    return (datos - datos.mean(axis=0)) / datos.std(axis=0, ddof=1)


def referencias_sinteticas(
    datos: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Calcula C, lambda_1 y los scores de PC1 de datos ya estandarizados."""
    n = datos.shape[0]
    C = (datos.T @ datos) / (n - 1)
    autovalores, autovectores = np.linalg.eigh(C)
    v1 = autovectores[:, -1]
    if v1[0] < 0:
        v1 = -v1
    lambda1 = float(autovalores[-1])
    return C, lambda1, datos @ v1


def modelo_base(
    regla: str = "exacta",
    epocas: int = 200,
    eta_inicial: float = 0.5,
) -> dict:
    """Devuelve una configuración de modelo chica y válida."""
    return {
        "regla": regla,
        "epocas": epocas,
        "eta_inicial": eta_inicial,
        "decaimiento_eta": "inversa",
    }


def control_base() -> dict:
    """Devuelve los umbrales de control usados en los tests."""
    return {
        "norma_max": 5.0,
        "ratio_max": 5.0,
        "ratio_min_convergencia": 0.99,
        "pearson_min_convergencia": 0.99,
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


def test_actualizacion_exacta(modulo: ModuleType) -> None:
    """La regla exacta debe coincidir con el cálculo manual y dar norma 1."""
    pesos = np.array([1.0, 0.0])
    muestra = np.array([2.0, 1.0])
    eta = 0.1

    esperado = pesos + eta * 2.0 * muestra
    esperado = esperado / np.linalg.norm(esperado)

    resultado = modulo.actualizar_oja_exacta(pesos, muestra, eta)

    assert np.allclose(resultado, esperado)
    assert np.isclose(np.linalg.norm(resultado), 1.0)


def test_actualizacion_taylor(modulo: ModuleType) -> None:
    """La forma de Taylor debe coincidir con la ecuación 16.9."""
    pesos = np.array([1.0, 0.0])
    muestra = np.array([2.0, 1.0])
    eta = 0.1

    esperado = np.array([1.0, 0.2])
    resultado = modulo.actualizar_oja_taylor(pesos, muestra, eta)

    assert np.allclose(resultado, esperado)


def test_exacta_y_taylor_coinciden_para_eta_pequena(modulo: ModuleType) -> None:
    """Para eta pequeño, la forma de Taylor aproxima a la regla exacta."""
    rng = np.random.default_rng(0)
    pesos = rng.normal(size=5)
    pesos = pesos / np.linalg.norm(pesos)
    muestra = rng.normal(size=5)
    eta = 1e-6

    exacta = modulo.actualizar_oja_exacta(pesos, muestra, eta)
    taylor = modulo.actualizar_oja_taylor(pesos, muestra, eta)

    assert np.allclose(exacta, taylor, atol=1e-8)


def test_horario_eta(modulo: ModuleType) -> None:
    """Verifica los extremos, la monotonía y la divergencia de la suma."""
    assert modulo.horario_eta(0, 0.5, "inversa") == pytest.approx(0.5)
    assert modulo.horario_eta(1, 0.5, "inversa") == pytest.approx(0.25)
    assert modulo.horario_eta(0, 0.5, "raiz") == pytest.approx(0.5)
    assert modulo.horario_eta(1, 0.5, "raiz") == pytest.approx(0.5 / np.sqrt(2.0))
    assert modulo.horario_eta(100, 0.5, "constante") == pytest.approx(0.5)

    valores = [modulo.horario_eta(t, 0.5, "inversa") for t in range(100)]
    assert all(valor > 0.0 for valor in valores)
    assert all(a >= b for a, b in zip(valores, valores[1:]))

    suma_inversa = sum(modulo.horario_eta(t, 0.5, "inversa") for t in range(10000))
    suma_raiz = sum(modulo.horario_eta(t, 0.5, "raiz") for t in range(10000))
    assert suma_inversa > 1.0
    assert suma_raiz > 1.0

    with pytest.raises(ValueError):
        modulo.horario_eta(0, 0.5, "lineal")
    with pytest.raises(ValueError):
        modulo.horario_eta(-1, 0.5, "inversa")


def test_entrenar_oja_exacta_converge(modulo: ModuleType) -> None:
    """La regla exacta debe converger a la dirección de PC1."""
    datos = datos_sinteticos_con_senal()
    C, lambda1, pc1 = referencias_sinteticas(datos)

    resultado = modulo.entrenar_oja(
        datos,
        modelo_base("exacta", epocas=200, eta_inicial=0.5),
        semilla=0,
        C=C,
        lambda1=lambda1,
        pc1_scores=pc1,
        control=control_base(),
    )

    assert not resultado.divergio
    assert resultado.convergio
    assert resultado.pesos.shape == (datos.shape[1],)
    assert np.isclose(np.linalg.norm(resultado.pesos), 1.0, atol=1e-6)
    assert len(resultado.historial) == 200
    assert {
        "epoca",
        "eta",
        "norma_pesos",
        "varianza_proyectada",
        "ratio_varianza",
        "pearson_pc1",
        "delta_pesos",
    }.issubset(resultado.historial.columns)


def test_entrenar_oja_taylor_005_converge(modulo: ModuleType) -> None:
    """La forma de Taylor con eta 0.05 debe converger a PC1."""
    datos = datos_sinteticos_con_senal()
    C, lambda1, pc1 = referencias_sinteticas(datos)

    resultado = modulo.entrenar_oja(
        datos,
        modelo_base("taylor", epocas=1000, eta_inicial=0.05),
        semilla=0,
        C=C,
        lambda1=lambda1,
        pc1_scores=pc1,
        control=control_base(),
    )

    assert not resultado.divergio
    assert resultado.convergio


def test_entrenar_oja_taylor_05_divergencia_y_corte(
    modulo: ModuleType,
) -> None:
    """Taylor con eta 0.5 debe detectar la divergencia y cortar antes."""
    rng = np.random.default_rng(3)
    datos = rng.normal(scale=100.0, size=(100, 5))
    C, lambda1, pc1 = referencias_sinteticas(datos)

    resultado = modulo.entrenar_oja(
        datos,
        modelo_base("taylor", epocas=500, eta_inicial=0.5),
        semilla=0,
        C=C,
        lambda1=lambda1,
        pc1_scores=pc1,
        control=control_base(),
    )

    assert resultado.divergio
    assert 0 < resultado.epoca_detencion < 500
    assert resultado.motivo
    assert len(resultado.historial) == resultado.epoca_detencion


def test_referencia_pca_real(modulo: ModuleType) -> None:
    """Las referencias de 1.1 deben ser consistentes y no recalcularse."""
    datos = modulo.cargar_datos(RUTA_DATOS)
    config = modulo.cargar_config(RUTA_CONFIG)

    referencia = modulo.cargar_referencia_pca(config, datos)
    indice_followers = referencia.variables_modelo.index("followers")

    assert len(referencia.variables_modelo) == 27
    assert len(referencia.variables_todas) == 31
    assert referencia.cargas.shape == (27,)
    assert referencia.scores.shape == (50,)
    assert referencia.C.shape == (27, 27)
    assert referencia.C_completa.shape == (31, 31)
    assert referencia.lambda1 > 0.0
    assert referencia.cargas[indice_followers] > 0.0
    assert np.allclose(
        referencia.cargas @ referencia.C @ referencia.cargas,
        referencia.lambda1,
        rtol=1e-6,
    )
    assert np.allclose(
        np.var(referencia.scores, ddof=1),
        referencia.lambda1,
        rtol=1e-6,
    )
    assert np.isclose(np.mean(referencia.scores), 0.0, atol=1e-6)
    assert np.allclose(referencia.C, referencia.C.T)
    assert np.allclose(np.diag(referencia.C), 1.0)
    assert np.allclose(referencia.C_completa, referencia.C_completa.T)
    assert np.allclose(np.diag(referencia.C_completa), 1.0)


def test_referencia_pca_alinea_por_creator_id(
    modulo: ModuleType, tmp_path: Path
) -> None:
    """Los scores deben alinearse por creator_id, no por orden de fila."""
    datos = modulo.cargar_datos(RUTA_DATOS)
    config = modulo.cargar_config(RUTA_CONFIG)

    shutil.copy(RUTA_PCA / "cargas.csv", tmp_path / "cargas.csv")
    shutil.copy(RUTA_PCA / "autovalores.csv", tmp_path / "autovalores.csv")

    scores = pd.read_csv(RUTA_PCA / "scores.csv")
    scores.sample(frac=1.0, random_state=7).to_csv(tmp_path / "scores.csv", index=False)

    config_temp = deepcopy(config)
    config_temp["referencias"]["pca_directorio"] = str(tmp_path)

    referencia_ordenada = modulo.cargar_referencia_pca(config, datos)
    referencia_mezclada = modulo.cargar_referencia_pca(config_temp, datos)

    assert np.allclose(referencia_mezclada.scores, referencia_ordenada.scores)


def test_referencia_pca_falla_si_no_existe(modulo: ModuleType, tmp_path: Path) -> None:
    """Si faltan las referencias de 1.1, debe fallar sin recalcular."""
    datos = modulo.cargar_datos(RUTA_DATOS)
    config = modulo.cargar_config(RUTA_CONFIG)
    config_temp = deepcopy(config)
    config_temp["referencias"]["pca_directorio"] = str(tmp_path)

    with pytest.raises(FileNotFoundError):
        modulo.cargar_referencia_pca(config_temp, datos)


def test_agregar_indice_y_ranking(modulo: ModuleType) -> None:
    """El índice agregado y el ranking deben seguir la definición del plan."""
    datos = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])

    indice_0 = modulo.calcular_indice(datos, np.array([1.0, 0.0]))
    indice_1 = modulo.calcular_indice(datos, np.array([0.0, 1.0]))

    assert np.allclose(indice_0, datos[:, 0])
    assert np.allclose(indice_1, datos[:, 1])

    media, desvio = modulo.agregar_indice(np.vstack([indice_0, indice_1]))
    assert np.allclose(media, (indice_0 + indice_1) / 2.0)
    assert np.allclose(desvio, np.std([indice_0, indice_1], axis=0, ddof=1))

    media_una, desvio_una = modulo.agregar_indice(indice_0.reshape(1, -1))
    assert np.allclose(media_una, indice_0)
    assert np.allclose(desvio_una, 0.0)

    ranking = modulo.calcular_ranking(np.array([0.5, -0.2, 1.0]))
    assert np.array_equal(ranking, np.array([2, 3, 1]))


def test_correlaciones_indice_desde_matriz(modulo: ModuleType) -> None:
    """Las correlaciones deben derivarse de C sin recalcular desde los datos."""
    rng = np.random.default_rng(5)
    datos = rng.normal(size=(50, 5))
    datos = (datos - datos.mean(axis=0)) / datos.std(axis=0, ddof=1)
    C_completa = np.corrcoef(datos, rowvar=False)

    variables = ["a", "b", "c", "d", "e"]
    variables_modelo = variables[:4]
    pesos = np.array([0.5, -0.3, 0.2, 0.4])
    pesos = pesos / np.linalg.norm(pesos)

    indice = modulo.calcular_indice(datos[:, :4], pesos)
    varianza = float(np.var(indice, ddof=1))

    tabla = modulo.correlaciones_indice(
        pesos, C_completa, variables_modelo, variables, varianza
    )
    esperado = [
        float(np.corrcoef(indice, datos[:, columna])[0, 1])
        for columna in range(datos.shape[1])
    ]

    assert tabla["variable"].tolist() == variables
    assert np.allclose(tabla["pearson"].to_numpy(), esperado)
    assert tabla["incluida_en_modelo"].tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]


def test_comparar_pc1(modulo: ModuleType) -> None:
    """La comparación debe usar solo Pearson y la razón de varianza."""
    datos = datos_sinteticos_con_senal()
    C, lambda1, pc1 = referencias_sinteticas(datos)
    autovalores, autovectores = np.linalg.eigh(C)
    pesos = autovectores[:, -1]
    if pesos[0] < 0:
        pesos = -pesos
    indice = datos @ pesos

    metricas = modulo.comparar_pc1(indice, pc1, pesos, C, lambda1)

    assert set(metricas) == {
        "pearson_indice_pc1",
        "ratio_varianza",
        "varianza_indice",
        "varianza_pc1",
        "lambda1",
    }
    assert metricas["pearson_indice_pc1"] == pytest.approx(1.0)
    assert metricas["ratio_varianza"] == pytest.approx(1.0)
    assert metricas["varianza_indice"] == pytest.approx(lambda1)
    assert metricas["varianza_pc1"] == pytest.approx(lambda1)


def test_reproducibilidad(modulo: ModuleType) -> None:
    """La misma semilla debe producir los mismos pesos y el mismo índice."""
    datos = datos_sinteticos_con_senal()
    C, lambda1, pc1 = referencias_sinteticas(datos)
    modelo = modelo_base("exacta", epocas=100, eta_inicial=0.5)

    resultado_1 = modulo.entrenar_oja(
        datos,
        modelo,
        semilla=0,
        C=C,
        lambda1=lambda1,
        pc1_scores=pc1,
        control=control_base(),
    )
    resultado_2 = modulo.entrenar_oja(
        datos,
        modelo,
        semilla=0,
        C=C,
        lambda1=lambda1,
        pc1_scores=pc1,
        control=control_base(),
    )

    assert np.array_equal(resultado_1.pesos, resultado_2.pesos)
    pd.testing.assert_frame_equal(resultado_1.historial, resultado_2.historial)
    assert np.array_equal(
        modulo.calcular_indice(datos, resultado_1.pesos),
        modulo.calcular_indice(datos, resultado_2.pesos),
    )


def test_config_y_cli(modulo: ModuleType) -> None:
    """La precedencia debe ser CLI > TOML > valores por defecto."""
    config = modulo.cargar_config(RUTA_CONFIG)

    assert config["modelo"]["regla"] == "exacta"
    assert config["modelo"]["epocas"] == 5
    assert config["modelo"]["eta_inicial"] == 0.5
    assert config["modelo"]["decaimiento_eta"] == "inversa"

    args = modulo.parse_args(
        [
            "--regla",
            "taylor",
            "--eta-inicial",
            "0.05",
            "--epocas",
            "10",
            "--semillas",
            "0,2",
        ]
    )
    config_cli = modulo.aplicar_cli(config, args)

    assert config_cli["modelo"]["regla"] == "taylor"
    assert config_cli["modelo"]["eta_inicial"] == 0.05
    assert config_cli["modelo"]["epocas"] == 10
    assert config_cli["reproducibilidad"]["semillas"] == [0, 2]
    modulo.validar_config(config_cli)


@pytest.mark.parametrize(
    ("clave", "valor"),
    [
        ("regla", "otra"),
        ("eta_inicial", 0.6),
        ("decaimiento_eta", "lineal"),
        ("epocas", 0),
    ],
)
def test_config_invalida(modulo: ModuleType, clave: str, valor: object) -> None:
    """La configuración debe rechazar valores fuera de lo permitido."""
    config = deepcopy(modulo.cargar_config(RUTA_CONFIG))
    config["modelo"][clave] = valor

    with pytest.raises(ValueError):
        modulo.validar_config(config)


def test_resumir_reglas_y_guardar_configuraciones(
    modulo: ModuleType, tmp_path: Path
) -> None:
    """Solo las reglas de Taylor que convergen guardan su configuración."""
    filas = [
        {
            "regla": "exacta",
            "eta_inicial": 0.5,
            "semilla": 0,
            "divergio": False,
            "ratio_varianza": 0.999,
            "pearson_pc1": 0.999,
            "norma_final": 1.0,
            "convergio": True,
        },
        {
            "regla": "exacta",
            "eta_inicial": 0.5,
            "semilla": 1,
            "divergio": False,
            "ratio_varianza": 0.998,
            "pearson_pc1": 0.998,
            "norma_final": 1.0,
            "convergio": True,
        },
        {
            "regla": "taylor",
            "eta_inicial": 0.5,
            "semilla": 0,
            "divergio": True,
            "ratio_varianza": np.nan,
            "pearson_pc1": np.nan,
            "norma_final": np.nan,
            "convergio": False,
        },
        {
            "regla": "taylor",
            "eta_inicial": 0.5,
            "semilla": 1,
            "divergio": False,
            "ratio_varianza": 0.995,
            "pearson_pc1": 0.995,
            "norma_final": 1.0,
            "convergio": True,
        },
        {
            "regla": "taylor",
            "eta_inicial": 0.05,
            "semilla": 0,
            "divergio": False,
            "ratio_varianza": 0.999,
            "pearson_pc1": 0.999,
            "norma_final": 1.0,
            "convergio": True,
        },
        {
            "regla": "taylor",
            "eta_inicial": 0.05,
            "semilla": 1,
            "divergio": False,
            "ratio_varianza": 0.998,
            "pearson_pc1": 0.998,
            "norma_final": 1.0,
            "convergio": True,
        },
    ]
    resultados = pd.DataFrame(filas)

    resumen = modulo.resumir_reglas(resultados)

    assert len(resumen) == 3
    assert bool(resumen.loc[(resumen["regla"] == "exacta"), "convergio"].iloc[0])
    assert not bool(
        resumen.loc[
            (resumen["regla"] == "taylor") & (resumen["eta_inicial"] == 0.5),
            "convergio",
        ].iloc[0]
    )
    assert bool(
        resumen.loc[
            (resumen["regla"] == "taylor") & (resumen["eta_inicial"] == 0.05),
            "convergio",
        ].iloc[0]
    )

    config = modulo.cargar_config(RUTA_CONFIG)
    rutas = modulo.guardar_configuraciones_reglas(resumen, config, tmp_path)

    assert len(rutas) == 1
    assert (tmp_path / "config_regla_taylor_0.05.toml").exists()
    assert not (tmp_path / "config_regla_taylor_0.5.toml").exists()
    with (tmp_path / "config_regla_taylor_0.05.toml").open("rb") as archivo:
        config_guardada = tomllib.load(archivo)
    assert config_guardada["modelo"]["regla"] == "taylor"
    assert config_guardada["modelo"]["eta_inicial"] == 0.05


def test_integracion_smoke(modulo: ModuleType, tmp_path: Path) -> None:
    """La corrida reducida debe generar todas las salidas esperadas."""
    modulo.main(
        [
            "--config",
            str(RUTA_CONFIG),
            "--epocas",
            "5",
            "--semillas",
            "0",
            "--salida",
            str(tmp_path),
        ]
    )

    archivos = [
        "config_usada.toml",
        "normalizacion.csv",
        "pesos.csv",
        "indice.csv",
        "comparacion_pc1.csv",
        "correlaciones_indice.csv",
        "convergencia.csv",
        "sensibilidad.csv",
        "sensibilidad_resumen.csv",
        "comparacion_reglas.csv",
        "comparacion_reglas_resumen.csv",
        "resumen.txt",
        "fig_convergencia.png",
        "fig_pesos.png",
        "fig_indice.png",
        "fig_comparacion_pc1.png",
        "fig_indice_vs_negocio.png",
        "fig_sensibilidad.png",
        "fig_comparacion_reglas.png",
    ]
    for nombre in archivos:
        ruta = tmp_path / nombre
        assert ruta.exists(), f"Falta la salida: {nombre}"
        assert ruta.stat().st_size > 0, f"La salida está vacía: {nombre}"

    indice = pd.read_csv(tmp_path / "indice.csv")
    assert len(indice) == 50
    assert {
        "creator_id",
        "name",
        "indice",
        "indice_desvio",
        "ranking",
        "pc1",
        "pc1_ranking",
    }.issubset(indice.columns)

    pesos = pd.read_csv(tmp_path / "pesos.csv")
    assert len(pesos) == 27
    assert "carga_pc1" in pesos.columns

    normalizacion = pd.read_csv(tmp_path / "normalizacion.csv")
    assert len(normalizacion) == 27
    assert {"variable", "media", "desvio"}.issubset(normalizacion.columns)

    convergencia = pd.read_csv(tmp_path / "convergencia.csv")
    assert 1 <= len(convergencia) <= 5
    assert {
        "semilla",
        "epoca",
        "eta",
        "norma_pesos",
        "varianza_proyectada",
        "ratio_varianza",
        "pearson_pc1",
        "delta_pesos",
    }.issubset(convergencia.columns)

    sensibilidad = pd.read_csv(tmp_path / "sensibilidad.csv")
    assert len(sensibilidad) == 15

    reglas = pd.read_csv(tmp_path / "comparacion_reglas.csv")
    assert len(reglas) == 3
    assert {
        ("exacta", 0.5),
        ("taylor", 0.5),
        ("taylor", 0.05),
    } == set(zip(reglas["regla"], reglas["eta_inicial"]))

    comparacion = pd.read_csv(tmp_path / "comparacion_pc1.csv")
    assert {"metrica", "valor"}.issubset(comparacion.columns)
    valores = comparacion.set_index("metrica")["valor"]
    assert np.isfinite(valores["pearson_indice_pc1"])
    assert np.isfinite(valores["ratio_varianza"])

    with (tmp_path / "config_usada.toml").open("rb") as archivo:
        config_guardada = tomllib.load(archivo)
    assert config_guardada["modelo"]["epocas"] == 5
    assert config_guardada["reproducibilidad"]["semillas"] == [0]

    assert "Oja" in (tmp_path / "resumen.txt").read_text(encoding="utf-8")
