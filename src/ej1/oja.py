#!/usr/bin/env python3
"""Índice de creadores de contenido con la regla de Oja (TP4, ejercicio 1.3).

El script lee ``datos/creadores.csv``, estandariza las mismas 27 variables de
1.1 y 1.2, entrena la regla de Oja y construye un índice por creador. Después
compara el índice con la proyección sobre PC1 ya calculada en 1.1 y guarda
tablas, figuras y un resumen en ``resultados/ej1/oja/``.

Referencia teórica: ``artif-eng.pdf``, capítulo 16, pp. 131-134.
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ / "src" / "ej1" / "oja.toml"

COLUMNAS_ID = ["creator_id", "name"]

COLUMNAS_EXCLUIDAS = [
    "price_ars",
    "estimated_argentina_reach",
    "total_engagement_per_1000",
    "organic_posts_pct",
]

REGLAS = ("exacta", "taylor")
DECAIMIENTOS_ETA = ("constante", "inversa", "raiz")

VARIANTES_REGLA = (
    ("exacta", 0.5),
    ("taylor", 0.5),
    ("taylor", 0.05),
)

CONFIG_POR_DEFECTO: dict = {
    "datos": {
        "ruta": "datos/creadores.csv",
        "columnas_excluidas": list(COLUMNAS_EXCLUIDAS),
    },
    "modelo": {
        "regla": "exacta",
        "epocas": 500,
        "eta_inicial": 0.5,
        "decaimiento_eta": "inversa",
    },
    "control": {
        "norma_max": 5.0,
        "ratio_max": 5.0,
        "ratio_min_convergencia": 0.99,
        "pearson_min_convergencia": 0.99,
    },
    "referencias": {
        "pca_directorio": "resultados/ej1/pca_0.95",
        "correlaciones": "resultados/ej1/eda/correlaciones.csv",
    },
    "reproducibilidad": {
        "semillas": [0, 1, 2],
    },
    "sensibilidad": {
        "eta_inicial": [0.01, 0.05, 0.1, 0.2, 0.5],
        "decaimiento_eta": ["constante", "inversa", "raiz"],
    },
    "salida": {
        "directorio": "resultados/ej1/oja",
    },
}


@dataclass
class ResultadoOja:
    """Resultado completo de una corrida de la regla de Oja."""

    pesos: np.ndarray
    historial: pd.DataFrame
    divergio: bool
    epoca_detencion: int
    motivo: str
    convergio: bool
    regla: str
    eta_inicial: float
    semilla: int
    ratio_varianza: float
    pearson_pc1: float
    norma_final: float


@dataclass
class ReferenciaPCA:
    """Resultados de PCA y correlaciones de 1.1 usados como referencia."""

    variables_modelo: list[str]
    variables_todas: list[str]
    cargas: np.ndarray
    scores: np.ndarray
    lambda1: float
    C: np.ndarray
    C_completa: np.ndarray


def _resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    if camino.is_absolute():
        return camino
    return RAIZ / camino


def _fusionar(base: dict, extra: dict) -> dict:
    """Fusiona dos diccionarios de configuración sin modificar el original."""
    resultado = deepcopy(base)
    for clave, valor in extra.items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _fusionar(resultado[clave], valor)
        else:
            resultado[clave] = deepcopy(valor)
    return resultado


def _formatear_valor_toml(valor: object) -> str:
    """Formatea un valor simple para escribir un archivo TOML."""
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, Path):
        valor = str(valor)
    if isinstance(valor, str):
        escapado = valor.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escapado}"'
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, (int, float)):
        return repr(valor)
    if isinstance(valor, (list, tuple)):
        return "[" + ", ".join(_formatear_valor_toml(item) for item in valor) + "]"
    raise TypeError(f"No se puede escribir en TOML: {valor!r}")


def escribir_config(config: dict, ruta: Path) -> None:
    """Escribe una configuración de secciones planas en formato TOML."""
    lineas: list[str] = []
    for seccion, valores in config.items():
        lineas.append(f"[{seccion}]")
        for clave, valor in valores.items():
            lineas.append(f"{clave} = {_formatear_valor_toml(valor)}")
        lineas.append("")
    ruta.write_text("\n".join(lineas).rstrip() + "\n", encoding="utf-8")


def cargar_config(ruta: str | Path) -> dict:
    """Lee la configuración TOML y la fusiona con los valores por defecto."""
    camino = _resolver_ruta(ruta)
    if not camino.exists():
        raise FileNotFoundError(f"No se encontró la configuración: {camino}")
    with camino.open("rb") as archivo:
        cargado = tomllib.load(archivo)
    return _fusionar(CONFIG_POR_DEFECTO, cargado)


def validar_config(config: dict) -> None:
    """Valida las claves y los valores de la configuración."""
    modelo = config["modelo"]
    if modelo["regla"] not in REGLAS:
        raise ValueError(f"Regla inválida: {modelo['regla']!r}")
    if not isinstance(modelo["epocas"], int) or modelo["epocas"] < 1:
        raise ValueError("La cantidad de épocas debe ser un entero positivo.")
    eta_inicial = float(modelo["eta_inicial"])
    if not 0.0 < eta_inicial <= 0.5:
        raise ValueError("eta_inicial debe estar en (0, 0.5].")
    if modelo["decaimiento_eta"] not in DECAIMIENTOS_ETA:
        raise ValueError(f"Decaimiento de eta inválido: {modelo['decaimiento_eta']!r}")

    control = config["control"]
    if float(control["norma_max"]) <= 0.0:
        raise ValueError("norma_max debe ser positivo.")
    if float(control["ratio_max"]) <= 0.0:
        raise ValueError("ratio_max debe ser positivo.")
    for clave in ("ratio_min_convergencia", "pearson_min_convergencia"):
        if not 0.0 < float(control[clave]) <= 1.0:
            raise ValueError(f"{clave} debe estar en (0, 1].")

    semillas = config["reproducibilidad"]["semillas"]
    if not semillas or not all(isinstance(semilla, int) for semilla in semillas):
        raise ValueError("Las semillas deben ser una lista no vacía de enteros.")

    sensibilidad = config["sensibilidad"]
    if not sensibilidad["eta_inicial"]:
        raise ValueError(
            "La lista de eta_inicial de sensibilidad no puede estar vacía."
        )
    if not sensibilidad["decaimiento_eta"]:
        raise ValueError(
            "La lista de decaimiento_eta de sensibilidad no puede estar vacía."
        )
    for eta in sensibilidad["eta_inicial"]:
        if not 0.0 < float(eta) <= 0.5:
            raise ValueError("Los eta_inicial de sensibilidad deben estar en (0, 0.5].")
    for decaimiento in sensibilidad["decaimiento_eta"]:
        if decaimiento not in DECAIMIENTOS_ETA:
            raise ValueError(f"Decaimiento de sensibilidad inválido: {decaimiento!r}")

    for clave in ("pca_directorio", "correlaciones"):
        if clave not in config["referencias"]:
            raise ValueError(f"Falta la referencia: {clave}")
    if "ruta" not in config["datos"]:
        raise ValueError("Falta la ruta de datos.")
    if "directorio" not in config["salida"]:
        raise ValueError("Falta el directorio de salida.")


def _entero_positivo(texto: str) -> int:
    """Convierte un argumento a entero positivo."""
    try:
        valor = int(texto)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Se esperaba un entero.") from error
    if valor < 1:
        raise argparse.ArgumentTypeError("El valor debe ser mayor o igual a 1.")
    return valor


def _lista_semillas(texto: str) -> list[int]:
    """Convierte una lista separada por comas en una lista de enteros."""
    try:
        semillas = [int(parte) for parte in texto.split(",") if parte.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Las semillas deben ser enteros.") from error
    if not semillas:
        raise argparse.ArgumentTypeError("Debe indicar al menos una semilla.")
    return semillas


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Índice de creadores de contenido con la regla de Oja. "
            "La configuración se puede pisar por línea de comandos."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(RUTA_CONFIG),
        help="Archivo TOML de configuración.",
    )
    parser.add_argument("--datos", default=None, help="Ruta del CSV de creadores.")
    parser.add_argument("--salida", default=None, help="Directorio de resultados.")
    parser.add_argument(
        "--regla",
        choices=REGLAS,
        default=None,
        help="Regla de actualización.",
    )
    parser.add_argument(
        "--epocas",
        type=_entero_positivo,
        default=None,
        help="Cantidad de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--eta-inicial",
        type=float,
        default=None,
        help="Tasa de aprendizaje inicial en (0, 0.5].",
    )
    parser.add_argument(
        "--decaimiento-eta",
        choices=DECAIMIENTOS_ETA,
        default=None,
        help="Horario de decaimiento de eta.",
    )
    parser.add_argument(
        "--semillas",
        type=_lista_semillas,
        default=None,
        help="Semillas separadas por comas, por ejemplo 0,1,2.",
    )
    return parser.parse_args(argv)


def aplicar_cli(config: dict, args: argparse.Namespace) -> dict:
    """Aplica los argumentos de línea de comandos sobre la configuración."""
    resultado = deepcopy(config)
    if args.datos is not None:
        resultado["datos"]["ruta"] = args.datos
    if args.salida is not None:
        resultado["salida"]["directorio"] = args.salida
    if args.regla is not None:
        resultado["modelo"]["regla"] = args.regla
    if args.epocas is not None:
        resultado["modelo"]["epocas"] = args.epocas
    if args.eta_inicial is not None:
        resultado["modelo"]["eta_inicial"] = args.eta_inicial
    if args.decaimiento_eta is not None:
        resultado["modelo"]["decaimiento_eta"] = args.decaimiento_eta
    if args.semillas is not None:
        resultado["reproducibilidad"]["semillas"] = args.semillas
    return resultado


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    """Lee el CSV de creadores."""
    camino = _resolver_ruta(ruta)
    if not camino.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {camino}")
    datos = pd.read_csv(camino)
    if datos.empty:
        raise ValueError("El dataset está vacío.")
    faltantes = [columna for columna in COLUMNAS_ID if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
    return datos


def columnas_entrada(
    datos: pd.DataFrame, columnas_excluidas: Sequence[str]
) -> list[str]:
    """Devuelve las columnas numéricas que entran al modelo."""
    numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
    excluidas = set(columnas_excluidas)
    return [columna for columna in numericas if columna not in excluidas]


def estandarizar(matriz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estandariza cada columna con media 0 y desvío muestral 1."""
    valores = np.asarray(matriz, dtype=float)
    if valores.ndim != 2:
        raise ValueError("La matriz de datos debe tener dos dimensiones.")
    medias = valores.mean(axis=0)
    desvios = valores.std(axis=0, ddof=1)
    if np.any(desvios == 0):
        raise ValueError("No se puede estandarizar una variable constante.")
    return (valores - medias) / desvios, medias, desvios


def horario_eta(t: int, eta_inicial: float, decaimiento: str) -> float:
    """Devuelve la tasa de aprendizaje para el paso global t."""
    if t < 0:
        raise ValueError("El paso de tiempo debe ser no negativo.")
    if eta_inicial <= 0.0:
        raise ValueError("eta_inicial debe ser positivo.")
    if decaimiento == "constante":
        return float(eta_inicial)
    if decaimiento == "inversa":
        return float(eta_inicial / (1.0 + t))
    if decaimiento == "raiz":
        return float(eta_inicial / np.sqrt(1.0 + t))
    raise ValueError(f"Decaimiento de eta inválido: {decaimiento!r}")


def actualizar_oja_exacta(
    pesos: np.ndarray, muestra: np.ndarray, eta: float
) -> np.ndarray:
    """Aplica la regla normalizada exacta de Oja (ecuaciones 16.7-16.8)."""
    pesos = np.asarray(pesos, dtype=float)
    muestra = np.asarray(muestra, dtype=float)
    salida = float(muestra @ pesos)
    nuevo = pesos + eta * salida * muestra
    norma = float(np.linalg.norm(nuevo))
    if norma == 0.0:
        raise ValueError("La actualización de Oja produjo un vector nulo.")
    return nuevo / norma


def actualizar_oja_taylor(
    pesos: np.ndarray, muestra: np.ndarray, eta: float
) -> np.ndarray:
    """Aplica la forma de Taylor de Oja (ecuación 16.9)."""
    pesos = np.asarray(pesos, dtype=float)
    muestra = np.asarray(muestra, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        salida = float(muestra @ pesos)
        return pesos + eta * salida * (muestra - salida * pesos)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de Pearson entre dos vectores."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def entrenar_oja(
    datos: np.ndarray,
    modelo: dict,
    semilla: int,
    C: np.ndarray,
    lambda1: float,
    pc1_scores: np.ndarray,
    control: dict,
) -> ResultadoOja:
    """Entrena una corrida de Oja y devuelve pesos, historial y control."""
    datos = np.asarray(datos, dtype=float)
    C = np.asarray(C, dtype=float)
    pc1_scores = np.asarray(pc1_scores, dtype=float)
    regla = modelo["regla"]
    if regla not in REGLAS:
        raise ValueError(f"Regla inválida: {regla!r}")
    epocas = int(modelo["epocas"])
    eta_inicial = float(modelo["eta_inicial"])
    decaimiento = modelo["decaimiento_eta"]
    if epocas < 1:
        raise ValueError("La cantidad de épocas debe ser positiva.")
    if lambda1 <= 0.0:
        raise ValueError("lambda1 debe ser positivo.")

    norma_max = float(control["norma_max"])
    ratio_max = float(control["ratio_max"])
    ratio_min = float(control["ratio_min_convergencia"])
    pearson_min = float(control["pearson_min_convergencia"])

    rng = np.random.default_rng(semilla)
    pesos = rng.uniform(0.0, 1.0, size=datos.shape[1])
    pesos = pesos / np.linalg.norm(pesos)

    historial: list[dict] = []
    paso = 0
    divergio = False
    epoca_detencion = epocas
    motivo = ""
    eta_actual = eta_inicial

    for epoca in range(1, epocas + 1):
        pesos_previos = pesos.copy()
        orden = rng.permutation(datos.shape[0])
        for indice in orden:
            eta_actual = horario_eta(paso, eta_inicial, decaimiento)
            if regla == "exacta":
                pesos = actualizar_oja_exacta(pesos, datos[indice], eta_actual)
            else:
                pesos = actualizar_oja_taylor(pesos, datos[indice], eta_actual)
            paso += 1

        norma = float(np.linalg.norm(pesos))
        varianza = float(pesos @ C @ pesos)
        ratio = varianza / lambda1 if lambda1 > 0.0 else float("nan")
        pearson = abs(_pearson(datos @ pesos, pc1_scores))
        delta = float(np.linalg.norm(pesos - pesos_previos))

        historial.append(
            {
                "semilla": semilla,
                "epoca": epoca,
                "eta": eta_actual,
                "norma_pesos": norma,
                "varianza_proyectada": varianza,
                "ratio_varianza": ratio,
                "pearson_pc1": pearson,
                "delta_pesos": delta,
            }
        )

        if not (
            np.all(np.isfinite(pesos))
            and np.isfinite(varianza)
            and np.isfinite(ratio)
            and np.isfinite(pearson)
        ):
            divergio = True
            epoca_detencion = epoca
            motivo = "valores no finitos"
            break
        if norma > norma_max:
            divergio = True
            epoca_detencion = epoca
            motivo = f"norma de pesos > {norma_max:g}"
            break
        if ratio > ratio_max:
            divergio = True
            epoca_detencion = epoca
            motivo = f"razón de varianza > {ratio_max:g}"
            break
        if ratio < 0.0:
            divergio = True
            epoca_detencion = epoca
            motivo = "razón de varianza negativa"
            break

    norma_final = float(np.linalg.norm(pesos))
    if np.isfinite(norma_final) and norma_final > 0.0:
        pesos_finales = pesos / norma_final
    else:
        pesos_finales = pesos

    if np.all(np.isfinite(pesos_finales)):
        varianza_final = float(pesos_finales @ C @ pesos_finales)
        ratio_final = varianza_final / lambda1 if lambda1 > 0.0 else float("nan")
        pearson_final = abs(_pearson(datos @ pesos_finales, pc1_scores))
    else:
        varianza_final = float("nan")
        ratio_final = float("nan")
        pearson_final = float("nan")

    convergio = bool(
        not divergio
        and np.isfinite(ratio_final)
        and np.isfinite(pearson_final)
        and ratio_final >= ratio_min
        and pearson_final >= pearson_min
    )

    return ResultadoOja(
        pesos=pesos_finales,
        historial=pd.DataFrame(historial),
        divergio=divergio,
        epoca_detencion=epoca_detencion,
        motivo=motivo,
        convergio=convergio,
        regla=regla,
        eta_inicial=eta_inicial,
        semilla=semilla,
        ratio_varianza=ratio_final,
        pearson_pc1=pearson_final,
        norma_final=norma_final,
    )


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV y falla con un mensaje claro si no existe."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la referencia: {ruta}")
    return pd.read_csv(ruta)


def cargar_referencia_pca(config: dict, datos: pd.DataFrame) -> ReferenciaPCA:
    """Carga y valida PC1 y la matriz de correlación ya calculadas en 1.1."""
    pca_directorio = _resolver_ruta(config["referencias"]["pca_directorio"])
    ruta_correlaciones = _resolver_ruta(config["referencias"]["correlaciones"])

    cargas_df = _leer_csv(pca_directorio / "cargas.csv")
    scores_df = _leer_csv(pca_directorio / "scores.csv")
    autovalores_df = _leer_csv(pca_directorio / "autovalores.csv")
    correlaciones_df = _leer_csv(ruta_correlaciones)

    if "PC1" not in cargas_df.columns:
        raise ValueError("cargas.csv no contiene la columna PC1.")
    cargas_df = cargas_df.set_index(cargas_df.columns[0])
    if "PC1" not in scores_df.columns or "creator_id" not in scores_df.columns:
        raise ValueError("scores.csv debe contener creator_id y PC1.")
    autovalores_df = autovalores_df.set_index(autovalores_df.columns[0])
    correlaciones_df = correlaciones_df.set_index(correlaciones_df.columns[0])

    variables_modelo = columnas_entrada(datos, config["datos"]["columnas_excluidas"])
    variables_todas = list(correlaciones_df.index)
    columnas_numericas = datos.select_dtypes(include=[np.number]).columns.tolist()
    if set(variables_todas) != set(columnas_numericas):
        raise ValueError(
            "La matriz de correlación no coincide con las variables numéricas."
        )
    faltantes = [
        variable
        for variable in variables_modelo
        if variable not in cargas_df.index or variable not in correlaciones_df.index
    ]
    if faltantes:
        raise ValueError(f"Faltan variables en las referencias: {faltantes}")

    if "PC1" not in autovalores_df.index:
        raise ValueError("autovalores.csv no contiene la fila PC1.")
    lambda1 = float(autovalores_df.loc["PC1", "autovalor"])

    cargas = cargas_df.loc[variables_modelo, "PC1"].to_numpy(dtype=float)
    C = correlaciones_df.loc[variables_modelo, variables_modelo].to_numpy(dtype=float)
    C_completa = correlaciones_df.loc[variables_todas, variables_todas].to_numpy(
        dtype=float
    )

    if not scores_df["creator_id"].is_unique:
        raise ValueError("scores.csv tiene creator_id duplicados.")
    scores_df = scores_df.set_index("creator_id")
    faltantes_scores = set(datos["creator_id"]) - set(scores_df.index)
    if faltantes_scores:
        raise ValueError(f"Faltan creadores en scores.csv: {sorted(faltantes_scores)}")
    scores = scores_df.loc[datos["creator_id"], "PC1"].to_numpy(dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores.csv contiene valores no finitos.")

    indice_followers = variables_modelo.index("followers")
    if cargas[indice_followers] <= 0.0:
        raise ValueError("La carga de followers en PC1 debe ser positiva.")
    if not np.allclose(C, C.T, atol=1e-10):
        raise ValueError("La matriz de correlación de 27 variables no es simétrica.")
    if not np.allclose(C_completa, C_completa.T, atol=1e-10):
        raise ValueError("La matriz de correlación de 31 variables no es simétrica.")
    if not np.allclose(np.diag(C), 1.0, atol=1e-8):
        raise ValueError("La diagonal de la matriz de 27 variables no es 1.")
    if not np.allclose(np.diag(C_completa), 1.0, atol=1e-8):
        raise ValueError("La diagonal de la matriz de 31 variables no es 1.")

    varianza_cargas = float(cargas @ C @ cargas)
    if not np.isclose(varianza_cargas, lambda1, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "PC1 y la matriz de correlación no son consistentes: "
            f"{varianza_cargas} != {lambda1}."
        )
    varianza_scores = float(np.var(scores, ddof=1))
    if not np.isclose(varianza_scores, lambda1, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "La varianza de los scores de PC1 no coincide con lambda1: "
            f"{varianza_scores} != {lambda1}."
        )
    if not np.isclose(np.mean(scores), 0.0, atol=1e-5):
        raise ValueError("La media de los scores de PC1 no es aproximadamente 0.")

    return ReferenciaPCA(
        variables_modelo=variables_modelo,
        variables_todas=variables_todas,
        cargas=cargas,
        scores=scores,
        lambda1=lambda1,
        C=C,
        C_completa=C_completa,
    )


def calcular_indice(datos: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Proyecta los datos estandarizados sobre el vector de pesos."""
    return np.asarray(datos, dtype=float) @ np.asarray(pesos, dtype=float)


def agregar_indice(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve la media y el desvío muestral del índice entre semillas."""
    valores = np.asarray(indices, dtype=float)
    if valores.ndim == 1:
        valores = valores.reshape(1, -1)
    media = valores.mean(axis=0)
    if valores.shape[0] > 1:
        desvio = valores.std(axis=0, ddof=1)
    else:
        desvio = np.zeros(valores.shape[1])
    return media, desvio


def calcular_ranking(valores: np.ndarray) -> np.ndarray:
    """Devuelve un ranking 1-based en orden descendente."""
    serie = pd.Series(np.asarray(valores, dtype=float))
    return serie.rank(ascending=False, method="min").to_numpy(dtype=int)


def correlaciones_indice(
    pesos: np.ndarray,
    C_completa: np.ndarray,
    variables_modelo: Sequence[str],
    variables_todas: Sequence[str],
    varianza_indice: float,
) -> pd.DataFrame:
    """Deriva las correlaciones del índice usando la matriz de 1.1."""
    pesos = np.asarray(pesos, dtype=float)
    C_completa = np.asarray(C_completa, dtype=float)
    if varianza_indice <= 0.0:
        raise ValueError("La varianza del índice debe ser positiva.")
    indices_modelo = [variables_todas.index(variable) for variable in variables_modelo]
    covarianzas = C_completa[indices_modelo, :].T @ pesos
    pearson = covarianzas / np.sqrt(varianza_indice)
    return pd.DataFrame(
        {
            "variable": list(variables_todas),
            "incluida_en_modelo": [
                variable in variables_modelo for variable in variables_todas
            ],
            "pearson": pearson,
        }
    )


def comparar_pc1(
    indice: np.ndarray,
    pc1_scores: np.ndarray,
    pesos: np.ndarray,
    C: np.ndarray,
    lambda1: float,
) -> dict:
    """Compara el índice de Oja con PC1 usando Pearson y la varianza."""
    indice = np.asarray(indice, dtype=float)
    pc1_scores = np.asarray(pc1_scores, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    C = np.asarray(C, dtype=float)
    varianza_indice = float(pesos @ C @ pesos)
    varianza_pc1 = float(np.var(pc1_scores, ddof=1))
    return {
        "pearson_indice_pc1": _pearson(indice, pc1_scores),
        "ratio_varianza": varianza_indice / lambda1,
        "varianza_indice": varianza_indice,
        "varianza_pc1": varianza_pc1,
        "lambda1": float(lambda1),
    }


def _fila_resultado(resultado: ResultadoOja, extra: dict) -> dict:
    """Convierte un resultado en una fila para las tablas comparativas."""
    return {
        **extra,
        "semilla": resultado.semilla,
        "divergio": resultado.divergio,
        "epoca_detencion": resultado.epoca_detencion,
        "motivo": resultado.motivo,
        "ratio_varianza": resultado.ratio_varianza,
        "pearson_pc1": resultado.pearson_pc1,
        "norma_final": resultado.norma_final,
        "convergio": resultado.convergio,
    }


def _resumir_resultados(
    resultados: pd.DataFrame, columnas_grupo: Sequence[str]
) -> pd.DataFrame:
    """Agrupa resultados y calcula media ± desvío sobre corridas válidas."""
    filas: list[dict] = []
    for claves, grupo in resultados.groupby(
        list(columnas_grupo), dropna=False, sort=True
    ):
        if not isinstance(claves, tuple):
            claves = (claves,)
        validos = grupo[~grupo["divergio"]]
        fila: dict = dict(zip(columnas_grupo, claves))
        for columna in ("ratio_varianza", "pearson_pc1", "norma_final"):
            valores = validos[columna].to_numpy(dtype=float)
            valores = valores[np.isfinite(valores)]
            if valores.size:
                fila[f"{columna}_media"] = float(np.mean(valores))
                fila[f"{columna}_desvio"] = (
                    float(np.std(valores, ddof=1)) if valores.size > 1 else 0.0
                )
            else:
                fila[f"{columna}_media"] = float("nan")
                fila[f"{columna}_desvio"] = 0.0
        fila["divergencias"] = int(grupo["divergio"].sum())
        fila["convergio"] = bool(grupo["convergio"].all())
        filas.append(fila)
    return pd.DataFrame(filas)


def resumir_reglas(resultados: pd.DataFrame) -> pd.DataFrame:
    """Resume la comparación de las tres reglas."""
    return _resumir_resultados(resultados, ["regla", "eta_inicial"])


def guardar_configuraciones_reglas(
    resumen: pd.DataFrame, config: dict, directorio: Path
) -> list[Path]:
    """Guarda la configuración de cada regla de Taylor que converge."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []
    for _, fila in resumen.iterrows():
        if fila["regla"] != "taylor" or not bool(fila["convergio"]):
            continue
        eta = float(fila["eta_inicial"])
        config_regla = deepcopy(config)
        config_regla["modelo"]["regla"] = "taylor"
        config_regla["modelo"]["eta_inicial"] = eta
        config_regla["salida"]["directorio"] = str(
            Path(config["salida"]["directorio"]) / f"regla_taylor_{eta:g}"
        )
        ruta = directorio / f"config_regla_taylor_{eta:g}.toml"
        escribir_config(config_regla, ruta)
        rutas.append(ruta)
    return rutas


def _guardar_figura(fig: plt.Figure, ruta: Path) -> None:
    """Guarda y cierra una figura."""
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _figura_convergencia(convergencia: pd.DataFrame, ruta: Path) -> None:
    """Grafica la convergencia de la corrida final por semilla."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for semilla, grupo in convergencia.groupby("semilla"):
        axes[0].plot(grupo["epoca"], grupo["ratio_varianza"], label=f"{semilla}")
        axes[1].plot(grupo["epoca"], grupo["pearson_pc1"], label=f"{semilla}")
        axes[2].plot(grupo["epoca"], grupo["norma_pesos"], label=f"{semilla}")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Razón de varianza")
    axes[1].set_title("Pearson con PC1")
    axes[2].set_title("Norma de los pesos")
    for eje in axes:
        eje.set_xlabel("época")
        eje.set_xscale("log")
        eje.grid(alpha=0.3)
    axes[0].legend(title="semilla", fontsize=8)
    _guardar_figura(fig, ruta)


def _figura_pesos(
    variables: Sequence[str],
    pesos_medio: np.ndarray,
    pesos_desvio: np.ndarray,
    cargas_pc1: np.ndarray,
    ruta: Path,
) -> None:
    """Compara las cargas de Oja con las de PC1."""
    orden = np.argsort(-np.abs(pesos_medio))
    posiciones = np.arange(len(variables))
    fig, eje = plt.subplots(figsize=(10, 10))
    eje.barh(
        posiciones,
        pesos_medio[orden],
        xerr=pesos_desvio[orden],
        color="#4c72b0",
        alpha=0.8,
        label="Oja",
    )
    eje.scatter(
        cargas_pc1[orden],
        posiciones,
        color="#c44e52",
        zorder=3,
        label="PC1",
    )
    eje.set_yticks(posiciones)
    eje.set_yticklabels([variables[indice] for indice in orden], fontsize=8)
    eje.invert_yaxis()
    eje.axvline(0.0, color="black", linewidth=0.8)
    eje.set_xlabel("carga estandarizada")
    eje.set_title("Cargas del índice de Oja y de PC1")
    eje.legend()
    eje.grid(axis="x", alpha=0.3)
    _guardar_figura(fig, ruta)


def _figura_indice(
    nombres: Sequence[str], indice: np.ndarray, desvio: np.ndarray, ruta: Path
) -> None:
    """Grafica el índice de los 50 creadores."""
    orden = np.argsort(-indice)
    posiciones = np.arange(len(indice))
    fig, eje = plt.subplots(figsize=(10, 14))
    eje.barh(
        posiciones,
        indice[orden],
        xerr=desvio[orden],
        color="#55a868",
        alpha=0.85,
    )
    eje.set_yticks(posiciones)
    eje.set_yticklabels([nombres[indice] for indice in orden], fontsize=7)
    eje.invert_yaxis()
    eje.axvline(0.0, color="black", linewidth=0.8)
    eje.set_xlabel("índice de Oja (unidades estandarizadas)")
    eje.set_title("Índice de creadores de contenido")
    eje.grid(axis="x", alpha=0.3)
    _guardar_figura(fig, ruta)


def _figura_comparacion_pc1(
    indice: np.ndarray,
    pc1: np.ndarray,
    metricas: dict,
    ruta: Path,
) -> None:
    """Grafica el índice de Oja contra la proyección sobre PC1."""
    fig, eje = plt.subplots(figsize=(6, 6))
    eje.scatter(indice, pc1, s=24, alpha=0.8)
    minimo = float(min(indice.min(), pc1.min()))
    maximo = float(max(indice.max(), pc1.max()))
    eje.plot([minimo, maximo], [minimo, maximo], color="gray", linestyle="--")
    eje.set_xlabel("índice de Oja")
    eje.set_ylabel("score de PC1")
    eje.set_title("Índice de Oja contra PC1")
    texto = (
        f"Pearson = {metricas['pearson_indice_pc1']:.4f}\n"
        f"razón de varianza = {metricas['ratio_varianza']:.4f}"
    )
    eje.text(
        0.05,
        0.95,
        texto,
        transform=eje.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    eje.grid(alpha=0.3)
    _guardar_figura(fig, ruta)


def _figura_indice_vs_negocio(
    datos: pd.DataFrame,
    indice: np.ndarray,
    correlaciones: pd.DataFrame,
    ruta: Path,
) -> None:
    """Grafica el índice contra followers y engagement total."""
    variables = ["followers", "total_engagement_per_1000"]
    titulos = ["Followers", "Engagement total por 1000"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for eje, variable, titulo in zip(axes, variables, titulos):
        valores = datos[variable].to_numpy(dtype=float)
        eje.scatter(valores, indice, s=24, alpha=0.8)
        eje.set_xscale("log")
        eje.set_xlabel(variable)
        eje.set_ylabel("índice de Oja")
        correlacion = correlaciones.loc[
            correlaciones["variable"] == variable, "pearson"
        ].iloc[0]
        eje.set_title(f"{titulo} (Pearson = {correlacion:.3f})")
        eje.grid(alpha=0.3)
    _guardar_figura(fig, ruta)


def _figura_sensibilidad(resumen: pd.DataFrame, ruta: Path) -> None:
    """Grafica la sensibilidad de la regla exacta a eta y al decaimiento."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    especificaciones = [
        ("ratio_varianza_media", "Razón de varianza"),
        ("pearson_pc1_media", "Pearson con PC1"),
    ]
    for eje, columna, titulo in zip(axes, *zip(*especificaciones)):
        tabla = resumen.pivot(
            index="eta_inicial",
            columns="decaimiento_eta",
            values=columna,
        )
        valores = tabla.to_numpy(dtype=float)
        imagen = eje.imshow(valores, aspect="auto", cmap="viridis")
        eje.set_xticks(range(len(tabla.columns)))
        eje.set_xticklabels(tabla.columns)
        eje.set_yticks(range(len(tabla.index)))
        eje.set_yticklabels([f"{valor:g}" for valor in tabla.index])
        eje.set_xlabel("decaimiento de eta")
        eje.set_ylabel("eta inicial")
        eje.set_title(titulo)
        for fila in range(valores.shape[0]):
            for columna_indice in range(valores.shape[1]):
                valor = valores[fila, columna_indice]
                if np.isfinite(valor):
                    eje.text(
                        columna_indice,
                        fila,
                        f"{valor:.4f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
        fig.colorbar(imagen, ax=eje)
    _guardar_figura(fig, ruta)


def _historial_medio(historiales: list[pd.DataFrame]) -> pd.DataFrame:
    """Calcula la media y el desvío por época de varios historiales."""
    combinado = pd.concat(
        [
            historial.assign(semilla=indice)
            for indice, historial in enumerate(historiales)
        ],
        ignore_index=True,
    )
    agregado = combinado.groupby("epoca").agg(
        norma_media=("norma_pesos", "mean"),
        norma_desvio=("norma_pesos", "std"),
        ratio_media=("ratio_varianza", "mean"),
        ratio_desvio=("ratio_varianza", "std"),
        pearson_media=("pearson_pc1", "mean"),
        pearson_desvio=("pearson_pc1", "std"),
    )
    return agregado.reset_index().fillna(0.0)


def _figura_comparacion_reglas(
    historiales: dict[tuple[str, float], dict[int, pd.DataFrame]],
    reglas: pd.DataFrame,
    ruta: Path,
) -> None:
    """Compara exacta, Taylor 0.5 y Taylor 0.05."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for (regla, eta), por_semilla in historiales.items():
        validas = reglas.loc[
            (reglas["regla"] == regla)
            & (reglas["eta_inicial"] == eta)
            & (~reglas["divergio"]),
            "semilla",
        ].tolist()
        lista = [
            historial
            for semilla, historial in por_semilla.items()
            if semilla in validas
        ]
        if not lista:
            continue
        media = _historial_medio(lista)
        etiqueta = f"{regla} η0={eta:g} (n={len(lista)})"
        for eje, columna_media, columna_desvio in zip(
            axes,
            ("ratio_media", "pearson_media", "norma_media"),
            ("ratio_desvio", "pearson_desvio", "norma_desvio"),
        ):
            eje.plot(media["epoca"], media[columna_media], label=etiqueta)
            eje.fill_between(
                media["epoca"],
                media[columna_media] - media[columna_desvio],
                media[columna_media] + media[columna_desvio],
                alpha=0.2,
            )
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Razón de varianza")
    axes[1].set_title("Pearson con PC1")
    axes[2].set_title("Norma de los pesos")
    for eje in axes:
        eje.set_xlabel("época")
        eje.set_xscale("log")
        eje.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    _guardar_figura(fig, ruta)


def _media_desvio(valores: Sequence[float]) -> tuple[float, float]:
    """Devuelve media y desvío muestral ignorando valores no finitos."""
    arreglo = np.asarray(valores, dtype=float)
    arreglo = arreglo[np.isfinite(arreglo)]
    if arreglo.size == 0:
        return float("nan"), float("nan")
    media = float(np.mean(arreglo))
    desvio = float(np.std(arreglo, ddof=1)) if arreglo.size > 1 else 0.0
    return media, desvio


def _resumen_texto(
    config: dict,
    referencia: ReferenciaPCA,
    resultados_finales: list[ResultadoOja],
    comparacion: dict,
    correlaciones: pd.DataFrame,
    sensibilidad_resumen: pd.DataFrame,
    reglas_resumen: pd.DataFrame,
    configuraciones: list[Path],
    salida: Path,
) -> str:
    """Construye el resumen numérico de la corrida."""
    lineas: list[str] = [
        "Resumen de ejecución — TP4, ejercicio 1.3 (Índice de Oja)",
        f"Datos: {_resolver_ruta(config['datos']['ruta'])}",
        f"Variables de entrada: {len(referencia.variables_modelo)}",
        "Creadores: 50",
        f"Regla final: {config['modelo']['regla']}",
        f"Épocas: {config['modelo']['epocas']}",
        f"eta inicial: {config['modelo']['eta_inicial']}",
        f"Decaimiento de eta: {config['modelo']['decaimiento_eta']}",
        f"Semillas: {config['reproducibilidad']['semillas']}",
        "",
        "Referencias de 1.1:",
        f"  cargas: {_resolver_ruta(config['referencias']['pca_directorio']) / 'cargas.csv'}",
        f"  scores: {_resolver_ruta(config['referencias']['pca_directorio']) / 'scores.csv'}",
        f"  autovalores: {_resolver_ruta(config['referencias']['pca_directorio']) / 'autovalores.csv'}",
        f"  correlaciones: {_resolver_ruta(config['referencias']['correlaciones'])}",
        "",
        "Validaciones numéricas de las referencias:",
        f"  v1^T C v1 = {float(referencia.cargas @ referencia.C @ referencia.cargas):.10f}",
        f"  lambda1 = {referencia.lambda1:.10f}",
        f"  var(PC1) = {float(np.var(referencia.scores, ddof=1)):.10f}",
        f"  media(PC1) = {float(np.mean(referencia.scores)):.10f}",
        "",
        "Corridas finales:",
        "semilla  ratio_varianza  pearson_pc1  norma_final  divergio  convergio",
    ]
    for resultado in resultados_finales:
        lineas.append(
            f"{resultado.semilla:>7}  {resultado.ratio_varianza:>14.6f}"
            f"  {resultado.pearson_pc1:>11.6f}  {resultado.norma_final:>11.6f}"
            f"  {str(resultado.divergio):>8}  {str(resultado.convergio):>9}"
        )
    ratios = [resultado.ratio_varianza for resultado in resultados_finales]
    pearson = [resultado.pearson_pc1 for resultado in resultados_finales]
    media_ratio, desvio_ratio = _media_desvio(ratios)
    media_pearson, desvio_pearson = _media_desvio(pearson)
    lineas.extend(
        [
            "",
            "Comparación con PC1:",
            f"  Pearson índice vs PC1: {comparacion['pearson_indice_pc1']:.10f}",
            f"  Razón de varianza: {comparacion['ratio_varianza']:.10f}",
            f"  Varianza del índice: {comparacion['varianza_indice']:.10f}",
            f"  Varianza de PC1: {comparacion['varianza_pc1']:.10f}",
            f"  lambda1: {comparacion['lambda1']:.10f}",
            "",
            "Media ± desvío de las corridas finales:",
            f"  ratio_varianza: {media_ratio:.6f} ± {desvio_ratio:.6f}",
            f"  pearson_pc1: {media_pearson:.6f} ± {desvio_pearson:.6f}",
            "",
            "Correlaciones del índice (Pearson):",
            f"{'variable':<32}{'incluida_en_modelo':>18}{'pearson':>12}",
        ]
    )
    for _, fila in correlaciones.iterrows():
        lineas.append(
            f"{str(fila['variable']):<32}"
            f"{str(bool(fila['incluida_en_modelo'])):>18}"
            f"{float(fila['pearson']):>12.6f}"
        )
    lineas.extend(
        [
            "",
            "Sensibilidad de la regla exacta:",
            "eta_inicial  decaimiento  ratio_media  ratio_desvio  pearson_media"
            "  pearson_desvio  divergencias  convergio",
        ]
    )
    for _, fila in sensibilidad_resumen.iterrows():
        lineas.append(
            f"{float(fila['eta_inicial']):>11g}  {fila['decaimiento_eta']:>12}"
            f"  {float(fila['ratio_varianza_media']):>11.6f}"
            f"  {float(fila['ratio_varianza_desvio']):>12.6f}"
            f"  {float(fila['pearson_pc1_media']):>13.6f}"
            f"  {float(fila['pearson_pc1_desvio']):>14.6f}"
            f"  {int(fila['divergencias']):>12}  {str(bool(fila['convergio'])):>9}"
        )
    lineas.extend(
        [
            "",
            "Comparación de reglas:",
            "Las medias excluyen las semillas que divergieron.",
            "regla  eta_inicial  ratio_media  pearson_media  divergencias  convergio",
        ]
    )
    for _, fila in reglas_resumen.iterrows():
        lineas.append(
            f"{fila['regla']:>6}  {float(fila['eta_inicial']):>11g}"
            f"  {float(fila['ratio_varianza_media']):>11.6f}"
            f"  {float(fila['pearson_pc1_media']):>13.6f}"
            f"  {int(fila['divergencias']):>12}  {str(bool(fila['convergio'])):>9}"
        )
    lineas.extend(
        [
            "",
            "Configuraciones guardadas:",
        ]
    )
    if configuraciones:
        for ruta in configuraciones:
            lineas.append(f"  {ruta}")
    else:
        lineas.append("  ninguna corrida de Taylor convergió")
    lineas.extend(
        [
            "",
            "Archivos generados:",
            f"  {salida}",
            "",
            "Nota: este resumen contiene solo resultados numéricos;",
            "la interpretación de marketing se discute fuera de estos archivos.",
        ]
    )
    return "\n".join(lineas) + "\n"


def _guardar_tablas(
    salida: Path,
    config: dict,
    normalizacion: pd.DataFrame,
    pesos: pd.DataFrame,
    indice: pd.DataFrame,
    comparacion: dict,
    correlaciones: pd.DataFrame,
    convergencia: pd.DataFrame,
    sensibilidad: pd.DataFrame,
    sensibilidad_resumen: pd.DataFrame,
    reglas: pd.DataFrame,
    reglas_resumen: pd.DataFrame,
) -> None:
    """Guarda todos los CSV de la corrida."""
    escribir_config(config, salida / "config_usada.toml")
    normalizacion.to_csv(salida / "normalizacion.csv", index=False)
    pesos.to_csv(salida / "pesos.csv", index=False)
    indice.to_csv(salida / "indice.csv", index=False)
    pd.DataFrame(
        [{"metrica": clave, "valor": valor} for clave, valor in comparacion.items()]
    ).to_csv(salida / "comparacion_pc1.csv", index=False)
    correlaciones.to_csv(salida / "correlaciones_indice.csv", index=False)
    convergencia.to_csv(salida / "convergencia.csv", index=False)
    sensibilidad.to_csv(salida / "sensibilidad.csv", index=False)
    sensibilidad_resumen.to_csv(salida / "sensibilidad_resumen.csv", index=False)
    reglas.to_csv(salida / "comparacion_reglas.csv", index=False)
    reglas_resumen.to_csv(salida / "comparacion_reglas_resumen.csv", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la corrida completa del índice de Oja."""
    args = parse_args(argv)
    config = cargar_config(args.config)
    config = aplicar_cli(config, args)
    validar_config(config)

    salida = _resolver_ruta(config["salida"]["directorio"])
    salida.mkdir(parents=True, exist_ok=True)

    datos = cargar_datos(config["datos"]["ruta"])
    variables = columnas_entrada(datos, config["datos"]["columnas_excluidas"])
    matriz = datos[variables].to_numpy(dtype=float)
    estandarizados, medias, desvios = estandarizar(matriz)
    referencia = cargar_referencia_pca(config, datos)

    semillas = list(config["reproducibilidad"]["semillas"])
    control = config["control"]
    modelo = config["modelo"]

    resultados_finales = [
        entrenar_oja(
            estandarizados,
            modelo,
            semilla,
            referencia.C,
            referencia.lambda1,
            referencia.scores,
            control,
        )
        for semilla in semillas
    ]

    pesos = np.vstack([resultado.pesos for resultado in resultados_finales])
    indice_followers = variables.index("followers")
    for fila in range(pesos.shape[0]):
        if pesos[fila, indice_followers] < 0.0:
            pesos[fila] *= -1.0

    indices = np.vstack([calcular_indice(estandarizados, fila) for fila in pesos])
    indice, indice_desvio = agregar_indice(indices)
    ranking = calcular_ranking(indice)
    pc1_ranking = calcular_ranking(referencia.scores)
    pesos_medio = pesos.mean(axis=0)
    pesos_desvio = (
        pesos.std(axis=0, ddof=1) if pesos.shape[0] > 1 else np.zeros(pesos.shape[1])
    )

    indice_df = pd.DataFrame(
        {
            "creator_id": datos["creator_id"],
            "name": datos["name"],
            "indice": indice,
            "indice_desvio": indice_desvio,
            "ranking": ranking,
            "pc1": referencia.scores,
            "pc1_ranking": pc1_ranking,
            "followers": datos["followers"],
            "price_usd": datos["price_usd"],
            "total_engagement_per_1000": datos["total_engagement_per_1000"],
        }
    ).sort_values("ranking")
    indice_df = indice_df.reset_index(drop=True)

    comparacion = comparar_pc1(
        indice,
        referencia.scores,
        pesos_medio,
        referencia.C,
        referencia.lambda1,
    )
    correlaciones = correlaciones_indice(
        pesos_medio,
        referencia.C_completa,
        variables,
        referencia.variables_todas,
        comparacion["varianza_indice"],
    )

    pesos_df = pd.DataFrame({"variable": variables})
    for posicion, semilla in enumerate(semillas):
        pesos_df[f"peso_semilla_{semilla}"] = pesos[posicion]
    pesos_df["peso_medio"] = pesos_medio
    pesos_df["peso_desvio"] = pesos_desvio
    pesos_df["carga_pc1"] = referencia.cargas

    normalizacion_df = pd.DataFrame(
        {"variable": variables, "media": medias, "desvio": desvios}
    )

    convergencia_df = pd.concat(
        [
            resultado.historial.assign(regla=resultado.regla)
            for resultado in resultados_finales
        ],
        ignore_index=True,
    )

    sensibilidad_filas: list[dict] = []
    for eta in config["sensibilidad"]["eta_inicial"]:
        for decaimiento in config["sensibilidad"]["decaimiento_eta"]:
            for semilla in semillas:
                modelo_sensibilidad = {
                    "regla": "exacta",
                    "epocas": modelo["epocas"],
                    "eta_inicial": eta,
                    "decaimiento_eta": decaimiento,
                }
                resultado = entrenar_oja(
                    estandarizados,
                    modelo_sensibilidad,
                    semilla,
                    referencia.C,
                    referencia.lambda1,
                    referencia.scores,
                    control,
                )
                sensibilidad_filas.append(
                    _fila_resultado(
                        resultado,
                        {
                            "eta_inicial": float(eta),
                            "decaimiento_eta": decaimiento,
                        },
                    )
                )
    sensibilidad_df = pd.DataFrame(sensibilidad_filas)
    sensibilidad_resumen = _resumir_resultados(
        sensibilidad_df, ["eta_inicial", "decaimiento_eta"]
    )

    reglas_filas: list[dict] = []
    historiales_reglas: dict[tuple[str, float], dict[int, pd.DataFrame]] = {}
    for regla, eta in VARIANTES_REGLA:
        historiales_reglas[(regla, eta)] = {}
        for semilla in semillas:
            modelo_regla = {
                "regla": regla,
                "epocas": modelo["epocas"],
                "eta_inicial": eta,
                "decaimiento_eta": modelo["decaimiento_eta"],
            }
            resultado = entrenar_oja(
                estandarizados,
                modelo_regla,
                semilla,
                referencia.C,
                referencia.lambda1,
                referencia.scores,
                control,
            )
            reglas_filas.append(
                _fila_resultado(
                    resultado,
                    {"regla": regla, "eta_inicial": float(eta)},
                )
            )
            historiales_reglas[(regla, eta)][semilla] = resultado.historial
    reglas_df = pd.DataFrame(reglas_filas)
    reglas_resumen = resumir_reglas(reglas_df)
    configuraciones = guardar_configuraciones_reglas(reglas_resumen, config, salida)

    _guardar_tablas(
        salida,
        config,
        normalizacion_df,
        pesos_df,
        indice_df,
        comparacion,
        correlaciones,
        convergencia_df,
        sensibilidad_df,
        sensibilidad_resumen,
        reglas_df,
        reglas_resumen,
    )

    _figura_convergencia(convergencia_df, salida / "fig_convergencia.png")
    _figura_pesos(
        variables,
        pesos_medio,
        pesos_desvio,
        referencia.cargas,
        salida / "fig_pesos.png",
    )
    _figura_indice(
        datos["name"].to_numpy(),
        indice,
        indice_desvio,
        salida / "fig_indice.png",
    )
    _figura_comparacion_pc1(
        indice,
        referencia.scores,
        comparacion,
        salida / "fig_comparacion_pc1.png",
    )
    _figura_indice_vs_negocio(
        datos,
        indice,
        correlaciones,
        salida / "fig_indice_vs_negocio.png",
    )
    _figura_sensibilidad(
        sensibilidad_resumen,
        salida / "fig_sensibilidad.png",
    )
    _figura_comparacion_reglas(
        historiales_reglas,
        reglas_df,
        salida / "fig_comparacion_reglas.png",
    )

    resumen = _resumen_texto(
        config,
        referencia,
        resultados_finales,
        comparacion,
        correlaciones,
        sensibilidad_resumen,
        reglas_resumen,
        configuraciones,
        salida,
    )
    (salida / "resumen.txt").write_text(resumen, encoding="utf-8")

    print(f"Resultados guardados en: {salida}")
    print(
        "Corrida final: "
        f"{len(resultados_finales)} semillas, "
        f"{sum(not resultado.divergio for resultado in resultados_finales)} "
        "sin divergencia."
    )
    print(
        "Comparación con PC1: "
        f"Pearson={comparacion['pearson_indice_pc1']:.6f}, "
        f"razón de varianza={comparacion['ratio_varianza']:.6f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
