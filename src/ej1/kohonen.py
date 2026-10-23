#!/usr/bin/env python3
"""Red de Kohonen para agrupar creadores de contenido (TP4, ejercicio 1.2).

El script lee ``datos/creadores.csv``, estandariza las variables, entrena una
red de Kohonen y guarda pesos, asignaciones, métricas, la matriz U y las
figuras del mapa en ``resultados/ej1/kohonen_base/`` por defecto.

Referencia teórica: ``artif-eng.pdf``, capítulo 14, pp. 121-124.
"""

from __future__ import annotations

import argparse
import textwrap
import time
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"
RUTA_CONFIG = RAIZ / "src" / "ej1" / "kohonen.toml"
RUTA_RESULTADOS = RAIZ / "resultados" / "ej1" / "kohonen_base"

COLUMNAS_ID = ["creator_id", "name"]

COLUMNAS_EXCLUIDAS = [
    "price_ars",
    "estimated_argentina_reach",
    "total_engagement_per_1000",
    "organic_posts_pct",
]

TOPOLOGIAS = ("rectangular", "hexagonal")
DISTANCIAS = ("euclidiana", "exponencial")
INICIALIZACIONES = ("uniforme", "muestras")
ESQUEMAS_RADIO = ("constante", "lineal", "exponencial", "inversa")
ESQUEMAS_ETA = ("lineal", "exponencial", "inversa")

CONFIG_POR_DEFECTO: dict = {
    "datos": {
        "ruta": "datos/creadores.csv",
        "estandarizar": True,
        "columnas_excluidas": list(COLUMNAS_EXCLUIDAS),
    },
    "modelo": {
        "grilla": 6,
        "topologia": "rectangular",
        "distancia": "euclidiana",
        "inicializacion": "muestras",
        "radio_inicial_relativo": 0.5,
        "radio_final": 1.0,
        "decaimiento_radio": "lineal",
        "eta_inicial": 0.5,
        "eta_final": 0.001,
        "decaimiento_eta": "lineal",
        "iteraciones": 20000,
    },
    "reproducibilidad": {
        "semillas": [0, 1, 2],
    },
    "salida": {
        "directorio": "resultados/ej1/kohonen_base",
    },
}


@dataclass
class ResultadoEntrenamiento:
    """Resultado completo de una corrida de la red de Kohonen."""

    pesos: np.ndarray
    bmu: np.ndarray
    distancias_muestras: np.ndarray
    qe: float
    te: float
    unidades_muertas: int
    ocupacion: float
    semilla: int


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


def guardar_config(config: dict, ruta: Path) -> None:
    """Guarda una configuración plana de secciones en formato TOML."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    lineas: list[str] = []
    for seccion, valores in config.items():
        lineas.append(f"[{seccion}]")
        for clave, valor in valores.items():
            lineas.append(f"{clave} = {_formatear_valor_toml(valor)}")
        lineas.append("")

    ruta.write_text("\n".join(lineas), encoding="utf-8")


def validar_config(config: dict) -> None:
    """Valida las secciones y los valores principales de la configuración."""
    if not isinstance(config, dict):
        raise ValueError("La configuración debe ser un diccionario.")

    for seccion in ("datos", "modelo", "reproducibilidad", "salida"):
        if seccion not in config or not isinstance(config[seccion], dict):
            raise ValueError(f"Falta la sección obligatoria: {seccion}")

    datos = config["datos"]
    if not isinstance(datos.get("ruta"), str) or not datos["ruta"]:
        raise ValueError("datos.ruta debe ser un texto no vacío.")

    columnas = datos.get("columnas_excluidas")
    if not isinstance(columnas, list) or not all(
        isinstance(columna, str) for columna in columnas
    ):
        raise ValueError("datos.columnas_excluidas debe ser una lista de textos.")

    if not isinstance(datos.get("estandarizar", True), bool):
        raise ValueError("datos.estandarizar debe ser un booleano.")

    modelo = config["modelo"]
    grilla = modelo.get("grilla")
    if not isinstance(grilla, int) or isinstance(grilla, bool) or grilla < 2:
        raise ValueError("modelo.grilla debe ser un entero mayor o igual que 2.")

    if modelo.get("topologia") not in TOPOLOGIAS:
        raise ValueError(f"modelo.topologia debe ser una de {TOPOLOGIAS}.")
    if modelo.get("distancia") not in DISTANCIAS:
        raise ValueError(f"modelo.distancia debe ser una de {DISTANCIAS}.")
    if modelo.get("inicializacion") not in INICIALIZACIONES:
        raise ValueError(f"modelo.inicializacion debe ser una de {INICIALIZACIONES}.")

    radio_relativo = modelo.get("radio_inicial_relativo")
    radio_final = modelo.get("radio_final")
    if not isinstance(radio_relativo, (int, float)) or radio_relativo <= 0.0:
        raise ValueError("modelo.radio_inicial_relativo debe ser positivo.")
    if not isinstance(radio_final, (int, float)) or radio_final <= 0.0:
        raise ValueError("modelo.radio_final debe ser positivo.")
    if radio_relativo * grilla < radio_final - 1e-12:
        raise ValueError(
            "El radio inicial (radio_inicial_relativo * grilla) debe ser "
            "mayor o igual que radio_final."
        )

    if modelo.get("decaimiento_radio") not in ESQUEMAS_RADIO:
        raise ValueError(f"modelo.decaimiento_radio debe ser uno de {ESQUEMAS_RADIO}.")

    eta_inicial = modelo.get("eta_inicial")
    eta_final = modelo.get("eta_final")
    if not isinstance(eta_inicial, (int, float)) or not 0.0 < eta_inicial < 1.0:
        raise ValueError("modelo.eta_inicial debe estar en (0, 1).")
    if not isinstance(eta_final, (int, float)) or not 0.0 < eta_final < eta_inicial:
        raise ValueError("modelo.eta_final debe ser positivo y menor que eta_inicial.")

    if modelo.get("decaimiento_eta") not in ESQUEMAS_ETA:
        raise ValueError(f"modelo.decaimiento_eta debe ser uno de {ESQUEMAS_ETA}.")

    iteraciones = modelo.get("iteraciones")
    if not isinstance(iteraciones, int) or isinstance(iteraciones, bool):
        raise ValueError("modelo.iteraciones debe ser un entero.")
    if iteraciones < 1:
        raise ValueError("modelo.iteraciones debe ser mayor que cero.")

    semillas = config["reproducibilidad"].get("semillas")
    if (
        not isinstance(semillas, list)
        or not semillas
        or not all(
            isinstance(semilla, int) and not isinstance(semilla, bool)
            for semilla in semillas
        )
    ):
        raise ValueError("reproducibilidad.semillas debe ser una lista de enteros.")

    directorio = config["salida"].get("directorio")
    if not isinstance(directorio, (str, Path)) or not str(directorio):
        raise ValueError("salida.directorio debe ser un texto no vacío.")


def cargar_config(ruta: Path) -> dict:
    """Lee un TOML y completa las claves ausentes con los valores por defecto."""
    with Path(ruta).open("rb") as archivo:
        leida = tomllib.load(archivo)

    config = _fusionar(CONFIG_POR_DEFECTO, leida)
    validar_config(config)
    return config


def lista_semillas(texto: str) -> list[int]:
    """Convierte una lista separada por comas en semillas enteras."""
    try:
        valores = [int(parte.strip()) for parte in texto.split(",") if parte.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Las semillas deben ser enteros separados por comas, por ejemplo 0,1,2."
        ) from error

    if not valores:
        raise argparse.ArgumentTypeError("Debe indicar al menos una semilla.")
    return valores


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Red de Kohonen para los creadores de contenido.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG,
        help="Archivo TOML con la configuración base.",
    )
    parser.add_argument("--datos", type=Path, default=None, help="CSV de entrada.")
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Directorio donde guardar los resultados.",
    )
    parser.add_argument(
        "--grilla", type=int, default=None, help="Tamaño k de la grilla."
    )
    parser.add_argument(
        "--topologia",
        choices=TOPOLOGIAS,
        default=None,
        help="Topología de la grilla.",
    )
    parser.add_argument(
        "--distancia",
        choices=DISTANCIAS,
        default=None,
        help="Medida de similitud para elegir la BMU.",
    )
    parser.add_argument(
        "--inicializacion",
        choices=INICIALIZACIONES,
        default=None,
        help="Método de inicialización de pesos.",
    )
    parser.add_argument(
        "--radio-inicial-relativo",
        type=float,
        default=None,
        help="Radio inicial como fracción de la grilla.",
    )
    parser.add_argument(
        "--radio-final",
        type=float,
        default=None,
        help="Radio final de la vecindad.",
    )
    parser.add_argument(
        "--decaimiento-radio",
        choices=ESQUEMAS_RADIO,
        default=None,
        help="Esquema de decaimiento del radio.",
    )
    parser.add_argument(
        "--eta-inicial",
        type=float,
        default=None,
        help="Tasa de aprendizaje inicial.",
    )
    parser.add_argument(
        "--eta-final",
        type=float,
        default=None,
        help="Tasa de aprendizaje final.",
    )
    parser.add_argument(
        "--decaimiento-eta",
        choices=ESQUEMAS_ETA,
        default=None,
        help="Esquema de decaimiento de la tasa de aprendizaje.",
    )
    parser.add_argument(
        "--iteraciones",
        type=int,
        default=None,
        help="Cantidad total de actualizaciones.",
    )
    parser.add_argument(
        "--semillas",
        type=lista_semillas,
        default=None,
        help="Semillas separadas por comas.",
    )
    return parser.parse_args(argv)


def resolver_config(config: dict, args: argparse.Namespace) -> dict:
    """Aplica los valores de la CLI sobre la configuración leída."""
    resultado = deepcopy(config)

    if args.datos is not None:
        resultado["datos"]["ruta"] = str(args.datos)
    if args.salida is not None:
        resultado["salida"]["directorio"] = str(args.salida)

    for clave in (
        "grilla",
        "topologia",
        "distancia",
        "inicializacion",
        "radio_inicial_relativo",
        "radio_final",
        "decaimiento_radio",
        "eta_inicial",
        "eta_final",
        "decaimiento_eta",
        "iteraciones",
    ):
        valor = getattr(args, clave, None)
        if valor is not None:
            resultado["modelo"][clave] = valor

    if args.semillas is not None:
        resultado["reproducibilidad"]["semillas"] = list(args.semillas)

    validar_config(resultado)
    return resultado


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    if camino.is_absolute():
        return camino
    return RAIZ / camino


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lee y valida el CSV de creadores."""
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")

    datos = pd.read_csv(ruta)
    if datos.empty:
        raise ValueError("El dataset está vacío.")

    for columna in COLUMNAS_ID:
        if columna not in datos.columns:
            raise ValueError(f"Falta la columna obligatoria: {columna}")

    if datos[COLUMNAS_ID].isna().any().any():
        raise ValueError("Hay valores nulos en las columnas identificadoras.")
    if datos["creator_id"].duplicated().any():
        raise ValueError("Hay creator_id duplicados.")
    if datos.isna().any().any():
        raise ValueError("Hay valores nulos en el dataset.")

    numericas = datos.select_dtypes(include=np.number)
    if numericas.shape[1] == 0:
        raise ValueError("El dataset no tiene variables numéricas.")
    if not np.isfinite(numericas.to_numpy(dtype=float)).all():
        raise ValueError("Hay valores infinitos en las variables numéricas.")

    return datos


def columnas_entrada(
    datos: pd.DataFrame, columnas_excluidas: Sequence[str]
) -> list[str]:
    """Devuelve las variables numéricas que entran a la red."""
    excluidas = set(columnas_excluidas)
    numericas = datos.select_dtypes(include=np.number).columns.tolist()
    return [columna for columna in numericas if columna not in excluidas]


def estandarizar(matriz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estandariza cada columna con media 0 y desvío muestral 1."""
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim != 2:
        raise ValueError("La matriz de datos debe tener dos dimensiones.")

    medias = matriz.mean(axis=0)
    desvios = matriz.std(axis=0, ddof=1)
    if np.any(desvios == 0):
        raise ValueError("No se puede estandarizar una variable constante.")

    return (matriz - medias) / desvios, medias, desvios


def preparar_entrada(
    matriz: np.ndarray, aplicar: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve la entrada del modelo y los parámetros de normalización."""
    if aplicar:
        return estandarizar(matriz)

    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim != 2:
        raise ValueError("La matriz de datos debe tener dos dimensiones.")

    return matriz.copy(), np.zeros(matriz.shape[1]), np.ones(matriz.shape[1])


def distancias_grilla(k: int, topologia: str) -> np.ndarray:
    """Calcula las distancias entre todas las neuronas de la grilla."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError("La grilla debe tener k >= 2.")
    if topologia not in TOPOLOGIAS:
        raise ValueError(f"Topología desconocida: {topologia}")

    filas = np.repeat(np.arange(k), k)
    columnas = np.tile(np.arange(k), k)

    if topologia == "rectangular":
        diferencias_filas = filas[:, None] - filas[None, :]
        diferencias_columnas = columnas[:, None] - columnas[None, :]
        return np.sqrt(diferencias_filas**2 + diferencias_columnas**2)

    q = columnas - filas // 2
    r = filas
    diferencia_q = q[:, None] - q[None, :]
    diferencia_r = r[:, None] - r[None, :]
    return (
        np.abs(diferencia_q)
        + np.abs(diferencia_r)
        + np.abs(diferencia_q + diferencia_r)
    ) / 2.0


def _normalizar_filas(matriz: np.ndarray) -> np.ndarray:
    """Normaliza cada fila a norma unitaria, dejando las nulas en cero."""
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return np.divide(
        matriz,
        normas,
        out=np.zeros_like(matriz, dtype=float),
        where=normas > 0.0,
    )


def distancias_pesos_muestras(pesos: np.ndarray, muestras: np.ndarray) -> np.ndarray:
    """Devuelve la distancia euclidiana de cada muestra a cada neurona."""
    pesos = np.asarray(pesos, dtype=float)
    muestras = np.asarray(muestras, dtype=float)
    return np.linalg.norm(muestras[:, None, :] - pesos[None, :, :], axis=2)


def seleccionar_bmus(
    pesos: np.ndarray,
    muestras: np.ndarray,
    distancia: str = "euclidiana",
) -> np.ndarray:
    """Selecciona la BMU de cada muestra de forma vectorizada."""
    pesos = np.asarray(pesos, dtype=float)
    muestras = np.asarray(muestras, dtype=float)

    if distancia == "euclidiana":
        return np.argmin(distancias_pesos_muestras(pesos, muestras), axis=1)

    if distancia == "exponencial":
        pesos_normales = _normalizar_filas(pesos)
        muestras_normales = _normalizar_filas(muestras)
        diferencias = muestras_normales[:, None, :] - pesos_normales[None, :, :]
        similitudes = -np.sum(diferencias**2, axis=2)
        return np.argmax(similitudes, axis=1)

    raise ValueError(f"Distancia desconocida: {distancia}")


def seleccionar_bmu(
    pesos: np.ndarray,
    muestra: np.ndarray,
    distancia: str = "euclidiana",
) -> int:
    """Selecciona la BMU de una única muestra."""
    muestra = np.asarray(muestra, dtype=float).reshape(1, -1)
    return int(seleccionar_bmus(pesos, muestra, distancia)[0])


def actualizar_pesos(
    pesos: np.ndarray,
    muestra: np.ndarray,
    bmu: int,
    mascara_vecindad: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Aplica la regla de Kohonen a la BMU y su vecindad."""
    del bmu
    nuevos = np.array(pesos, dtype=float, copy=True)
    mascara = np.asarray(mascara_vecindad, dtype=bool)
    nuevos[mascara] += eta * (np.asarray(muestra, dtype=float) - nuevos[mascara])
    return nuevos


def _calcular_esquema(
    cantidad: int,
    valor_inicial: float,
    valor_final: float,
    esquema: str,
    nombre: str,
) -> np.ndarray:
    """Calcula un esquema decreciente de radio o tasa de aprendizaje."""
    if cantidad < 1:
        raise ValueError(f"{nombre}: la cantidad de iteraciones debe ser positiva.")
    if valor_inicial <= 0.0 or valor_final <= 0.0:
        raise ValueError(f"{nombre}: los valores deben ser positivos.")

    if cantidad == 1:
        return np.array([valor_inicial], dtype=float)

    if esquema == "constante":
        return np.full(cantidad, valor_inicial, dtype=float)

    if valor_final > valor_inicial:
        raise ValueError(f"{nombre}: el valor final no puede superar al inicial.")

    avance = np.linspace(0.0, 1.0, cantidad)

    if esquema == "lineal":
        return valor_inicial + (valor_final - valor_inicial) * avance

    if esquema == "exponencial":
        return valor_inicial * (valor_final / valor_inicial) ** avance

    if esquema == "inversa":
        if np.isclose(valor_inicial, valor_final):
            return np.full(cantidad, valor_final, dtype=float)
        return valor_inicial / (1.0 + avance * (valor_inicial / valor_final - 1.0))

    raise ValueError(f"{nombre}: esquema desconocido: {esquema}")


def calcular_radios(
    cantidad: int,
    radio_inicial: float,
    radio_final: float,
    esquema: str,
) -> np.ndarray:
    """Calcula la secuencia de radios de vecindad."""
    return _calcular_esquema(
        cantidad,
        radio_inicial,
        radio_final,
        esquema,
        "radio",
    )


def calcular_tasas(
    cantidad: int,
    eta_inicial: float,
    eta_final: float,
    esquema: str,
) -> np.ndarray:
    """Calcula la secuencia de tasas de aprendizaje."""
    return _calcular_esquema(
        cantidad,
        eta_inicial,
        eta_final,
        esquema,
        "tasa",
    )


def matriz_u(pesos: np.ndarray, k: int, topologia: str) -> np.ndarray:
    """Calcula la matriz U con las distancias a los vecinos inmediatos."""
    pesos = np.asarray(pesos, dtype=float)
    distancias = distancias_grilla(k, topologia)
    matriz = np.full((k, k), np.nan, dtype=float)

    for indice in range(k * k):
        vecinos = np.flatnonzero(distancias[indice] == 1.0)
        if vecinos.size == 0:
            continue
        distancias_pesos = np.linalg.norm(pesos[vecinos] - pesos[indice], axis=1)
        matriz.flat[indice] = float(np.mean(distancias_pesos))

    return matriz


def calcular_qe(distancias_muestras: np.ndarray) -> float:
    """Calcula el error de cuantización medio."""
    distancias = np.asarray(distancias_muestras, dtype=float)
    if distancias.size == 0:
        return 0.0
    return float(np.mean(np.min(distancias, axis=1)))


def calcular_te(
    distancias_muestras: np.ndarray,
    distancias_grilla_neuronal: np.ndarray,
) -> float:
    """Calcula el error topográfico a partir de la primera y segunda BMU."""
    distancias = np.asarray(distancias_muestras, dtype=float)
    if distancias.shape[0] == 0 or distancias.shape[1] < 2:
        return 0.0

    orden = np.argsort(distancias, axis=1)
    primeras = orden[:, 0]
    segundas = orden[:, 1]
    no_vecinas = distancias_grilla_neuronal[primeras, segundas] > 1.0 + 1e-12
    return float(np.mean(no_vecinas))


def contar_unidades_muertas(bmus: np.ndarray, cantidad_neuronas: int) -> int:
    """Cuenta las neuronas que nunca ganaron."""
    bmus = np.asarray(bmus, dtype=int)
    if bmus.size == 0:
        return cantidad_neuronas
    return int(cantidad_neuronas - np.unique(bmus).size)


def calcular_ocupacion(bmus: np.ndarray, cantidad_neuronas: int) -> float:
    """Calcula la fracción de neuronas que ganaron al menos una vez."""
    if cantidad_neuronas <= 0:
        return 0.0
    muertas = contar_unidades_muertas(bmus, cantidad_neuronas)
    return float(1.0 - muertas / cantidad_neuronas)


def _inicializar_pesos(
    matriz: np.ndarray,
    k: int,
    inicializacion: str,
    generador: np.random.Generator,
) -> np.ndarray:
    """Inicializa los pesos con muestras o con valores uniformes."""
    cantidad_neuronas = k * k
    cantidad_variables = matriz.shape[1]

    if inicializacion == "muestras":
        indices = generador.integers(0, matriz.shape[0], size=cantidad_neuronas)
        return matriz[indices].copy()

    if inicializacion == "uniforme":
        minimos = matriz.min(axis=0)
        maximos = matriz.max(axis=0)
        return generador.uniform(
            minimos,
            maximos,
            size=(cantidad_neuronas, cantidad_variables),
        )

    raise ValueError(f"Inicialización desconocida: {inicializacion}")


def entrenar_som(
    matriz: np.ndarray,
    modelo: dict,
    semilla: int,
) -> ResultadoEntrenamiento:
    """Entrena una red de Kohonen y devuelve pesos y métricas."""
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim != 2:
        raise ValueError("La matriz de entrada debe tener dos dimensiones.")
    if matriz.shape[0] < 2:
        raise ValueError("Se necesitan al menos dos muestras.")

    k = int(modelo["grilla"])
    topologia = str(modelo["topologia"])
    distancia = str(modelo["distancia"])
    inicializacion = str(modelo["inicializacion"])
    iteraciones = int(modelo["iteraciones"])

    radio_inicial = float(modelo["radio_inicial_relativo"]) * k
    radio_final = float(modelo["radio_final"])
    eta_inicial = float(modelo["eta_inicial"])
    eta_final = float(modelo["eta_final"])

    radios = calcular_radios(
        iteraciones,
        radio_inicial,
        radio_final,
        str(modelo["decaimiento_radio"]),
    )
    tasas = calcular_tasas(
        iteraciones,
        eta_inicial,
        eta_final,
        str(modelo["decaimiento_eta"]),
    )
    distancias_grilla_neuronal = distancias_grilla(k, topologia)

    secuencia = np.random.SeedSequence(semilla)
    generador_pesos, generador_orden = [
        np.random.default_rng(semilla_hija) for semilla_hija in secuencia.spawn(2)
    ]

    pesos = _inicializar_pesos(
        matriz,
        k,
        inicializacion,
        generador_pesos,
    )

    cantidad_muestras = matriz.shape[0]
    for paso in range(iteraciones):
        if paso % cantidad_muestras == 0:
            orden = generador_orden.permutation(cantidad_muestras)
        indice = int(orden[paso % cantidad_muestras])
        muestra = matriz[indice]

        bmu = seleccionar_bmu(pesos, muestra, distancia)
        mascara = distancias_grilla_neuronal[bmu] <= radios[paso]
        pesos = actualizar_pesos(pesos, muestra, bmu, mascara, tasas[paso])

    distancias_muestras = distancias_pesos_muestras(pesos, matriz)
    bmu_final = np.argmin(distancias_muestras, axis=1)
    cantidad_neuronas = k * k

    return ResultadoEntrenamiento(
        pesos=pesos,
        bmu=bmu_final,
        distancias_muestras=distancias_muestras,
        qe=calcular_qe(distancias_muestras),
        te=calcular_te(distancias_muestras, distancias_grilla_neuronal),
        unidades_muertas=contar_unidades_muertas(bmu_final, cantidad_neuronas),
        ocupacion=calcular_ocupacion(bmu_final, cantidad_neuronas),
        semilla=semilla,
    )


def _coordenadas_hexagonales(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve coordenadas axiales convertidas a cartesianas para graficar."""
    filas = np.repeat(np.arange(k), k)
    columnas = np.tile(np.arange(k), k)
    q = columnas - filas // 2
    r = filas
    x = q + r / 2.0
    y = r * np.sqrt(3.0) / 2.0
    return x.astype(float), y.astype(float)


def _color_texto_contraste(
    cmap: plt.Colormap,
    norm: Normalize,
    valor: float,
) -> str:
    """Elige texto blanco o negro según la luminancia del fondo."""
    rojo, verde, azul, _ = cmap(norm(valor))
    luminancia = 0.299 * rojo + 0.587 * verde + 0.114 * azul
    return "white" if luminancia < 0.5 else "black"


def _dibujar_mapa(
    eje: plt.Axes,
    valores: np.ndarray,
    k: int,
    topologia: str,
    cmap: plt.Colormap,
    norm: Normalize,
    anotaciones: Sequence[str] | None = None,
    fuente: float = 8.0,
) -> None:
    """Dibuja una grilla rectangular o hexagonal con un valor por neurona."""
    valores = np.asarray(valores, dtype=float)

    if topologia == "rectangular":
        matriz = valores.reshape(k, k)
        eje.imshow(matriz, cmap=cmap, norm=norm, origin="upper")
        eje.set_xticks([])
        eje.set_yticks([])
        if anotaciones is not None:
            for indice, texto in enumerate(anotaciones):
                if not texto:
                    continue
                fila, columna = divmod(indice, k)
                eje.text(
                    columna,
                    fila,
                    texto,
                    ha="center",
                    va="center",
                    fontsize=fuente,
                    color=_color_texto_contraste(cmap, norm, valores[indice]),
                )
        return

    x, y = _coordenadas_hexagonales(k)
    for indice, valor in enumerate(valores):
        hexagono = RegularPolygon(
            (x[indice], y[indice]),
            numVertices=6,
            radius=0.52,
            orientation=np.pi / 6.0,
            facecolor=cmap(norm(valor)),
            edgecolor="white",
            linewidth=0.8,
        )
        eje.add_patch(hexagono)
        if anotaciones is not None and anotaciones[indice]:
            eje.text(
                x[indice],
                y[indice],
                anotaciones[indice],
                ha="center",
                va="center",
                fontsize=fuente,
                color=_color_texto_contraste(cmap, norm, valor),
            )
    eje.set_aspect("equal")
    eje.autoscale_view()
    eje.axis("off")


def _graficar_neuronas(
    pesos: np.ndarray,
    columnas: Sequence[str],
    k: int,
    topologia: str,
    ruta: Path,
) -> None:
    """Genera los planos de componentes del mapa final."""
    cantidad = len(columnas)
    columnas_grilla = 5
    filas_grilla = int(np.ceil(cantidad / columnas_grilla))
    figura, ejes = plt.subplots(
        filas_grilla,
        columnas_grilla,
        figsize=(20, 3.4 * filas_grilla),
        constrained_layout=True,
    )

    maximo = float(np.max(np.abs(pesos))) if pesos.size else 1.0
    maximo = max(maximo, 1e-9)
    norm = Normalize(vmin=-maximo, vmax=maximo)
    cmap = plt.get_cmap("coolwarm")

    for indice, (eje, nombre) in enumerate(zip(ejes.flat, columnas)):
        _dibujar_mapa(
            eje,
            pesos[:, indice],
            k,
            topologia,
            cmap,
            norm,
        )
        eje.set_title(nombre, fontsize=7)

    for eje in list(ejes.flat)[cantidad:]:
        eje.axis("off")

    figura.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ejes.ravel().tolist(),
        shrink=0.6,
        label="Peso",
    )
    figura.suptitle(
        "Pesos de las neuronas por variable (mapa de Kohonen)",
        fontsize=14,
    )
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _graficar_distribucion(
    conteos: np.ndarray,
    asignaciones: pd.DataFrame,
    k: int,
    topologia: str,
    ruta: Path,
) -> None:
    """Genera el mapa de conteos y el listado de creadores por neurona."""
    figura, (eje_mapa, eje_lista) = plt.subplots(
        1,
        2,
        figsize=(17, 8),
        gridspec_kw={"width_ratios": [1, 1.2]},
    )

    cmap = plt.get_cmap("Blues")
    norm = Normalize(vmin=0.0, vmax=max(1.0, float(np.max(conteos))))
    anotaciones = [str(int(conteo)) if conteo > 0 else "" for conteo in conteos]
    _dibujar_mapa(
        eje_mapa,
        conteos,
        k,
        topologia,
        cmap,
        norm,
        anotaciones,
        fuente=10,
    )
    eje_mapa.set_title("Creadores asignados por neurona", fontsize=12)
    figura.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=eje_mapa,
        fraction=0.046,
        pad=0.04,
        label="Cantidad de creadores",
    )

    eje_lista.axis("off")
    lineas: list[str] = []
    for indice in range(k * k):
        if conteos[indice] == 0:
            continue
        fila, columna = divmod(indice, k)
        nombres = asignaciones.loc[asignaciones["neurona"] == indice, "name"].tolist()
        texto = f"({fila},{columna}) n={int(conteos[indice])}: " + ", ".join(nombres)
        lineas.append(textwrap.fill(texto, width=62, subsequent_indent="    "))

    eje_lista.text(
        0.0,
        1.0,
        "\n".join(lineas) if lineas else "Sin creadores asignados",
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        transform=eje_lista.transAxes,
    )
    eje_lista.set_title("Detalle de creadores por neurona", fontsize=12)
    figura.suptitle(
        "Distribución de creadores en la grilla de Kohonen",
        fontsize=14,
    )
    figura.tight_layout(rect=(0, 0, 1, 0.96))
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _graficar_matriz_u(
    matriz: np.ndarray,
    conteos: np.ndarray,
    k: int,
    topologia: str,
    ruta: Path,
) -> None:
    """Genera el mapa de la matriz U con los conteos por neurona."""
    figura, eje = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("magma")
    valores = matriz.ravel()
    minimo = float(np.nanmin(valores)) if np.isfinite(valores).any() else 0.0
    maximo = float(np.nanmax(valores)) if np.isfinite(valores).any() else 1.0
    if np.isclose(minimo, maximo):
        maximo = minimo + 1e-9
    norm = Normalize(vmin=minimo, vmax=maximo)

    anotaciones = []
    for indice, valor in enumerate(valores):
        texto = f"{valor:.2f}"
        if conteos[indice] > 0:
            texto += f"\n({int(conteos[indice])})"
        anotaciones.append(texto)

    _dibujar_mapa(
        eje,
        valores,
        k,
        topologia,
        cmap,
        norm,
        anotaciones,
        fuente=8,
    )
    eje.set_title("Matriz U: distancia media entre neuronas vecinas", fontsize=12)
    figura.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=eje,
        fraction=0.046,
        pad=0.04,
        label="Distancia media",
    )
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _guardar_resultados(
    config: dict,
    datos: pd.DataFrame,
    columnas: Sequence[str],
    medias: np.ndarray,
    desvios: np.ndarray,
    resultados: Sequence[ResultadoEntrenamiento],
    representativo: ResultadoEntrenamiento,
    tiempo_total: float,
) -> Path:
    """Guarda tablas, resumen y figuras de la corrida final."""
    ruta_salida = resolver_ruta(config["salida"]["directorio"])
    ruta_salida.mkdir(parents=True, exist_ok=True)

    config_salida = deepcopy(config)
    config_salida["reproducibilidad"]["semilla_representativa"] = int(
        representativo.semilla
    )
    guardar_config(config_salida, ruta_salida / "config_usada.toml")

    normalizacion = pd.DataFrame(
        {
            "variable": list(columnas),
            "media": medias,
            "desvio": desvios,
        }
    )
    normalizacion.to_csv(
        ruta_salida / "normalizacion.csv",
        index=False,
        float_format="%.8f",
    )

    k = int(config["modelo"]["grilla"])
    topologia = str(config["modelo"]["topologia"])
    filas, columnas_grilla = np.divmod(np.arange(k * k), k)
    pesos = pd.DataFrame(
        {
            "neurona": np.arange(k * k),
            "fila": filas,
            "columna": columnas_grilla,
        }
    )
    for indice, columna in enumerate(columnas):
        pesos[columna] = representativo.pesos[:, indice]
    pesos.to_csv(ruta_salida / "pesos.csv", index=False, float_format="%.8f")

    conteos = np.bincount(representativo.bmu, minlength=k * k)
    qe_local = np.full(k * k, np.nan, dtype=float)
    for indice in range(k * k):
        mascara = representativo.bmu == indice
        if np.any(mascara):
            qe_local[indice] = float(
                np.mean(representativo.distancias_muestras[mascara, indice])
            )

    matriz = matriz_u(representativo.pesos, k, topologia)
    neuronas = pd.DataFrame(
        {
            "neurona": np.arange(k * k),
            "fila": filas,
            "columna": columnas_grilla,
            "asignados": conteos,
            "qe_local": qe_local,
            "u": matriz.ravel(),
        }
    )
    neuronas.to_csv(
        ruta_salida / "neuronas.csv",
        index=False,
        float_format="%.8f",
    )

    distancias_bmu = representativo.distancias_muestras[
        np.arange(len(datos)),
        representativo.bmu,
    ]
    asignaciones = pd.DataFrame(
        {
            "creator_id": datos["creator_id"].to_numpy(),
            "name": datos["name"].to_numpy(),
            "neurona": representativo.bmu,
            "fila": filas[representativo.bmu],
            "columna": columnas_grilla[representativo.bmu],
            "distancia_bmu": distancias_bmu,
        }
    )
    asignaciones.to_csv(
        ruta_salida / "asignaciones.csv",
        index=False,
        float_format="%.8f",
    )

    pd.DataFrame(matriz).to_csv(
        ruta_salida / "matriz_u.csv",
        index_label="fila",
        float_format="%.8f",
    )

    metricas = pd.DataFrame(
        [
            {
                "semilla": resultado.semilla,
                "qe": resultado.qe,
                "te": resultado.te,
                "unidades_muertas": resultado.unidades_muertas,
                "ocupacion": resultado.ocupacion,
                "iteraciones": int(config["modelo"]["iteraciones"]),
            }
            for resultado in resultados
        ]
    )
    metricas.to_csv(
        ruta_salida / "metricas.csv",
        index=False,
        float_format="%.8f",
    )

    qes = np.array([resultado.qe for resultado in resultados], dtype=float)
    lineas = [
        "Resumen de ejecución — TP4, ejercicio 1.2 (Red de Kohonen)",
        f"Datos: {resolver_ruta(config['datos']['ruta'])}",
        f"Variables de entrada: {len(columnas)}",
        "Estandarización: "
        f"{'sí' if config['datos'].get('estandarizar', True) else 'no'}",
        f"Grilla: {k} x {k}",
        f"Topología: {topologia}",
        f"Distancia: {config['modelo']['distancia']}",
        f"Inicialización: {config['modelo']['inicializacion']}",
        f"Iteraciones: {config['modelo']['iteraciones']}",
        f"Semillas: {config['reproducibilidad']['semillas']}",
        f"Semilla representativa: {representativo.semilla}",
        f"QE media: {float(np.mean(qes)):.8f}",
        f"QE desvío: {float(np.std(qes, ddof=1)) if len(qes) > 1 else 0.0:.8f}",
        f"TE representativo: {representativo.te:.8f}",
        f"Unidades muertas representativas: {representativo.unidades_muertas}",
        f"Ocupación representativa: {representativo.ocupacion:.8f}",
        f"Tiempo total de entrenamiento: {tiempo_total:.3f} s",
        "",
        "Archivos generados:",
        "  config_usada.toml",
        "  normalizacion.csv",
        "  pesos.csv",
        "  neuronas.csv",
        "  asignaciones.csv",
        "  matriz_u.csv",
        "  metricas.csv",
        "  fig_neuronas.png",
        "  fig_distribucion.png",
        "  fig_matriz_u.png",
    ]
    (ruta_salida / "resumen.txt").write_text(
        "\n".join(lineas) + "\n",
        encoding="utf-8",
    )

    _graficar_neuronas(
        representativo.pesos,
        columnas,
        k,
        topologia,
        ruta_salida / "fig_neuronas.png",
    )
    _graficar_distribucion(
        conteos,
        asignaciones,
        k,
        topologia,
        ruta_salida / "fig_distribucion.png",
    )
    _graficar_matriz_u(
        matriz,
        conteos,
        k,
        topologia,
        ruta_salida / "fig_matriz_u.png",
    )

    return ruta_salida


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    config = cargar_config(args.config)
    config = resolver_config(config, args)

    datos = cargar_datos(resolver_ruta(config["datos"]["ruta"]))
    columnas = columnas_entrada(datos, config["datos"]["columnas_excluidas"])
    matriz = datos[columnas].to_numpy(dtype=float)
    entrada, medias, desvios = preparar_entrada(
        matriz, config["datos"].get("estandarizar", True)
    )

    inicio = time.perf_counter()
    resultados: list[ResultadoEntrenamiento] = []
    for semilla in config["reproducibilidad"]["semillas"]:
        resultados.append(entrenar_som(entrada, config["modelo"], int(semilla)))
    tiempo_total = time.perf_counter() - inicio

    qes = np.array([resultado.qe for resultado in resultados], dtype=float)
    indice_representativo = int(np.argmin(np.abs(qes - float(np.mean(qes)))))
    representativo = resultados[indice_representativo]

    ruta_salida = _guardar_resultados(
        config,
        datos,
        columnas,
        medias,
        desvios,
        resultados,
        representativo,
        tiempo_total,
    )

    print("=" * 78)
    print("TP4 — Ejercicio 1.2: Red de Kohonen")
    print("=" * 78)
    print(f"Datos: {resolver_ruta(config['datos']['ruta'])}")
    print(f"Variables: {len(columnas)}")
    print(f"Grilla: {config['modelo']['grilla']} x {config['modelo']['grilla']}")
    print(f"Semillas: {config['reproducibilidad']['semillas']}")
    print(f"Semilla representativa: {representativo.semilla}")
    print(f"QE media: {float(np.mean(qes)):.8f}")
    print(f"Resultados: {ruta_salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
