#!/usr/bin/env python3
"""EDA y PCA del catálogo de creadores de contenido (TP4, ejercicio 1.1).

El script lee ``datos/creadores.csv``, calcula estadísticos descriptivos,
correlaciones y un Análisis de Componentes Principales, y guarda tablas y
figuras en ``resultados/ej1/``. No modifica los datos de entrada.

Referencia teórica: ``artif-eng.pdf``, capítulo 13, pp. 115-119.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"
RUTA_RESULTADOS = RAIZ / "resultados" / "ej1"

COLUMNAS_ID = ["creator_id", "name"]

COLUMNAS_TOPICO = [
    "topic_football_pct",
    "topic_makeup_pct",
    "topic_dogs_pct",
    "topic_art_pct",
]

COLUMNAS_CONTENIDO = [
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "content_showcase_pct",
]

COLUMNAS_ENGAGEMENT = [
    "likes_per_1000",
    "comments_per_1000",
    "saves_per_1000",
    "shares_per_1000",
]

# Orden pensado para leer la matriz de correlación por bloques.
COLUMNAS_ORDEN_EDA = [
    "followers",
    "average_reach",
    "average_impressions",
    "average_video_views",
    "estimated_argentina_reach",
    "price_usd",
    "price_ars",
    "likes_per_1000",
    "comments_per_1000",
    "saves_per_1000",
    "shares_per_1000",
    "total_engagement_per_1000",
    "profile_visits_per_1000",
    "average_watch_time_seconds",
    "argentina_audience_pct",
    "non_follower_reach_pct",
    "sponsored_posts_pct",
    "organic_posts_pct",
    "topic_football_pct",
    "topic_makeup_pct",
    "topic_dogs_pct",
    "topic_art_pct",
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "content_showcase_pct",
    "monthly_posts",
    "average_caption_words",
    "video_posts_pct",
    "short_form_posts_pct",
    "trend_posts_pct",
]

UMBRALES_REFERENCIA = [0.80, 0.85, 0.90, 0.95, 0.99]


@dataclass
class ResultadoPCA:
    """Resultado completo del PCA sobre variables estandarizadas."""

    columnas: list[str]
    umbral: float
    medias: np.ndarray
    desvios: np.ndarray
    Z: np.ndarray
    S: np.ndarray
    autovalores: np.ndarray
    autovectores: np.ndarray
    autovalores_pos: np.ndarray
    autovectores_pos: np.ndarray
    scores: np.ndarray
    ratios: np.ndarray
    acumulada: np.ndarray
    q: int
    tolerancia: float


def fraccion(texto: str) -> float:
    """Convierte el argumento de varianza a una fracción en (0, 1]."""
    try:
        valor = float(texto)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La varianza debe ser un número entre 0 y 1, por ejemplo 0.95."
        ) from error

    if not 0.0 < valor <= 1.0:
        raise argparse.ArgumentTypeError(
            "La varianza debe estar en el intervalo (0, 1], por ejemplo 0.95."
        )
    return valor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "EDA y PCA del catálogo de creadores de contenido. "
            "La varianza se indica como fracción: 0.95 = 95%."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--varianza",
        type=fraccion,
        default=0.95,
        help="Fracción mínima de varianza total a conservar.",
    )
    return parser.parse_args(argv)


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV de creadores."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")
    return pd.read_csv(ruta)


def validar_datos(datos: pd.DataFrame) -> None:
    """Valida las condiciones mínimas del dataset."""
    if datos.empty:
        raise ValueError("El dataset está vacío.")

    faltantes = [columna for columna in COLUMNAS_ID if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")

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


def detectar_redundancias(
    datos: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Detecta variables duplicadas, derivadas, sumas y complementos exactos.

    Devuelve una tabla explicativa y la lista de columnas que deben excluirse
    del PCA. Los grupos cerrados (tópicos y contenido) se informan, pero no se
    excluyen: se conservan completos para interpretar el problema.
    """
    filas: list[dict[str, str]] = []
    excluidas: list[str] = []

    def agregar(
        variable: str,
        tipo: str,
        accion: str,
        motivo: str,
        evidencia: str,
    ) -> None:
        filas.append(
            {
                "variable": variable,
                "tipo": tipo,
                "accion": accion,
                "motivo": motivo,
                "evidencia": evidencia,
            }
        )

    if {"price_usd", "price_ars"}.issubset(datos.columns):
        ratio = datos["price_ars"] / datos["price_usd"]
        if np.allclose(ratio.to_numpy(dtype=float), 1500.0, rtol=1e-9, atol=1e-6):
            agregar(
                "price_ars",
                "precio",
                "excluir",
                "Duplicado exacto de price_usd",
                "price_ars / price_usd = 1500 en todas las filas",
            )
            excluidas.append("price_ars")

    if {
        "average_reach",
        "argentina_audience_pct",
        "estimated_argentina_reach",
    }.issubset(datos.columns):
        esperado = (
            datos["average_reach"] * datos["argentina_audience_pct"] / 100.0
        )
        diferencia = (datos["estimated_argentina_reach"] - esperado).abs().max()
        if diferencia <= 1.0:
            agregar(
                "estimated_argentina_reach",
                "derivada",
                "excluir",
                "Se calcula a partir de average_reach y argentina_audience_pct",
                f"Diferencia máxima por redondeo: {diferencia:.2f} cuentas",
            )
            excluidas.append("estimated_argentina_reach")

    if set(COLUMNAS_ENGAGEMENT + ["total_engagement_per_1000"]).issubset(
        datos.columns
    ):
        suma = datos[COLUMNAS_ENGAGEMENT].sum(axis=1)
        diferencia = (datos["total_engagement_per_1000"] - suma).abs().max()
        if diferencia <= 1e-9:
            agregar(
                "total_engagement_per_1000",
                "suma",
                "excluir",
                "Es la suma de likes, comentarios, guardados y compartidos",
                f"Diferencia máxima: {diferencia:.2e}",
            )
            excluidas.append("total_engagement_per_1000")

    if {"sponsored_posts_pct", "organic_posts_pct"}.issubset(datos.columns):
        suma = datos["sponsored_posts_pct"] + datos["organic_posts_pct"]
        if np.allclose(suma.to_numpy(dtype=float), 100.0):
            agregar(
                "organic_posts_pct",
                "complemento",
                "excluir",
                "Es el complemento exacto de sponsored_posts_pct",
                "sponsored_posts_pct + organic_posts_pct = 100 en todas las filas",
            )
            excluidas.append("organic_posts_pct")

    for grupo, nombre in [
        (COLUMNAS_TOPICO, "tópicos"),
        (COLUMNAS_CONTENIDO, "contenido"),
    ]:
        if set(grupo).issubset(datos.columns):
            suma = datos[grupo].sum(axis=1)
            if np.allclose(suma.to_numpy(dtype=float), 100.0):
                agregar(
                    f"{nombre}_*",
                    "grupo cerrado",
                    "conservar",
                    f"Las {len(grupo)} categorías de {nombre} suman 100",
                    "Dependencia lineal: la matriz de correlación pierde un rango",
                )

    return pd.DataFrame(filas), list(dict.fromkeys(excluidas))


def columnas_para_pca(datos: pd.DataFrame) -> list[str]:
    """Devuelve las variables numéricas que entran al PCA."""
    _, excluidas = detectar_redundancias(datos)
    numericas = datos.select_dtypes(include=np.number).columns.tolist()
    return [columna for columna in numericas if columna not in excluidas]


def resumen_descriptivo(
    datos: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """Calcula estadísticos descriptivos y cantidad de outliers por variable."""
    valores = datos[columnas]
    q1 = valores.quantile(0.25)
    q3 = valores.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = ((valores < limite_inferior) | (valores > limite_superior)).sum()

    return pd.DataFrame(
        {
            "n": valores.count(),
            "media": valores.mean(),
            "desvio": valores.std(ddof=1),
            "minimo": valores.min(),
            "q1": q1,
            "mediana": valores.median(),
            "q3": q3,
            "maximo": valores.max(),
            "asimetria": valores.skew(),
            "curtosis": valores.kurt(),
            "n_outliers": outliers,
        }
    )


def tabla_outliers(
    datos: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """Lista los creadores fuera del rango [Q1 - 1.5 IQR, Q3 + 1.5 IQR]."""
    filas: list[dict[str, str | int]] = []

    for columna in columnas:
        q1 = datos[columna].quantile(0.25)
        q3 = datos[columna].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        mascara = (datos[columna] < limite_inferior) | (
            datos[columna] > limite_superior
        )
        if mascara.any():
            filas.append(
                {
                    "variable": columna,
                    "n_outliers": int(mascara.sum()),
                    "creator_ids": ", ".join(
                        datos.loc[mascara, "creator_id"].astype(str)
                    ),
                }
            )

    if not filas:
        return pd.DataFrame(columns=["variable", "n_outliers", "creator_ids"])
    return pd.DataFrame(filas)


def matriz_correlaciones(
    datos: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """Calcula la matriz de correlación de Pearson."""
    return datos[columnas].corr(method="pearson")


def pares_correlacionados(
    correlaciones: pd.DataFrame, cantidad: int = 15
) -> pd.DataFrame:
    """Devuelve los pares de variables con mayor correlación absoluta."""
    filas: list[dict[str, str | float]] = []
    columnas = correlaciones.columns.tolist()

    for indice, variable_1 in enumerate(columnas):
        for variable_2 in columnas[indice + 1 :]:
            filas.append(
                {
                    "variable_1": variable_1,
                    "variable_2": variable_2,
                    "correlacion": float(
                        correlaciones.loc[variable_1, variable_2]
                    ),
                }
            )

    tabla = pd.DataFrame(filas)
    tabla["_abs"] = tabla["correlacion"].abs()
    tabla = tabla.sort_values("_abs", ascending=False).head(cantidad)
    return tabla.drop(columns="_abs").reset_index(drop=True)


def correlaciones_objetivo(
    datos: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """Correlaciona cada variable con alcance, engagement y precio."""
    objetivos = [
        columna
        for columna in ["followers", "total_engagement_per_1000", "price_usd"]
        if columna in datos.columns
    ]
    if not objetivos:
        return pd.DataFrame(index=columnas)

    columnas_unicas = list(dict.fromkeys(columnas + objetivos))
    correlaciones = datos[columnas_unicas].corr(method="pearson")
    return pd.DataFrame(
        {
            f"r_{objetivo}": correlaciones.loc[columnas, objetivo]
            for objetivo in objetivos
        }
    )


def estandarizar(datos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estandariza cada columna con media 0 y desvío muestral 1."""
    matriz = np.asarray(datos, dtype=float)
    if matriz.ndim != 2:
        raise ValueError("La matriz de datos debe tener dos dimensiones.")

    medias = matriz.mean(axis=0)
    desvios = matriz.std(axis=0, ddof=1)
    if np.any(desvios == 0):
        raise ValueError("No se puede estandarizar una variable constante.")

    return (matriz - medias) / desvios, medias, desvios


def descomponer_matriz(
    matriz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene autovalores y autovectores de una matriz simétrica."""
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    if not np.allclose(matriz, matriz.T):
        raise ValueError("La matriz debe ser simétrica.")

    autovalores, autovectores = np.linalg.eigh(matriz)
    orden = np.argsort(autovalores)[::-1]
    return autovalores[orden], autovectores[:, orden]


def seleccionar_q(autovalores: np.ndarray, umbral: float) -> int:
    """Devuelve la menor cantidad de componentes que alcanza el umbral."""
    if not 0.0 < umbral <= 1.0:
        raise ValueError("El umbral debe estar en (0, 1].")

    valores = np.asarray(autovalores, dtype=float)
    positivos = valores[valores > 0]
    if positivos.size == 0:
        raise ValueError("No hay autovalores positivos.")

    ratios = positivos / positivos.sum()
    q = int(np.searchsorted(np.cumsum(ratios), umbral, side="left") + 1)
    return min(q, positivos.size)


def orientar_cargas(
    autovectores: np.ndarray, columnas: list[str]
) -> np.ndarray:
    """Fija el signo de cada componente para que los gráficos sean estables."""
    cargas = autovectores.copy()

    for componente in range(cargas.shape[1]):
        if componente == 0 and "followers" in columnas:
            indice = columnas.index("followers")
        else:
            indice = int(np.argmax(np.abs(cargas[:, componente])))

        if cargas[indice, componente] < 0:
            cargas[:, componente] *= -1

    return cargas


def ajustar_pca(
    datos: pd.DataFrame, columnas: list[str], umbral: float
) -> ResultadoPCA:
    """Ajusta el PCA sobre la matriz de correlación."""
    matriz = datos[columnas].to_numpy(dtype=float)
    estandarizados, medias, desvios = estandarizar(matriz)

    n = estandarizados.shape[0]
    if n < 2:
        raise ValueError("Se necesitan al menos dos observaciones.")

    covarianza = (estandarizados.T @ estandarizados) / (n - 1)
    autovalores, autovectores = descomponer_matriz(covarianza)

    tolerancia = (
        max(covarianza.shape)
        * np.finfo(float).eps
        * float(np.max(np.abs(autovalores)))
    )
    mascara = autovalores > tolerancia
    autovalores_pos = autovalores[mascara]
    autovectores_pos = autovectores[:, mascara]

    if autovalores_pos.size == 0:
        raise ValueError("Todos los autovalores son nulos.")

    ratios = autovalores_pos / autovalores_pos.sum()
    acumulada = np.cumsum(ratios)
    q = seleccionar_q(autovalores_pos, umbral)
    autovectores_pos = orientar_cargas(autovectores_pos, columnas)
    scores = estandarizados @ autovectores_pos

    return ResultadoPCA(
        columnas=columnas,
        umbral=umbral,
        medias=medias,
        desvios=desvios,
        Z=estandarizados,
        S=covarianza,
        autovalores=autovalores,
        autovectores=autovectores,
        autovalores_pos=autovalores_pos,
        autovectores_pos=autovectores_pos,
        scores=scores,
        ratios=ratios,
        acumulada=acumulada,
        q=q,
        tolerancia=tolerancia,
    )


def validar_pca(resultado: ResultadoPCA) -> dict[str, float]:
    """Comprueba numéricamente la descomposición y los scores."""
    residuo = float(
        np.max(
            np.abs(
                resultado.S @ resultado.autovectores
                - resultado.autovectores * resultado.autovalores
            )
        )
    )
    varianza_scores = resultado.scores.var(axis=0, ddof=1)
    error_varianza = float(
        np.max(np.abs(varianza_scores - resultado.autovalores_pos))
    )

    return {
        "residuo_autovalores": residuo,
        "error_varianza_scores": error_varianza,
        "suma_ratios": float(resultado.ratios.sum()),
    }


def tabla_autovalores(resultado: ResultadoPCA) -> pd.DataFrame:
    """Tabla con autovalores, proporción de varianza y componentes elegidas."""
    denominador = float(resultado.autovalores_pos.sum())
    componentes: list[dict[str, object]] = []
    acumulado = 0.0

    for indice, autovalor in enumerate(resultado.autovalores):
        es_nula = bool(autovalor <= resultado.tolerancia)
        ratio = 0.0 if es_nula else float(autovalor) / denominador
        acumulado += ratio
        componentes.append(
            {
                "componente": f"PC{indice + 1}",
                "autovalor": float(autovalor),
                "ratio_varianza": ratio,
                "ratio_acumulado": acumulado,
                "nula": es_nula,
                "seleccionada": bool(indice < resultado.q and not es_nula),
            }
        )

    return pd.DataFrame(componentes)


def tabla_cargas(resultado: ResultadoPCA) -> pd.DataFrame:
    """Tabla de cargas de todas las componentes con varianza no nula."""
    nombres = [
        f"PC{indice + 1}"
        for indice in range(resultado.autovalores_pos.size)
    ]
    cargas = pd.DataFrame(
        resultado.autovectores_pos,
        index=resultado.columnas,
        columns=nombres,
    )
    cargas["comunalidad_q"] = np.sum(
        resultado.autovectores_pos[:, : resultado.q] ** 2, axis=1
    )
    return cargas


def tabla_scores(datos: pd.DataFrame, resultado: ResultadoPCA) -> pd.DataFrame:
    """Scores de todos los creadores en las componentes no nulas."""
    nombres = [
        f"PC{indice + 1}"
        for indice in range(resultado.autovalores_pos.size)
    ]
    scores = pd.DataFrame(resultado.scores, columns=nombres)
    return pd.concat(
        [datos[COLUMNAS_ID].reset_index(drop=True), scores],
        axis=1,
    )


def tabla_pcs_negocio(
    datos: pd.DataFrame, resultado: ResultadoPCA
) -> pd.DataFrame:
    """Correlación entre cada componente y variables de negocio."""
    objetivos = [
        columna
        for columna in [
            "followers",
            "total_engagement_per_1000",
            "price_usd",
            "argentina_audience_pct",
            "sponsored_posts_pct",
            "topic_football_pct",
            "topic_makeup_pct",
            "topic_dogs_pct",
            "topic_art_pct",
        ]
        if columna in datos.columns
    ]

    filas: list[dict[str, float | str]] = []
    for componente in range(resultado.autovalores_pos.size):
        fila: dict[str, float | str] = {"componente": f"PC{componente + 1}"}
        for objetivo in objetivos:
            fila[f"r_{objetivo}"] = float(
                np.corrcoef(
                    resultado.scores[:, componente],
                    datos[objetivo].to_numpy(dtype=float),
                )[0, 1]
            )
        filas.append(fila)

    return pd.DataFrame(filas)


def tabla_seleccion(resultado: ResultadoPCA) -> pd.DataFrame:
    """Compara la cantidad de componentes para distintos umbrales."""
    umbrales = sorted(set(UMBRALES_REFERENCIA + [resultado.umbral]))
    filas: list[dict[str, float | int | bool]] = []

    for umbral in umbrales:
        q = seleccionar_q(resultado.autovalores_pos, umbral)
        filas.append(
            {
                "umbral": float(umbral),
                "componentes": q,
                "varianza_acumulada": float(resultado.acumulada[q - 1]),
                "umbral_elegido": bool(np.isclose(umbral, resultado.umbral)),
            }
        )

    return pd.DataFrame(filas)


def guardar_tablas(
    ruta_eda: Path,
    ruta_pca: Path,
    descriptivos: pd.DataFrame,
    correlaciones: pd.DataFrame,
    correlaciones_obj: pd.DataFrame,
    pares: pd.DataFrame,
    redundancias: pd.DataFrame,
    outliers: pd.DataFrame,
    autovalores: pd.DataFrame,
    cargas: pd.DataFrame,
    scores: pd.DataFrame,
    pcs_negocio: pd.DataFrame,
    seleccion: pd.DataFrame,
    resultado: ResultadoPCA,
    checks: dict[str, float],
) -> None:
    """Guarda todas las tablas y el resumen de la corrida."""
    ruta_eda.mkdir(parents=True, exist_ok=True)
    ruta_pca.mkdir(parents=True, exist_ok=True)

    descriptivos.to_csv(
        ruta_eda / "descriptivos.csv", float_format="%.6f"
    )
    correlaciones.to_csv(
        ruta_eda / "correlaciones.csv", float_format="%.6f"
    )
    correlaciones_obj.to_csv(
        ruta_eda / "correlaciones_objetivo.csv", float_format="%.6f"
    )
    pares.to_csv(
        ruta_eda / "pares_correlacion.csv",
        index=False,
        float_format="%.6f",
    )
    redundancias.to_csv(ruta_eda / "redundancias.csv", index=False)
    outliers.to_csv(ruta_eda / "outliers.csv", index=False)

    autovalores.to_csv(
        ruta_pca / "autovalores.csv",
        index=False,
        float_format="%.8f",
    )
    cargas.to_csv(ruta_pca / "cargas.csv", float_format="%.6f")
    scores.to_csv(
        ruta_pca / "scores.csv", index=False, float_format="%.6f"
    )
    pcs_negocio.to_csv(
        ruta_pca / "pcs_vs_negocio.csv",
        index=False,
        float_format="%.6f",
    )
    seleccion.to_csv(
        ruta_pca / "seleccion_componentes.csv",
        index=False,
        float_format="%.6f",
    )

    autovalores_nulos = int(
        np.sum(resultado.autovalores <= resultado.tolerancia)
    )
    lineas = [
        "Resumen de ejecución — TP4, ejercicio 1.1 (EDA + PCA)",
        f"Datos: {RUTA_DATOS}",
        f"Umbral de varianza: {resultado.umbral:.2%}",
        f"Variables en el EDA: {len(COLUMNAS_ORDEN_EDA)}",
        f"Variables en el PCA: {len(resultado.columnas)}",
        f"Rango de la matriz de correlación: {resultado.autovalores_pos.size}",
        f"Autovalores nulos: {autovalores_nulos}",
        f"Componentes seleccionados: {resultado.q}",
        (
            "Varianza acumulada: "
            f"{resultado.acumulada[resultado.q - 1]:.4%}"
        ),
        "",
        "Chequeos numéricos:",
        f"  Residuo máximo S·v - λ·v: {checks['residuo_autovalores']:.3e}",
        (
            "  Error máximo de varianza de scores: "
            f"{checks['error_varianza_scores']:.3e}"
        ),
        f"  Suma de ratios de varianza: {checks['suma_ratios']:.12f}",
        "",
        "Archivos generados:",
        "  eda/descriptivos.csv",
        "  eda/correlaciones.csv",
        "  eda/correlaciones_objetivo.csv",
        "  eda/pares_correlacion.csv",
        "  eda/redundancias.csv",
        "  eda/outliers.csv",
        "  eda/fig_distribuciones.png",
        "  eda/fig_correlaciones.png",
        f"  {ruta_pca.name}/autovalores.csv",
        f"  {ruta_pca.name}/cargas.csv",
        f"  {ruta_pca.name}/scores.csv",
        f"  {ruta_pca.name}/pcs_vs_negocio.csv",
        f"  {ruta_pca.name}/seleccion_componentes.csv",
        f"  {ruta_pca.name}/fig_scree.png",
        f"  {ruta_pca.name}/fig_cargas.png",
        f"  {ruta_pca.name}/fig_biplot.png",
    ]
    (ruta_pca / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def graficar_eda(
    datos: pd.DataFrame,
    columnas: list[str],
    correlaciones: pd.DataFrame,
    ruta_eda: Path,
) -> None:
    """Genera los gráficos del EDA."""
    ruta_eda.mkdir(parents=True, exist_ok=True)

    cantidad = len(columnas)
    columnas_grilla = 5
    filas_grilla = int(np.ceil(cantidad / columnas_grilla))
    figura, ejes = plt.subplots(
        filas_grilla,
        columnas_grilla,
        figsize=(20, 3.1 * filas_grilla),
    )

    for eje, columna in zip(ejes.flat, columnas):
        eje.hist(
            datos[columna],
            bins=15,
            color="#4C72B0",
            edgecolor="white",
            alpha=0.9,
        )
        eje.set_title(columna, fontsize=8)
        eje.tick_params(labelsize=7)

    for eje in list(ejes.flat)[cantidad:]:
        eje.axis("off")

    figura.suptitle(
        "Distribuciones de las variables numéricas",
        fontsize=14,
    )
    figura.tight_layout(rect=(0, 0, 1, 0.97))
    figura.savefig(
        ruta_eda / "fig_distribuciones.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)

    orden = columnas
    matriz = correlaciones.loc[orden, orden]
    figura, eje = plt.subplots(figsize=(16, 14))
    imagen = eje.imshow(
        matriz.to_numpy(),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )

    eje.set_xticks(np.arange(len(orden)))
    eje.set_yticks(np.arange(len(orden)))
    eje.set_xticklabels(orden, rotation=90, fontsize=7)
    eje.set_yticklabels(orden, fontsize=7)
    eje.set_title("Matriz de correlación de Pearson", fontsize=14)

    for limite in [7, 14, 16, 18, 22, 26]:
        eje.axhline(limite - 0.5, color="white", linewidth=1.2)
        eje.axvline(limite - 0.5, color="white", linewidth=1.2)

    figura.colorbar(
        imagen,
        ax=eje,
        fraction=0.046,
        pad=0.04,
        label="Correlación de Pearson",
    )
    figura.tight_layout()
    figura.savefig(
        ruta_eda / "fig_correlaciones.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)


def graficar_pca(
    datos: pd.DataFrame, resultado: ResultadoPCA, ruta_pca: Path
) -> None:
    """Genera scree plot, cargas y biplot del PCA."""
    ruta_pca.mkdir(parents=True, exist_ok=True)

    componentes = np.arange(1, resultado.autovalores_pos.size + 1)
    figura, eje = plt.subplots(figsize=(11, 6))
    eje.bar(
        componentes,
        resultado.autovalores_pos,
        color="#4C72B0",
        alpha=0.85,
        label="Autovalor",
    )
    eje.set_xlabel("Componente principal")
    eje.set_ylabel("Autovalor")
    eje.set_title(
        "Varianza explicada por componente",
        fontsize=13,
    )

    eje_varianza = eje.twinx()
    eje_varianza.plot(
        componentes,
        resultado.acumulada,
        color="#C44E52",
        marker="o",
        markersize=3,
        label="Varianza acumulada",
    )
    eje_varianza.axhline(
        resultado.umbral,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Umbral {resultado.umbral:.0%}",
    )
    eje_varianza.axvline(
        resultado.q,
        color="#2E7D32",
        linestyle=":",
        linewidth=1.5,
        label=f"q = {resultado.q}",
    )
    eje_varianza.set_ylim(0.0, 1.02)
    eje_varianza.set_ylabel("Varianza acumulada")
    eje_varianza.annotate(
        f"q = {resultado.q}\n{resultado.acumulada[resultado.q - 1]:.1%}",
        xy=(resultado.q, resultado.acumulada[resultado.q - 1]),
        xytext=(resultado.q + 1.5, max(0.1, resultado.umbral - 0.18)),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#2E7D32"},
    )

    lineas_1, etiquetas_1 = eje.get_legend_handles_labels()
    lineas_2, etiquetas_2 = eje_varianza.get_legend_handles_labels()
    eje.legend(
        lineas_1 + lineas_2,
        etiquetas_1 + etiquetas_2,
        loc="center right",
        fontsize=8,
    )
    figura.tight_layout()
    figura.savefig(
        ruta_pca / "fig_scree.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)

    cantidad_pc = min(3, resultado.autovalores_pos.size)
    figura, ejes = plt.subplots(
        1,
        cantidad_pc,
        figsize=(6 * cantidad_pc, 8),
        squeeze=False,
    )

    for componente in range(cantidad_pc):
        eje = ejes[0, componente]
        serie = pd.Series(
            resultado.autovectores_pos[:, componente],
            index=resultado.columnas,
        )
        elegidas = serie.abs().sort_values(ascending=False).head(12).index
        serie = serie.loc[elegidas].sort_values()
        colores = [
            "#C44E52" if valor < 0 else "#4C72B0"
            for valor in serie.to_numpy()
        ]
        eje.barh(serie.index, serie.to_numpy(), color=colores, alpha=0.9)
        eje.axvline(0.0, color="black", linewidth=0.8)
        eje.set_title(
            (
                f"PC{componente + 1} "
                f"({resultado.ratios[componente]:.1%} de varianza)"
            ),
            fontsize=11,
        )
        eje.tick_params(axis="y", labelsize=8)
        eje.tick_params(axis="x", labelsize=8)
        eje.set_xlabel("Carga")

    figura.suptitle(
        "Cargas de las tres primeras componentes",
        fontsize=14,
    )
    figura.tight_layout(rect=(0, 0, 1, 0.96))
    figura.savefig(
        ruta_pca / "fig_cargas.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)

    if resultado.autovalores_pos.size < 2:
        return

    figura, eje = plt.subplots(figsize=(13, 10))
    etiquetas_topico = {
        "topic_football_pct": "Fútbol",
        "topic_makeup_pct": "Maquillaje",
        "topic_dogs_pct": "Perros",
        "topic_art_pct": "Arte",
    }
    colores_topico = {
        "topic_football_pct": "#1f77b4",
        "topic_makeup_pct": "#e377c2",
        "topic_dogs_pct": "#2ca02c",
        "topic_art_pct": "#ff7f0e",
    }
    topicos = [
        topico for topico in etiquetas_topico if topico in datos.columns
    ]

    # Primero las flechas, para que los puntos queden por encima.
    cargas_2d = resultado.autovectores_pos[:, :2]
    normas = np.linalg.norm(cargas_2d, axis=1)
    maximas = np.argsort(normas)[-8:]
    maximo_score = float(np.max(np.abs(resultado.scores[:, :2])))
    maximo_norma = float(np.max(normas[maximas])) if maximas.size else 0.0
    if maximo_norma > 0.0:
        escala = 0.8 * maximo_score / maximo_norma
        for indice in maximas:
            avance_x = cargas_2d[indice, 0] * escala
            avance_y = cargas_2d[indice, 1] * escala
            eje.arrow(
                0.0,
                0.0,
                avance_x,
                avance_y,
                color="gray",
                width=0.0015,
                head_width=0.08,
                alpha=0.6,
                length_includes_head=True,
                zorder=1,
            )
            eje.text(
                avance_x * 1.08,
                avance_y * 1.08,
                resultado.columnas[indice],
                fontsize=7,
                color="dimgray",
                ha="left" if avance_x >= 0 else "right",
                va="bottom" if avance_y >= 0 else "top",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                    "pad": 0.8,
                },
                zorder=2,
            )

    if topicos:
        dominante = datos[topicos].idxmax(axis=1)
        for topico in topicos:
            mascara = (dominante == topico).to_numpy()
            eje.scatter(
                resultado.scores[mascara, 0],
                resultado.scores[mascara, 1],
                s=50,
                color=colores_topico[topico],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
                label=etiquetas_topico[topico],
                zorder=3,
            )
    else:
        eje.scatter(
            resultado.scores[:, 0],
            resultado.scores[:, 1],
            s=50,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )

    for indice, nombre in enumerate(datos["name"].astype(str)):
        eje.annotate(
            nombre,
            (resultado.scores[indice, 0], resultado.scores[indice, 1]),
            fontsize=5.5,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
            zorder=4,
        )

    eje.margins(0.08)
    eje.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    eje.axvline(0.0, color="black", linewidth=0.6, linestyle="--")
    eje.set_xlabel(f"PC1 ({resultado.ratios[0]:.1%} de varianza)")
    eje.set_ylabel(f"PC2 ({resultado.ratios[1]:.1%} de varianza)")
    eje.set_title(
        "Biplot PC1-PC2: creadores y cargas de variables",
        fontsize=13,
    )
    if topicos:
        eje.legend(title="Tópico dominante", fontsize=8, title_fontsize=9)
    eje.text(
        0.01,
        0.01,
        "Flechas escaladas para visualización",
        transform=eje.transAxes,
        fontsize=7,
        color="dimgray",
    )
    figura.tight_layout()
    figura.savefig(
        ruta_pca / "fig_biplot.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)


def imprimir_reporte(
    datos: pd.DataFrame,
    descriptivos: pd.DataFrame,
    redundancias: pd.DataFrame,
    pares: pd.DataFrame,
    correlaciones_obj: pd.DataFrame,
    autovalores: pd.DataFrame,
    resultado: ResultadoPCA,
    pcs_negocio: pd.DataFrame,
    seleccion: pd.DataFrame,
    checks: dict[str, float],
    ruta_eda: Path,
    ruta_pca: Path,
) -> None:
    """Imprime un resumen legible de la corrida."""

    def titulo(texto: str) -> None:
        print("\n" + "=" * 78)
        print(texto)
        print("=" * 78)

    titulo("TP4 — Ejercicio 1.1: Exploración y representación (EDA + PCA)")
    print("Referencia teórica: artif-eng.pdf, capítulo 13, pp. 115-119")
    print(f"Datos: {RUTA_DATOS}")
    print(f"Filas: {len(datos)}")
    print(f"Columnas numéricas: {len(COLUMNAS_ORDEN_EDA)}")
    print(f"Variables en el PCA: {len(resultado.columnas)}")
    print(f"Umbral de varianza: {resultado.umbral:.2%}")

    titulo("1. Estadísticos descriptivos")
    print(descriptivos.round(3).to_string())

    titulo("2. Variables redundantes y grupos cerrados")
    print(redundancias.to_string(index=False))

    titulo("3. Pares de variables más correlacionados")
    print(pares.head(15).round(3).to_string(index=False))

    titulo("4. Correlaciones con followers, engagement y precio")
    for columna in correlaciones_obj.columns:
        objetivo = columna.removeprefix("r_")
        serie = correlaciones_obj[columna].drop(
            labels=[objetivo], errors="ignore"
        )
        serie = serie.drop(
            labels=["price_ars", "estimated_argentina_reach"],
            errors="ignore",
        )
        serie = serie.reindex(serie.abs().sort_values(ascending=False).index)
        print(f"\n{columna}:")
        print(serie.head(8).round(3).to_string())

    titulo("5. Varianza explicada por componente")
    print(autovalores.head(15).to_string(index=False))
    nulas = autovalores[autovalores["nula"]]
    if not nulas.empty:
        print(
            "\nComponentes con autovalor nulo por dependencia lineal: "
            + ", ".join(nulas["componente"].tolist())
        )
    print(
        f"\nComponentes seleccionados: {resultado.q} "
        f"({resultado.acumulada[resultado.q - 1]:.2%} de varianza)"
    )
    print("\nComparación de umbrales:")
    print(seleccion.to_string(index=False))

    titulo("6. Cargas principales")
    for componente in range(min(3, resultado.autovalores_pos.size)):
        serie = pd.Series(
            resultado.autovectores_pos[:, componente],
            index=resultado.columnas,
        )
        serie = serie.reindex(
            serie.abs().sort_values(ascending=False).index
        )
        print(
            f"\nPC{componente + 1} "
            f"({resultado.ratios[componente]:.2%} de varianza):"
        )
        print(serie.head(8).round(3).to_string())

    titulo("7. Correlación de las componentes con variables de negocio")
    print(pcs_negocio.head(4).round(3).to_string(index=False))

    titulo("8. Chequeos numéricos")
    print(f"Residuo máximo S·v - λ·v: {checks['residuo_autovalores']:.3e}")
    print(
        "Error máximo de varianza de scores: "
        f"{checks['error_varianza_scores']:.3e}"
    )
    print(f"Suma de ratios de varianza: {checks['suma_ratios']:.12f}")

    titulo("9. Archivos generados")
    print(f"EDA: {ruta_eda}")
    print(f"PCA: {ruta_pca}")


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    datos = cargar_datos(RUTA_DATOS)
    validar_datos(datos)

    numericas = datos.select_dtypes(include=np.number).columns.tolist()
    columnas_eda = [
        columna for columna in COLUMNAS_ORDEN_EDA if columna in numericas
    ]
    columnas_eda += [
        columna for columna in numericas if columna not in columnas_eda
    ]

    redundancias, excluidas = detectar_redundancias(datos)
    columnas_pca = [
        columna for columna in numericas if columna not in excluidas
    ]

    descriptivos = resumen_descriptivo(datos, columnas_eda)
    correlaciones = matriz_correlaciones(datos, columnas_eda)
    pares = pares_correlacionados(correlaciones)
    correlaciones_obj = correlaciones_objetivo(datos, columnas_eda)
    outliers = tabla_outliers(datos, columnas_eda)

    resultado = ajustar_pca(datos, columnas_pca, args.varianza)
    checks = validar_pca(resultado)

    autovalores = tabla_autovalores(resultado)
    cargas = tabla_cargas(resultado)
    scores = tabla_scores(datos, resultado)
    pcs_negocio = tabla_pcs_negocio(datos, resultado)
    seleccion = tabla_seleccion(resultado)

    ruta_eda = RUTA_RESULTADOS / "eda"
    ruta_pca = RUTA_RESULTADOS / f"pca_{args.varianza:.2f}"
    ruta_eda.mkdir(parents=True, exist_ok=True)
    ruta_pca.mkdir(parents=True, exist_ok=True)

    guardar_tablas(
        ruta_eda,
        ruta_pca,
        descriptivos,
        correlaciones,
        correlaciones_obj,
        pares,
        redundancias,
        outliers,
        autovalores,
        cargas,
        scores,
        pcs_negocio,
        seleccion,
        resultado,
        checks,
    )

    graficar_eda(datos, columnas_eda, correlaciones, ruta_eda)
    graficar_pca(datos, resultado, ruta_pca)

    imprimir_reporte(
        datos,
        descriptivos,
        redundancias,
        pares,
        correlaciones_obj,
        autovalores,
        resultado,
        pcs_negocio,
        seleccion,
        checks,
        ruta_eda,
        ruta_pca,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
