#!/usr/bin/env python3
"""Compara redes de Kohonen entrenadas con 27 variables y con PC1..PC10.

Script de análisis: solo lee corridas finales de ``kohonen.py`` (con
``pesos_semillas.csv`` y ``asignaciones_semillas.csv``) y no entrena nada.

La QE de cada corrida está en las unidades de su propio espacio de entrada, así
que no se puede comparar directamente. Para compararlas se calcula una QE común
en el espacio de las 27 variables estandarizadas: los pesos de las corridas
sobre PCs se retroproyectan con las cargas de 1.1 (``w = s Eᵀ``) y se mide la
distancia de cada creador a la neurona que le asignó su propio modelo.

Además se mide cuánto coinciden las agrupaciones con el índice de Rand
ajustado (ARI), entre semillas de una misma corrida (estabilidad) y entre
corridas distintas.
"""

from __future__ import annotations

import argparse
import itertools
import tomllib
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"
RUTA_CARGAS = RAIZ / "resultados" / "ej1" / "pca_0.90" / "cargas.csv"
RUTA_SALIDA = RAIZ / "resultados" / "ej1" / "ej1.2-comparacion"

CORRIDAS_POR_DEFECTO = [
    "completa_k3=resultados/ej1/ej1.2-full/kohonen_k3",
    "completa_k4=resultados/ej1/ej1.2-full/kohonen_k4",
    "completa_k5=resultados/ej1/ej1.2-full/kohonen_k5",
    "pca10_k3=resultados/ej1/ej1.2-pca10/kohonen_k3",
    "pca10_k4=resultados/ej1/ej1.2-pca10/kohonen_k4",
    "pca10_k4_euclidiana=resultados/ej1/ej1.2-pca10/kohonen_k4_euclidiana",
    "pca10_k5=resultados/ej1/ej1.2-pca10/kohonen_k5",
]

COLORES = {"27 variables": "#4C72B0", "PC1..PC10": "#DD8452"}


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    if camino.is_absolute():
        return camino
    return RAIZ / camino


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Compara corridas finales de Kohonen en un espacio común.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corrida",
        action="append",
        default=None,
        help="Corrida a comparar como etiqueta=directorio. Se puede repetir.",
    )
    parser.add_argument("--datos", type=Path, default=RUTA_DATOS)
    parser.add_argument("--cargas", type=Path, default=RUTA_CARGAS)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    return parser.parse_args(argv)


def indice_rand_ajustado(a: np.ndarray, b: np.ndarray) -> float:
    """Índice de Rand ajustado (Hubert y Arabie, 1985) entre dos particiones.

    Vale 1 si las particiones son iguales (salvo nombres de grupo), cerca de 0
    si coinciden lo que se espera por azar y puede ser negativo.
    """
    _, a = np.unique(np.asarray(a), return_inverse=True)
    _, b = np.unique(np.asarray(b), return_inverse=True)
    if a.shape != b.shape:
        raise ValueError("Las particiones deben tener el mismo tamaño.")

    contingencia = np.zeros((a.max() + 1, b.max() + 1), dtype=float)
    np.add.at(contingencia, (a, b), 1.0)

    def pares(valores: np.ndarray) -> float:
        return float(np.sum(valores * (valores - 1) / 2))

    indice = pares(contingencia)
    pares_a = pares(contingencia.sum(axis=1))
    pares_b = pares(contingencia.sum(axis=0))
    total = a.size * (a.size - 1) / 2
    esperado = pares_a * pares_b / total
    maximo = (pares_a + pares_b) / 2
    if maximo == esperado:
        return 1.0
    return float((indice - esperado) / (maximo - esperado))


def retroproyectar(pesos_pc: np.ndarray, cargas: np.ndarray) -> np.ndarray:
    """Lleva pesos en coordenadas de PCs al espacio de variables estandarizadas."""
    return np.asarray(pesos_pc, dtype=float) @ np.asarray(cargas, dtype=float).T


def qe_asignada(datos: np.ndarray, pesos: np.ndarray, bmu: np.ndarray) -> float:
    """Distancia media de cada muestra a la neurona que le asignó el modelo."""
    diferencias = np.asarray(datos, dtype=float) - np.asarray(pesos)[bmu]
    return float(np.mean(np.linalg.norm(diferencias, axis=1)))


def concordancia(
    asignaciones_a: pd.DataFrame,
    asignaciones_b: pd.DataFrame,
    misma_corrida: bool,
) -> list[float]:
    """ARI para cada par de semillas entre dos corridas.

    Si es la misma corrida, solo se comparan semillas distintas: el resultado
    mide la estabilidad de la agrupación frente a la semilla.
    """
    valores = []
    for semilla_a, grupo_a in asignaciones_a.groupby("semilla"):
        for semilla_b, grupo_b in asignaciones_b.groupby("semilla"):
            if misma_corrida and semilla_a >= semilla_b:
                continue
            unidas = grupo_a.merge(grupo_b, on="creator_id", suffixes=("_a", "_b"))
            valores.append(
                indice_rand_ajustado(
                    unidas["neurona_a"].to_numpy(), unidas["neurona_b"].to_numpy()
                )
            )
    return valores


def _media_desvio(valores: Sequence[float]) -> tuple[float, float]:
    arreglo = np.asarray(valores, dtype=float)
    desvio = float(np.std(arreglo, ddof=1)) if arreglo.size > 1 else 0.0
    return float(np.mean(arreglo)), desvio


def desglose_topografico(
    datos: np.ndarray, pesos: np.ndarray, filas: np.ndarray, columnas: np.ndarray
) -> dict[str, float]:
    """Separa el TE según dónde cae la segunda neurona más parecida.

    Solo para grilla rectangular, donde el TE cuenta como error a las vecinas
    en diagonal (distancia √2). ``vecina`` es ``1 - TE``; ``diagonal`` y
    ``lejana`` (dos casillas o más) suman el TE.
    """
    distancias = np.linalg.norm(datos[:, None, :] - pesos[None, :, :], axis=2)
    orden = np.argsort(distancias, axis=1)
    primera, segunda = orden[:, 0], orden[:, 1]
    en_grilla = np.hypot(
        filas[primera] - filas[segunda], columnas[primera] - columnas[segunda]
    )
    diagonal = np.isclose(en_grilla, np.sqrt(2.0))
    vecina = en_grilla <= 1.0 + 1e-12
    return {
        "vecina": float(np.mean(vecina)),
        "diagonal": float(np.mean(diagonal)),
        "lejana": float(np.mean(~vecina & ~diagonal)),
    }


def entrada_propia(config: dict, columnas: Sequence[str]) -> pd.DataFrame:
    """Reconstruye la entrada con la que se entrenó la corrida."""
    datos = pd.read_csv(resolver_ruta(config["datos"]["ruta"])).set_index("creator_id")
    matriz = datos[list(columnas)].to_numpy(dtype=float)
    if config["datos"].get("estandarizar", True):
        matriz = (matriz - matriz.mean(axis=0)) / matriz.std(axis=0, ddof=1)
    return pd.DataFrame(matriz, index=datos.index, columns=list(columnas))


def cargar_corrida(directorio: Path) -> dict:
    """Lee la configuración, métricas y salidas por semilla de una corrida."""
    with (directorio / "config_usada.toml").open("rb") as archivo:
        config = tomllib.load(archivo)
    return {
        "config": config,
        "metricas": pd.read_csv(directorio / "metricas.csv"),
        "pesos": pd.read_csv(directorio / "pesos_semillas.csv"),
        "asignaciones": pd.read_csv(directorio / "asignaciones_semillas.csv"),
    }


def datos_estandarizados(ruta: Path, variables: Sequence[str]) -> pd.DataFrame:
    """Z-score (ddof=1) de las variables originales, indexado por creator_id."""
    datos = pd.read_csv(ruta).set_index("creator_id")
    matriz = datos[list(variables)].to_numpy(dtype=float)
    z = (matriz - matriz.mean(axis=0)) / matriz.std(axis=0, ddof=1)
    return pd.DataFrame(z, index=datos.index, columns=list(variables))


def metricas_corrida(
    etiqueta: str,
    corrida: dict,
    z: pd.DataFrame,
    cargas: pd.DataFrame,
) -> pd.DataFrame:
    """QE propia, QE común, TE, unidades muertas y ocupación por semilla."""
    pesos = corrida["pesos"]
    columnas_pesos = [
        c for c in pesos.columns if c not in {"semilla", "neurona", "fila", "columna"}
    ]
    en_pcs = set(columnas_pesos).issubset(cargas.columns)
    if not en_pcs and set(columnas_pesos) != set(z.columns):
        raise ValueError(f"{etiqueta}: las variables no son ni PCs ni las de 1.1.")

    modelo = corrida["config"]["modelo"]
    if modelo["topologia"] != "rectangular":
        raise ValueError(f"{etiqueta}: el desglose del TE es solo para rectangular.")
    propia = entrada_propia(corrida["config"], columnas_pesos)
    filas = []
    for fila_metricas in corrida["metricas"].itertuples():
        semilla = int(fila_metricas.semilla)
        pesos_semilla = pesos[pesos["semilla"] == semilla].sort_values("neurona")
        asignadas = corrida["asignaciones"][
            corrida["asignaciones"]["semilla"] == semilla
        ].set_index("creator_id")
        datos = z.loc[asignadas.index]

        if en_pcs:
            matriz_pesos = retroproyectar(
                pesos_semilla[columnas_pesos].to_numpy(),
                cargas.loc[z.columns, columnas_pesos].to_numpy(),
            )
        else:
            matriz_pesos = pesos_semilla[list(z.columns)].to_numpy()

        desglose = desglose_topografico(
            propia.loc[asignadas.index].to_numpy(),
            pesos_semilla[columnas_pesos].to_numpy(),
            pesos_semilla["fila"].to_numpy(),
            pesos_semilla["columna"].to_numpy(),
        )
        filas.append(
            {
                "corrida": etiqueta,
                "entrada": "PC1..PC10" if en_pcs else "27 variables",
                "k": int(modelo["grilla"]),
                "distancia": modelo["distancia"],
                "semilla": semilla,
                "qe_propia": float(fila_metricas.qe),
                "qe_comun": qe_asignada(
                    datos.to_numpy(), matriz_pesos, asignadas["neurona"].to_numpy()
                ),
                "te": float(fila_metricas.te),
                "te_diagonal": desglose["diagonal"],
                "te_lejana": desglose["lejana"],
                "unidades_muertas": int(fila_metricas.unidades_muertas),
                "ocupacion": float(fila_metricas.ocupacion),
            }
        )
    return pd.DataFrame(filas)


def resumir(por_semilla: pd.DataFrame) -> pd.DataFrame:
    """Media y desvío muestral por corrida, conservando el orden de entrada."""
    filas = []
    for etiqueta, grupo in por_semilla.groupby("corrida", sort=False):
        fila = {
            "corrida": etiqueta,
            "entrada": grupo["entrada"].iloc[0],
            "k": int(grupo["k"].iloc[0]),
            "distancia": grupo["distancia"].iloc[0],
            "semillas": len(grupo),
        }
        for metrica in (
            "qe_propia",
            "qe_comun",
            "te",
            "te_diagonal",
            "te_lejana",
            "unidades_muertas",
            "ocupacion",
        ):
            media, desvio = _media_desvio(grupo[metrica])
            fila[f"{metrica}_media"] = media
            fila[f"{metrica}_desvio"] = desvio
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_concordancia(corridas: dict[str, dict]) -> pd.DataFrame:
    """ARI medio entre todos los pares de corridas (incluida cada una consigo)."""
    filas = []
    for a, b in itertools.combinations_with_replacement(corridas, 2):
        valores = concordancia(
            corridas[a]["asignaciones"],
            corridas[b]["asignaciones"],
            misma_corrida=a == b,
        )
        if not valores:
            continue
        media, desvio = _media_desvio(valores)
        filas.append(
            {
                "corrida_a": a,
                "corrida_b": b,
                "ari_media": media,
                "ari_desvio": desvio,
                "pares": len(valores),
            }
        )
    return pd.DataFrame(filas)


def _figura_metricas(resumen: pd.DataFrame, concordancias: pd.DataFrame, ruta: Path):
    paneles = [
        ("qe_comun", "QE común (27 variables estandarizadas)", "menor es mejor"),
        ("te", "Error topográfico (TE)", "menor es mejor"),
        ("unidades_muertas", "Unidades muertas", "neuronas sin creadores"),
        ("ari", "Estabilidad entre semillas (ARI)", "1 = misma agrupación"),
    ]
    estabilidad = concordancias[
        concordancias["corrida_a"] == concordancias["corrida_b"]
    ]
    estabilidad = estabilidad.set_index("corrida_a")

    fig, ejes = plt.subplots(2, 2, figsize=(13, 8.5))
    posiciones = np.arange(len(resumen))
    colores = [COLORES[entrada] for entrada in resumen["entrada"]]
    for eje, (metrica, titulo, nota) in zip(ejes.ravel(), paneles, strict=True):
        if metrica == "ari":
            medias = estabilidad.reindex(resumen["corrida"])["ari_media"].to_numpy()
            desvios = estabilidad.reindex(resumen["corrida"])["ari_desvio"].to_numpy()
        else:
            medias = resumen[f"{metrica}_media"].to_numpy()
            desvios = resumen[f"{metrica}_desvio"].to_numpy()
        eje.bar(posiciones, medias, yerr=desvios, color=colores, capsize=4)
        for x, y in zip(posiciones, medias, strict=True):
            if np.isfinite(y):
                eje.annotate(
                    f"{y:.2f}",
                    (x, y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        eje.set_xticks(posiciones, resumen["corrida"], rotation=30, ha="right")
        eje.set_title(f"{titulo}\n({nota})", fontsize=10)
        eje.grid(axis="y", alpha=0.3)

    manejadores = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORES.values()]
    fig.legend(
        manejadores,
        list(COLORES),
        loc="upper center",
        ncol=2,
        title="Entrada de la red (barras: media ± desvío sobre semillas)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _figura_topografia(resumen: pd.DataFrame, ruta: Path) -> None:
    posiciones = np.arange(len(resumen))
    diagonal = resumen["te_diagonal_media"].to_numpy()
    lejana = resumen["te_lejana_media"].to_numpy()

    fig, eje = plt.subplots(figsize=(10, 5))
    eje.bar(posiciones, diagonal, color="#8DA0CB", label="Vecina en diagonal (√2)")
    eje.bar(
        posiciones,
        lejana,
        bottom=diagonal,
        color="#E15759",
        label="Lejana (2 casillas o más)",
    )
    for x, (d, lj) in enumerate(zip(diagonal, lejana, strict=True)):
        eje.annotate(
            f"TE {d + lj:.3f}\nlejana {lj:.3f}",
            (x, d + lj),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    eje.set_xticks(posiciones, resumen["corrida"], rotation=30, ha="right")
    eje.set_ylabel("Fracción de creadores")
    eje.set_ylim(0, max(0.05, float(np.max(diagonal + lejana))) * 1.3)
    eje.set_title(
        "Desglose del error topográfico: dónde cae la 2ª neurona más parecida\n"
        "Solo la parte lejana indica un mapa plegado (media sobre semillas)",
        fontsize=10,
    )
    eje.legend()
    eje.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _figura_concordancia(concordancias: pd.DataFrame, etiquetas: list[str], ruta):
    matriz = np.full((len(etiquetas), len(etiquetas)), np.nan)
    indice = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
    for fila in concordancias.itertuples():
        i, j = indice[fila.corrida_a], indice[fila.corrida_b]
        matriz[i, j] = matriz[j, i] = fila.ari_media

    fig, eje = plt.subplots(figsize=(8, 6.5))
    imagen = eje.imshow(matriz, cmap="viridis", vmin=0, vmax=1)
    for i, j in itertools.product(range(len(etiquetas)), repeat=2):
        if np.isfinite(matriz[i, j]):
            eje.text(
                j,
                i,
                f"{matriz[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matriz[i, j] < 0.6 else "black",
                fontsize=9,
            )
    eje.set_xticks(range(len(etiquetas)), etiquetas, rotation=35, ha="right")
    eje.set_yticks(range(len(etiquetas)), etiquetas)
    eje.set_title(
        "Concordancia de agrupaciones (ARI medio entre pares de semillas)\n"
        "Diagonal: estabilidad de cada corrida frente a la semilla",
        fontsize=10,
    )
    fig.colorbar(imagen, ax=eje, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _resumen_texto(
    resumen: pd.DataFrame, concordancias: pd.DataFrame, residuo: float
) -> str:
    lineas = [
        "Comparación de redes de Kohonen — TP4, ejercicio 1.2",
        "",
        "QE propia: distancia media creador–neurona en el espacio de entrada de",
        "cada corrida. No es comparable entre 27 variables y PC1..PC10.",
        "QE común: la misma distancia medida en las 27 variables estandarizadas,",
        "retroproyectando los pesos de las corridas sobre PCs con las cargas de 1.1.",
        f"Residuo medio de PC11..PC25 (piso de la QE común con 10 PCs): {residuo:.4f}",
        "",
        "Métricas (media ± desvío sobre semillas):",
    ]
    for fila in resumen.itertuples():
        lineas.append(
            f"  {fila.corrida:<22} k={fila.k} {fila.distancia:<11} "
            f"QE propia {fila.qe_propia_media:.4f} ± {fila.qe_propia_desvio:.4f} | "
            f"QE común {fila.qe_comun_media:.4f} ± {fila.qe_comun_desvio:.4f} | "
            f"TE {fila.te_media:.4f} ± {fila.te_desvio:.4f} "
            f"(diagonal {fila.te_diagonal_media:.4f}, "
            f"lejana {fila.te_lejana_media:.4f}) | "
            f"muertas {fila.unidades_muertas_media:.2f} | "
            f"ocupación {fila.ocupacion_media:.4f}"
        )
    lineas += ["", "Concordancia (ARI medio ± desvío, pares de semillas):"]
    for fila in concordancias.itertuples():
        lineas.append(
            f"  {fila.corrida_a:<22} vs {fila.corrida_b:<22} "
            f"{fila.ari_media:.4f} ± {fila.ari_desvio:.4f} (n={fila.pares})"
        )
    return "\n".join(lineas) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    especificaciones = args.corrida or CORRIDAS_POR_DEFECTO

    corridas: dict[str, dict] = {}
    for especificacion in especificaciones:
        etiqueta, _, directorio = especificacion.partition("=")
        if not directorio:
            raise ValueError(f"Corrida inválida, se espera etiqueta=dir: {etiqueta}")
        corridas[etiqueta] = cargar_corrida(resolver_ruta(directorio))

    cargas = pd.read_csv(resolver_ruta(args.cargas), index_col=0)
    cargas = cargas[[c for c in cargas.columns if c.startswith("PC")]]
    z = datos_estandarizados(resolver_ruta(args.datos), cargas.index)

    cargas_10 = cargas[[f"PC{i}" for i in range(1, 11)]].to_numpy()
    reconstruido = z.to_numpy() @ cargas_10 @ cargas_10.T
    residuo = float(np.mean(np.linalg.norm(z.to_numpy() - reconstruido, axis=1)))

    por_semilla = pd.concat(
        [
            metricas_corrida(etiqueta, corrida, z, cargas)
            for etiqueta, corrida in corridas.items()
        ],
        ignore_index=True,
    )
    resumen = resumir(por_semilla)
    concordancias = tabla_concordancia(corridas)

    salida = resolver_ruta(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    por_semilla.to_csv(
        salida / "metricas_por_semilla.csv", index=False, float_format="%.8f"
    )
    resumen.to_csv(salida / "metricas_resumen.csv", index=False, float_format="%.8f")
    concordancias.to_csv(salida / "concordancia.csv", index=False, float_format="%.8f")
    _figura_metricas(resumen, concordancias, salida / "fig_metricas.png")
    _figura_topografia(resumen, salida / "fig_topografia.png")
    _figura_concordancia(concordancias, list(corridas), salida / "fig_concordancia.png")
    texto = _resumen_texto(resumen, concordancias, residuo)
    (salida / "resumen.txt").write_text(texto, encoding="utf-8")

    print(texto)
    print(f"Resultados: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
