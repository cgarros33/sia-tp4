#!/usr/bin/env python3
"""Figuras del ejercicio 1 rediseñadas para la presentación (TP4).

Script de análisis: solo lee resultados de 1.1 (correlaciones y cargas del
PCA) y del barrido de Kohonen; no entrena nada.

- ``fig_correlaciones_ordenadas``: matriz de correlación de las 27 variables
  del PCA, ordenadas por su carga en PC1 para que se vean los bloques.
- ``fig_barrido``: QE media ± desvío de cada valor probado en cada paso del
  barrido secuencial, con un eje compartido para comparar pasos.
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CORRELACIONES = RAIZ / "resultados" / "ej1" / "eda" / "correlaciones.csv"
RUTA_CARGAS = RAIZ / "resultados" / "ej1" / "pca_0.90" / "cargas.csv"
RUTA_BARRIDO = (
    RAIZ / "resultados" / "ej1" / "ej1.2-pca10" / "kohonen_sweep_euclidiana" / "sweep4"
)
RUTA_SALIDA = RAIZ / "resultados" / "ej1" / "presentacion"

TITULOS_PASO = {
    "topologia": "Topología",
    "distancia": "Distancia",
    "inicializacion": "Inicialización",
    "radio_inicial_relativo": "Radio inicial",
    "decaimiento_radio": "Decaimiento del radio",
    "eta_inicial": "η inicial",
    "decaimiento_eta": "Decaimiento de η",
    "iteraciones": "Iteraciones",
}
COLOR_ELEGIDO = "#6250d6"
COLOR_RESTO = "#8f8e88"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRILLA = "#e4e3df"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlecolor": TINTA,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Figuras del ejercicio 1 para la presentación.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--correlaciones", type=Path, default=RUTA_CORRELACIONES)
    parser.add_argument("--cargas", type=Path, default=RUTA_CARGAS)
    parser.add_argument("--barrido", type=Path, default=RUTA_BARRIDO)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    return parser.parse_args(argv)


def _coma(valor: float, formato: str = ".2f") -> str:
    return format(valor, formato).replace(".", ",")


def etiqueta(variable: str) -> str:
    """Nombre corto de una variable para los ejes."""
    return (
        variable.replace("_per_1000", " /1000")
        .replace("_pct", " %")
        .replace("average_", "")
        .replace("_", " ")
    )


def orden_por_carga(cargas: pd.DataFrame, componente: str = "PC1") -> list[str]:
    """Variables de mayor a menor carga en la componente."""
    return list(cargas[componente].sort_values(ascending=False).index)


def formato_valor(parametro: str, valor: object) -> str:
    """Valor de un hiperparámetro tal como se muestra en la figura."""
    if isinstance(valor, str):
        return valor
    if parametro == "radio_inicial_relativo":
        return f"{_coma(float(valor), 'g')}·k"
    if parametro == "iteraciones":
        return f"{int(valor):,}".replace(",", ".")
    return _coma(float(valor), "g")


def mismo_valor(a: object, b: object) -> bool:
    """Compara valores del barrido que pueden estar guardados como texto."""
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _figura_correlaciones(
    correlaciones: pd.DataFrame, cargas: pd.DataFrame, ruta: Path
) -> None:
    orden = orden_por_carga(cargas)
    # Triángulo inferior sin la diagonal: se omiten la primera fila y la última
    # columna, que quedarían vacías.
    filas, columnas = orden[1:], orden[:-1]
    matriz = correlaciones.loc[filas, columnas].to_numpy()
    visible = np.ma.masked_where(np.triu(np.ones_like(matriz, dtype=bool), k=1), matriz)

    fig, eje = plt.subplots(figsize=(11, 10))
    imagen = eje.imshow(visible, cmap="RdBu_r", vmin=-1, vmax=1)
    for i, j in zip(*np.tril_indices(len(filas)), strict=True):
        if abs(matriz[i, j]) >= 0.9:
            eje.text(
                j,
                i,
                _coma(matriz[i, j]).replace("0,", ","),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
            )
    eje.set_xticks(
        range(len(columnas)), [etiqueta(v) for v in columnas], rotation=90, fontsize=9
    )
    eje.set_yticks(range(len(filas)), [etiqueta(v) for v in filas], fontsize=9)
    for borde in eje.spines.values():
        borde.set_visible(False)
    barra = fig.colorbar(imagen, ax=eje, fraction=0.035, pad=0.02)
    barra.set_label("Correlación de Pearson")
    eje.set_title(
        "Correlación entre las 27 variables (ya sin las 4 redundantes exactas)\n"
        "Ordenadas por su carga en PC1: arriba lo masivo, abajo lo profundo. "
        "Números: |r| ≥ 0,9",
        fontsize=12,
    )
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def leer_barrido(directorio: Path) -> list[tuple[str, pd.DataFrame, object]]:
    """Pasos del barrido: parámetro, resumen por valor y valor elegido."""
    pasos = pd.read_csv(directorio / "resumen_pasos.csv").sort_values("paso")
    resultado = []
    for fila in pasos.itertuples():
        archivo = directorio / f"paso_{fila.paso:02d}_{fila.parametro}_resumen.csv"
        resultado.append((fila.parametro, pd.read_csv(archivo), fila.valor_elegido))
    return resultado


def _figura_barrido(directorio: Path, ruta: Path) -> None:
    pasos = leer_barrido(directorio)
    acoplamientos = pd.read_csv(
        directorio / "acoplamientos" / "resumen_acoplamientos.csv"
    )
    with (directorio / "config_optima.toml").open("rb") as archivo:
        optima = tomllib.load(archivo)["modelo"]
    adoptados = acoplamientos[acoplamientos["adoptado"]]
    # QE del último paso, o la de un acoplamiento si se adoptó y la mejora.
    ultimo = pd.read_csv(directorio / "resumen_pasos.csv")["qe_media"].iloc[-1]
    qe_final = min([ultimo, *adoptados["qe_media"]])
    todos = pd.concat([p[1] for p in pasos])
    techo = (todos["qe_media"] + todos["qe_desvio"]).max()
    piso = (todos["qe_media"] - todos["qe_desvio"]).min()
    margen = 0.08 * (techo - piso)

    columnas = 4
    filas = int(np.ceil(len(pasos) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(16, 4.3 * filas), sharey=True)
    for indice, (eje, (parametro, resumen, elegido)) in enumerate(
        zip(ejes.flat, pasos, strict=False), start=1
    ):
        x = np.arange(len(resumen))
        es_elegido = resumen["valor"].map(lambda v, e=elegido: mismo_valor(v, e))
        colores = np.where(es_elegido, COLOR_ELEGIDO, COLOR_RESTO)
        for posicion, fila, color in zip(x, resumen.itertuples(), colores, strict=True):
            eje.errorbar(
                posicion,
                fila.qe_media,
                yerr=fila.qe_desvio,
                fmt="o",
                ms=9,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
            )
        eje.axhline(qe_final, color=TINTA_SECUNDARIA, lw=1, ls=":", zorder=0)
        etiquetas = [formato_valor(parametro, v) for v in resumen["valor"]]
        eje.set_xticks(
            x,
            etiquetas,
            rotation=35 if len(x) > 4 else 0,
            ha="right" if len(x) > 4 else "center",
            fontsize=9.5,
        )
        for etiqueta_x, elegida in zip(eje.get_xticklabels(), es_elegido, strict=True):
            if elegida:
                etiqueta_x.set_color(COLOR_ELEGIDO)
                etiqueta_x.set_fontweight("bold")
        eje.set_xlim(-0.6, len(x) - 0.4)
        eje.set_title(
            f"{indice}. {TITULOS_PASO.get(parametro, parametro)}", fontsize=12
        )
        eje.grid(axis="y", color=GRILLA, lw=0.6)
        if len(x) == 1:
            nota = "fija (barrido de control)"
        elif resumen["qe_media"].nunique() == 1:
            nota = "todas dan lo mismo"
        else:
            nota = ""
        if nota:
            eje.text(
                0.5,
                0.92,
                nota,
                transform=eje.transAxes,
                ha="center",
                fontsize=10,
                color=TINTA_SECUNDARIA,
            )
    for eje in ejes.flat[len(pasos) :]:
        eje.set_visible(False)
    for eje in ejes[:, 0]:
        eje.set_ylabel("QE (media ± desvío, 3 semillas)")
    ejes[0, 0].set_ylim(piso - margen, techo + margen)

    acople = ""
    if not adoptados.empty:
        fila = adoptados.iloc[0]
        acople = (
            f" Después, {TITULOS_PASO[fila['parametro_a']]} y su decaimiento juntos: "
            f"{formato_valor(fila['parametro_a'], fila['mejor_valor_a'])} "
            f"{fila['mejor_valor_b']} (mejora de {_coma(fila['qe_base'] - fila['qe_media'], '.4f')})."
        )
    fig.suptitle(
        f"Barrido secuencial de hiperparámetros (grilla {optima['grilla']}×"
        f"{optima['grilla']}, distancia {optima['distancia']}): cada paso fija el "
        "mejor valor y pasa al siguiente\n"
        f"Violeta = valor elegido · línea punteada = QE final ({_coma(qe_final)})."
        + acople,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    correlaciones = pd.read_csv(args.correlaciones, index_col=0)
    cargas = pd.read_csv(args.cargas, index_col=0)
    _figura_correlaciones(
        correlaciones, cargas, salida / "fig_correlaciones_ordenadas.png"
    )
    _figura_barrido(args.barrido, salida / "fig_barrido.png")
    print(f"Figuras: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
