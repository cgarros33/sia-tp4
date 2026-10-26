#!/usr/bin/env python3
"""Figuras de Hopfield para la presentación (TP4, ejercicios 2.1 y 2.2).

Script de análisis: solo lee las redes y consultas guardadas por
``hopfield.py`` y ``hopfield-runner.py``; no calcula pesos ni recupera.

- ``fig_letras_ortogonalidad``: las 4 letras almacenadas y su solapamiento.
- ``fig_letras_ruido``: resultado de las consultas según el ruido (media ±
  desvío sobre semillas) y un estado espurio.
- ``fig_qr_fallas``: dos casos en que la pseudoinversa no recupera el QR.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_LETRAS = RAIZ / "resultados" / "ej2" / "hopfield" / "letras_5x5"
RUTA_QR = RAIZ / "resultados" / "ej2" / "hopfield" / "qr_pseudoinversa"
RUTA_SALIDA = RAIZ / "resultados" / "ej2" / "hopfield" / "figuras"

CARPETA_RUIDO = re.compile(r"^ruido_([0-9.]+)_semilla_(\d+)$")
TIPOS = {
    "almacenado_correcto": "Recupera la letra",
    "almacenado_incorrecto": "Cae en otra letra",
    "espurio": "Estado espurio",
}
# Primeras tres posiciones de la paleta categórica validada.
COLOR_TIPO = {
    "almacenado_correcto": "#2a78d6",
    "almacenado_incorrecto": "#eb6834",
    "espurio": "#1baf7a",
}
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRILLA = "#e4e3df"
# Caso del estado espurio de letras y casos de falla de los QR.
CASO_ESPURIO = ("ruido_0.2_semilla_0", "espurio")
CASOS_QR = [
    ("esquina_0.5_semilla_0", "almacenado_incorrecto", "Esquina rota 50%"),
    ("ruido_0.45_semilla_0", "espurio", "Ruido aleatorio 45%"),
]

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
        description="Figuras de Hopfield para la presentación.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--letras", type=Path, default=RUTA_LETRAS)
    parser.add_argument("--qr", type=Path, default=RUTA_QR)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    return parser.parse_args(argv)


def _coma(valor: float, formato: str = ".2f") -> str:
    return format(valor, formato).replace(".", ",")


def leer_estados(ruta: Path) -> pd.DataFrame:
    """Estados guardados (patrones o consultas), indexados por nombre."""
    tabla = pd.read_csv(ruta)
    return tabla.set_index(tabla.columns[0])


def pixeles(tabla: pd.DataFrame) -> np.ndarray:
    """Matriz de píxeles (una fila por estado)."""
    columnas = [c for c in tabla.columns if re.fullmatch(r"p\d+", c)]
    return tabla[columnas].to_numpy(dtype=int)


def composicion_por_semilla(consultas: Path) -> pd.DataFrame:
    """Fracción de cada tipo de resultado por nivel de ruido y semilla."""
    filas = []
    for carpeta in sorted(consultas.iterdir()):
        coincidencia = CARPETA_RUIDO.match(carpeta.name)
        if not coincidencia:
            continue
        tipos = pd.read_csv(carpeta / "consultas.csv")["tipo_resultado"]
        fracciones = tipos.value_counts(normalize=True)
        filas.append(
            {
                "ruido": float(coincidencia.group(1)),
                "semilla": int(coincidencia.group(2)),
                "consultas": len(tipos),
                **{tipo: float(fracciones.get(tipo, 0.0)) for tipo in TIPOS},
            }
        )
    return pd.DataFrame(filas).sort_values(["ruido", "semilla"], ignore_index=True)


def resumir_composicion(por_semilla: pd.DataFrame) -> pd.DataFrame:
    """Media y desvío entre semillas de cada fracción, por nivel de ruido."""
    agrupado = por_semilla.groupby("ruido")[list(TIPOS)]
    return pd.concat(
        {"media": agrupado.mean(), "desvio": agrupado.std(ddof=1)}, axis=1
    ).swaplevel(axis=1)


def primer_caso(consultas: pd.DataFrame, tipo: str) -> pd.Series:
    """Primera consulta con el tipo de resultado pedido."""
    casos = consultas[consultas["tipo_resultado"] == tipo]
    if casos.empty:
        raise ValueError(f"No hay consultas de tipo {tipo}.")
    return casos.iloc[0]


def _imagen(eje: plt.Axes, estado: np.ndarray, lado: int, titulo: str) -> None:
    # 1 es negro y -1 blanco, como en el papel.
    eje.imshow(
        np.asarray(estado).reshape(lado, lado),
        cmap="gray_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    eje.set_xticks([])
    eje.set_yticks([])
    for borde in eje.spines.values():
        borde.set_visible(True)
        borde.set_color(GRILLA)
    eje.set_title(titulo, fontsize=10.5)


def _figura_letras_ortogonalidad(directorio: Path, ruta: Path) -> None:
    patrones = leer_estados(directorio / "patrones.csv")
    solapamiento = pd.read_csv(directorio / "solapamiento.csv", index_col=0)
    capacidad = pd.read_csv(directorio / "capacidad.csv").iloc[0]
    estabilidad = pd.read_csv(directorio / "estabilidad.csv", index_col=0)
    lado = int(patrones["filas"].iloc[0])
    nombres = list(patrones.index)

    fig = plt.figure(figsize=(13, 4.2))
    rejilla = fig.add_gridspec(
        1, len(nombres) + 2, width_ratios=[1] * len(nombres) + [0.35, 2.2], wspace=0.25
    )
    for i, nombre in enumerate(nombres):
        margen = estabilidad.loc[nombre, "margen_minimo"]
        _imagen(
            fig.add_subplot(rejilla[0, i]),
            pixeles(patrones.loc[[nombre]])[0],
            lado,
            f"{nombre}\nmargen {_coma(margen)}",
        )
    eje = fig.add_subplot(rejilla[0, len(nombres) + 1])
    valores = solapamiento.loc[nombres, nombres].to_numpy()
    imagen = eje.imshow(valores, cmap="RdBu_r", vmin=-1, vmax=1)
    for i, j in np.ndindex(valores.shape):
        eje.text(
            j,
            i,
            _coma(valores[i, j]),
            ha="center",
            va="center",
            fontsize=11,
            color="white" if abs(valores[i, j]) > 0.6 else TINTA,
        )
    eje.set_xticks(range(len(nombres)), nombres)
    eje.set_yticks(range(len(nombres)), nombres)
    eje.set_title("Solapamiento entre letras\n(0 = ortogonales)", fontsize=11)
    fig.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)

    estables = int(estabilidad["punto_fijo"].sum())
    fig.suptitle(
        f"{len(nombres)} letras en N = {int(capacidad['neuronas'])} neuronas: carga "
        f"p/N = {_coma(capacidad['carga'])} (límite 0,15) y {estables} de "
        f"{len(nombres)} son estables. Ningún par es ortogonal.",
        fontsize=12.5,
    )
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figura_letras_ruido(
    directorio: Path, por_semilla: pd.DataFrame, ruta: Path
) -> None:
    resumen = resumir_composicion(por_semilla)
    niveles = resumen.index.to_numpy()
    por_nivel = int(por_semilla.groupby("ruido")["consultas"].sum().iloc[0])
    semillas = por_semilla["semilla"].nunique()

    fig = plt.figure(figsize=(14, 5.4))
    rejilla = fig.add_gridspec(2, 4, width_ratios=[2.6, 0.12, 1, 1])
    eje = fig.add_subplot(rejilla[:, 0])
    x = np.arange(len(niveles))
    base = np.zeros(len(niveles))
    for tipo, etiqueta in TIPOS.items():
        medias = resumen[(tipo, "media")].to_numpy() * 100
        eje.bar(
            x,
            medias,
            0.65,
            bottom=base,
            color=COLOR_TIPO[tipo],
            edgecolor="white",
            lw=2,
            label=etiqueta,
        )
        base += medias
    correctos = resumen[("almacenado_correcto", "media")].to_numpy() * 100
    desvios = resumen[("almacenado_correcto", "desvio")].to_numpy() * 100
    for posicion, media, desvio in zip(x, correctos, desvios, strict=True):
        eje.text(
            posicion,
            max(media, 8) / 2,
            f"{media:.0f}%\n±{desvio:.0f}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
    eje.set_xticks(x, [f"{n * 100:.0f}%" for n in niveles])
    eje.set_xlabel("Píxeles invertidos al azar")
    eje.set_ylabel("% de consultas")
    eje.set_ylim(0, 100)
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    eje.set_title(
        f"Resultado de la recuperación ({por_nivel} consultas por nivel; "
        f"media ± desvío sobre {semillas} semillas)",
        fontsize=11,
    )

    carpeta, tipo = CASO_ESPURIO
    consultas = pd.read_csv(directorio / "consultas" / carpeta / "consultas.csv")
    caso = primer_caso(consultas, tipo)
    patrones = leer_estados(directorio / "patrones.csv")
    lado = int(patrones["filas"].iloc[0])
    iniciales = leer_estados(
        directorio / "consultas" / carpeta / "estados_iniciales.csv"
    )
    finales = leer_estados(directorio / "consultas" / carpeta / "estados_finales.csv")
    origen, cercano = caso["nombre_origen"], caso["nombre_mas_cercano"]
    paneles = [
        (pixeles(patrones.loc[[origen]])[0], f"{origen} original"),
        (
            pixeles(iniciales.loc[[caso["consulta"]]])[0],
            f"{origen} con {caso['ruido'] * 100:.0f}% de ruido",
        ),
        (
            pixeles(finales.loc[[caso["consulta"]]])[0],
            f"Estado final: espurio\nE = {_coma(caso['energia_final'])}"
            + (
                f" (igual que {origen})"
                if np.isclose(
                    caso["energia_final"],
                    pd.read_csv(directorio / "estabilidad.csv", index_col=0).loc[
                        origen, "energia_final"
                    ],
                )
                else ""
            ),
        ),
        (
            pixeles(patrones.loc[[cercano]])[0],
            f"Letra más cercana: {cercano}\n(a {caso['distancia_hamming']} píxeles)",
        ),
    ]
    for indice, (estado, titulo) in enumerate(paneles):
        _imagen(
            fig.add_subplot(rejilla[indice // 2, 2 + indice % 2]), estado, lado, titulo
        )
    fig.suptitle(
        "Con más ruido, la red cae en estados espurios: mínimos estables que no son "
        "ninguna letra",
        fontsize=12.5,
    )
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _corto(nombre: str) -> str:
    return nombre.replace("_menu_qr", "")


def _figura_qr_fallas(directorio: Path, ruta: Path) -> None:
    patrones = leer_estados(directorio / "patrones.csv")
    lado = int(patrones["filas"].iloc[0])
    fig, ejes = plt.subplots(
        len(CASOS_QR), 4, figsize=(13, 3.9 * len(CASOS_QR)), layout="constrained"
    )
    # Distancia de Hamming mínima entre pares de QR almacenados.
    matriz = pixeles(patrones)
    distancias = (matriz[:, None, :] != matriz[None, :, :]).sum(axis=2)
    minima = distancias[np.triu_indices(len(matriz), k=1)].min()
    for fila, (carpeta, tipo, descripcion) in zip(ejes, CASOS_QR, strict=True):
        ruta_consultas = directorio / "consultas" / carpeta
        caso = primer_caso(pd.read_csv(ruta_consultas / "consultas.csv"), tipo)
        original = pixeles(patrones.loc[[caso["nombre_origen"]]])[0]
        inicial = pixeles(
            leer_estados(ruta_consultas / "estados_iniciales.csv").loc[
                [caso["consulta"]]
            ]
        )[0]
        final = pixeles(
            leer_estados(ruta_consultas / "estados_finales.csv").loc[[caso["consulta"]]]
        )[0]
        origen = _corto(caso["nombre_origen"])
        if tipo == "almacenado_incorrecto":
            resultado = f"Devuelve otro QR válido:\n{_corto(caso['nombre_final'])}"
        else:
            resultado = (
                f"Estado espurio\n(a {caso['distancia_hamming']} píxeles de "
                f"{_corto(caso['nombre_mas_cercano'])})"
            )
        _imagen(fila[0], original, lado, f"Original: {origen}")
        _imagen(
            fila[1],
            inicial,
            lado,
            f"{descripcion}\n({int(caso['hamming_inicial'])} píxeles cambiados)",
        )
        _imagen(fila[2], final, lado, resultado)

        # Diferencias con el original en rojo sobre el QR original atenuado.
        distintos = (final != original).reshape(lado, lado)
        fondo = np.where(original.reshape(lado, lado) == 1, 0.72, 0.97)
        rgb = np.stack([fondo, fondo, fondo], axis=-1)
        rgb[distintos] = (0.85, 0.15, 0.15)
        fila[3].imshow(rgb, interpolation="nearest")
        fila[3].set_xticks([])
        fila[3].set_yticks([])
        aclaracion = (
            ": el par de QR\nmás parecido"
            if tipo == "almacenado_incorrecto" and distintos.sum() == minima
            else " en rojo"
        )
        fila[3].set_title(
            f"Diferencias con {origen}\n({int(distintos.sum())} píxeles{aclaracion})",
            fontsize=10.5,
        )
    fig.suptitle(
        "Límite de la pseudoinversa: cuando falla, a veces devuelve OTRO menú "
        "(peor que un espurio)",
        fontsize=12.5,
    )
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)

    por_semilla = composicion_por_semilla(args.letras / "consultas")
    por_semilla.to_csv(
        salida / "letras_composicion.csv", index=False, float_format="%.4f"
    )
    _figura_letras_ortogonalidad(args.letras, salida / "fig_letras_ortogonalidad.png")
    _figura_letras_ruido(args.letras, por_semilla, salida / "fig_letras_ruido.png")
    _figura_qr_fallas(args.qr, salida / "fig_qr_fallas.png")

    resumen = resumir_composicion(por_semilla)
    print("Composición de resultados de letras (media ± desvío sobre semillas):")
    print(resumen.round(3).to_string())
    print(f"Figuras: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
