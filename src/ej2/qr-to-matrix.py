#!/usr/bin/env python3
"""Conversor de códigos QR en PNG a matrices binarias (TP4, ejercicio 2.2).

El script lee una o más imágenes, las convierte a escala de grises y las
transforma en una matriz de ``1`` (negro) y ``-1`` (blanco). El resultado se
guarda en un CSV con una fila por imagen, en el mismo formato que usan los
datasets de letras del ejercicio 2.2.

Uso:
    python3 src/ej2/qr-to-matrix.py datos/hopfield/*.png \
        --salida datos/hopfield/qr_37x37.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

UMBRAL = 128


def leer_matriz(ruta: str | Path) -> np.ndarray:
    """Lee un PNG y devuelve su matriz binaria de ``1`` y ``-1``.

    Args:
        ruta: Ruta al archivo PNG.

    Returns:
        Matriz bidimensional con ``1`` para los píxeles negros y ``-1`` para
        los blancos.

    Raises:
        ValueError: Si la imagen no tiene una matriz bidimensional no vacía.
    """
    camino = Path(ruta)
    with Image.open(camino) as imagen:
        gris = imagen.convert("L")
        arreglo = np.asarray(gris, dtype=np.uint8)

    if arreglo.ndim != 2 or arreglo.size == 0:
        raise ValueError(f"La imagen no es una matriz 2D no vacía: {camino}")

    return np.where(arreglo < UMBRAL, 1, -1).astype(np.int8)


def construir_tabla(rutas: Sequence[str | Path]) -> pd.DataFrame:
    """Construye el dataset con una fila por imagen.

    Args:
        rutas: Rutas de los PNG que forman el dataset.

    Returns:
        Tabla con las columnas ``nombre``, ``filas``, ``columnas`` y una
        columna por píxel.

    Raises:
        ValueError: Si no hay imágenes, hay nombres duplicados o los tamaños
            no coinciden.
    """
    if not rutas:
        raise ValueError("Se necesita al menos un PNG para construir el dataset.")

    nombres = [Path(ruta).stem for ruta in rutas]
    if any(not nombre for nombre in nombres):
        raise ValueError("Todos los PNG deben tener un nombre de archivo válido.")

    duplicados = sorted({nombre for nombre in nombres if nombres.count(nombre) > 1})
    if duplicados:
        raise ValueError(f"Hay nombres de PNG duplicados: {duplicados}.")

    matrices = [leer_matriz(ruta) for ruta in rutas]
    forma = matrices[0].shape
    for ruta, matriz in zip(rutas, matrices, strict=True):
        if matriz.shape != forma:
            raise ValueError(
                "Todos los PNG deben tener el mismo tamaño: "
                f"{Path(ruta).name} tiene {matriz.shape}, se esperaba {forma}."
            )

    filas, columnas = forma
    cantidad_pixeles = filas * columnas
    ancho = max(4, len(str(cantidad_pixeles - 1)))
    columnas_pixeles = [f"p{indice:0{ancho}d}" for indice in range(cantidad_pixeles)]

    registros: list[dict[str, object]] = []
    for nombre, matriz in zip(nombres, matrices, strict=True):
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

    return pd.DataFrame(
        registros,
        columns=["nombre", "filas", "columnas", *columnas_pixeles],
    )


def escribir_tabla(tabla: pd.DataFrame, salida: str | Path) -> None:
    """Guarda la tabla en un CSV, creando las carpetas necesarias."""
    ruta = Path(salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el conversor desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Convierte PNG de códigos QR a un CSV de matrices binarias "
            "(1 = negro, -1 = blanco)."
        )
    )
    parser.add_argument(
        "pngs",
        nargs="+",
        type=Path,
        help="Rutas de los PNG a convertir.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help=(
            "Ruta del CSV de salida. Es obligatoria con varios PNG; con un "
            "solo PNG, si se omite, el CSV se imprime por stdout."
        ),
    )
    argumentos = parser.parse_args(argv)

    if argumentos.salida is None and len(argumentos.pngs) > 1:
        parser.error("Con varios PNG es obligatorio indicar --salida.")

    tabla = construir_tabla(argumentos.pngs)

    if argumentos.salida is None:
        tabla.to_csv(sys.stdout, index=False)
    else:
        escribir_tabla(tabla, argumentos.salida)
        print(
            f"Generado {argumentos.salida} con {len(tabla)} patrones.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
