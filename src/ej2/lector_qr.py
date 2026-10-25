"""Lectura de estados de Hopfield con un lector de QR (TP4, ejercicio 2.1).

OpenCV se usa solo para decodificar el estado como lo haría un lector de QR; no
es parte del modelo. Para leer un estado se agrega una zona de silencio de 4
módulos en blanco, se escala cada módulo a 10 × 10 píxeles con vecino más
cercano y se decodifica con ``cv2.QRCodeDetector``.
"""

from __future__ import annotations

import cv2
import numpy as np

MARGEN = 4
ESCALA = 10

_detector = cv2.QRCodeDetector()
# Los estados finales se repiten mucho (el estado mezcla de Hebb, los patrones
# recuperados): se lee cada estado distinto una sola vez.
_lecturas: dict[tuple[bytes, int, int], str] = {}


def a_imagen(
    estado: np.ndarray,
    filas: int,
    columnas: int,
    margen: int = MARGEN,
    escala: int = ESCALA,
) -> np.ndarray:
    """Imagen en escala de grises: 1 en negro, -1 en blanco y zona de silencio."""
    matriz = np.where(np.asarray(estado).reshape(filas, columnas) == 1, 0, 255)
    matriz = np.pad(matriz.astype(np.uint8), margen, constant_values=255)
    return np.repeat(np.repeat(matriz, escala, axis=0), escala, axis=1)


def leer(estado: np.ndarray, filas: int, columnas: int) -> str:
    """Texto decodificado del estado, o ``""`` si el lector no lo puede leer."""
    clave = (np.asarray(estado, dtype=np.int8).tobytes(), filas, columnas)
    if clave not in _lecturas:
        texto, _, _ = _detector.detectAndDecode(a_imagen(estado, filas, columnas))
        _lecturas[clave] = texto or ""
    return _lecturas[clave]
