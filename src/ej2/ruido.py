"""Generadores de ruido estructural para los QR (TP4, ejercicio 2.1).

Cada generador recibe un nivel, la forma de la imagen y un
``numpy.random.Generator`` y devuelve la máscara de los píxeles que se borran a
blanco (``-1``) junto con los parámetros efectivos que eligió, para que la
consulta se pueda reconstruir. Los generadores no reciben el patrón: la
posición del daño no depende del contenido del QR. El ruido aleatorio no está
acá; se reutiliza ``aplicar_ruido`` del runner.

Las definiciones exactas de cada modelo están en ``docs/noise.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Código de cada componente para derivar su flujo aleatorio:
# ``default_rng([semilla, código])``. ``aleatorio`` usa ``default_rng(semilla)``.
CODIGOS = {
    "esquina": 1,
    "finder": 2,
    "rayones": 3,
    "manchas": 4,
    "bloques": 5,
}


ESQUINAS = (
    "superior_izquierda",
    "superior_derecha",
    "inferior_izquierda",
    "inferior_derecha",
)


def borrar(patron: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Devuelve una copia del patrón con los píxeles de la máscara en blanco."""
    resultado = np.asarray(patron).astype(np.int8, copy=True)
    resultado[np.asarray(mascara, dtype=bool).reshape(-1)] = -1
    return resultado


def _origen_esquina(esquina: str, alto: int, ancho: int, filas: int, columnas: int):
    """Fila y columna de inicio de un rectángulo pegado a una esquina."""
    fila = filas - alto if esquina.startswith("inferior") else 0
    columna = columnas - ancho if esquina.endswith("derecha") else 0
    return fila, columna


def mascara_esquina(
    lado: int, filas: int, columnas: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Cuadrado de ``lado × lado`` en una de las cuatro esquinas, al azar."""
    if not 1 <= int(lado) <= min(filas, columnas):
        raise ValueError("El lado debe estar entre 1 y el tamaño de la imagen.")
    lado = int(lado)
    esquina = ESQUINAS[int(rng.integers(len(ESQUINAS)))]
    fila, columna = _origen_esquina(esquina, lado, lado, filas, columnas)
    mascara = np.zeros((filas, columnas), dtype=bool)
    mascara[fila : fila + lado, columna : columna + lado] = True
    return mascara, {"esquina": esquina, "lado": lado}


FINDERS = ESQUINAS[:3]
LADO_FINDER = 7


def mascara_finder(
    cantidad: int, filas: int, columnas: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Borra ``cantidad`` patrones de localización de 7 × 7, elegidos sin repetir.

    Los QR no tienen zona de silencio: los finders están pegados a las esquinas
    superior izquierda, superior derecha e inferior izquierda.
    """
    if not 1 <= int(cantidad) <= len(FINDERS):
        raise ValueError(f"La cantidad de finders va de 1 a {len(FINDERS)}.")
    indices = sorted(rng.choice(len(FINDERS), size=int(cantidad), replace=False))
    elegidos = [FINDERS[int(indice)] for indice in indices]
    mascara = np.zeros((filas, columnas), dtype=bool)
    for finder in elegidos:
        fila, columna = _origen_esquina(
            finder, LADO_FINDER, LADO_FINDER, filas, columnas
        )
        mascara[fila : fila + LADO_FINDER, columna : columna + LADO_FINDER] = True
    return mascara, {"finders": elegidos, "cantidad": int(cantidad)}


ORIENTACIONES = (
    "horizontal",
    "vertical",
    "diagonal_descendente",
    "diagonal_ascendente",
)


def mascara_rayones(
    cantidad: int, filas: int, columnas: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Líneas de un píxel de grosor, de borde a borde, que pueden cruzarse.

    Cada línea elige una orientación al azar y pasa por un píxel al azar.
    """
    if int(cantidad) < 1:
        raise ValueError("Hace falta al menos una línea.")
    i, j = np.indices((filas, columnas))
    mascara = np.zeros((filas, columnas), dtype=bool)
    lineas = []
    for _ in range(int(cantidad)):
        orientacion = ORIENTACIONES[int(rng.integers(len(ORIENTACIONES)))]
        fila, columna = divmod(int(rng.integers(filas * columnas)), columnas)
        if orientacion == "horizontal":
            mascara |= i == fila
        elif orientacion == "vertical":
            mascara |= j == columna
        elif orientacion == "diagonal_descendente":
            mascara |= i - j == fila - columna
        else:
            mascara |= i + j == fila + columna
        lineas.append({"orientacion": orientacion, "fila": fila, "columna": columna})
    return mascara, {"lineas": lineas, "cantidad": int(cantidad)}


def mascara_manchas(
    nivel: Sequence[int], filas: int, columnas: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Discos ``(i − ci)² + (j − cj)² ≤ r²`` con centros al azar.

    ``nivel`` es ``[cantidad, radio]``. Los discos se recortan en el borde y
    pueden solaparse.
    """
    cantidad, radio = (int(valor) for valor in nivel)
    if cantidad < 1 or radio < 0:
        raise ValueError("Hace falta al menos una mancha y un radio no negativo.")
    i, j = np.indices((filas, columnas))
    mascara = np.zeros((filas, columnas), dtype=bool)
    centros = []
    for _ in range(cantidad):
        fila, columna = divmod(int(rng.integers(filas * columnas)), columnas)
        mascara |= (i - fila) ** 2 + (j - columna) ** 2 <= radio**2
        centros.append([fila, columna])
    return mascara, {"centros": centros, "radio": radio, "cantidad": cantidad}


LADO_BLOQUE = 6


def mascara_bloques(
    fraccion: float, filas: int, columnas: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Borra ``round(fraccion · B)`` de los ``B`` bloques completos de 6 × 6.

    Con 37 × 37 hay 36 bloques completos; la última fila y la última columna,
    de un píxel, no se usan.
    """
    por_fila, por_columna = filas // LADO_BLOQUE, columnas // LADO_BLOQUE
    total = por_fila * por_columna
    cantidad = int(round(float(fraccion) * total))
    if not 1 <= cantidad <= total:
        raise ValueError("La fracción debe dar entre 1 y todos los bloques.")
    bloques = sorted(int(b) for b in rng.choice(total, size=cantidad, replace=False))
    mascara = np.zeros((filas, columnas), dtype=bool)
    for bloque in bloques:
        fila, columna = divmod(bloque, por_columna)
        mascara[
            LADO_BLOQUE * fila : LADO_BLOQUE * (fila + 1),
            LADO_BLOQUE * columna : LADO_BLOQUE * (columna + 1),
        ] = True
    return mascara, {"bloques": bloques, "cantidad": cantidad, "tamano": LADO_BLOQUE}


GENERADORES: dict[str, Callable[..., tuple[np.ndarray, dict]]] = {
    "esquina": mascara_esquina,
    "finder": mascara_finder,
    "rayones": mascara_rayones,
    "manchas": mascara_manchas,
    "bloques": mascara_bloques,
}
