#!/usr/bin/env python3
"""Motor de una red de Hopfield discreta (TP4, ejercicio 2.2).

El script lee un dataset de patrones binarios ``-1``/``1``, calcula la matriz de
pesos con la regla de Hebb y guarda la red entrenada junto con las tablas de
estabilidad, ortogonalidad, capacidad y energía.

El aprendizaje de Hopfield es la regla de Hebb en forma cerrada. La iteración
hasta el estado estable corresponde a la recuperación, no al aprendizaje.

Referencia teórica: ``artif-eng.pdf``, capítulo 15, pp. 125-130.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ / "src" / "ej2" / "hopfield.toml"

MODOS = ("asincronica", "sincronica")
ORDENES = ("secuencial", "aleatorio")

CONFIG_POR_DEFECTO: dict = {
    "datos": {
        "ruta": "datos/hopfield/letras_5x5.csv",
    },
    "modelo": {
        "modo": "asincronica",
        "orden": "secuencial",
        "semilla": 0,
        "max_barridos": 100,
    },
    "salida": {
        "directorio": "",
    },
}


@dataclass(frozen=True)
class Dataset:
    """Patrones binarios y metadatos de forma."""

    nombres: tuple[str, ...]
    patrones: np.ndarray
    filas: int
    columnas: int

    @property
    def n_neuronas(self) -> int:
        """Cantidad de neuronas de la red."""
        return self.filas * self.columnas

    @property
    def n_patrones(self) -> int:
        """Cantidad de patrones almacenados."""
        return len(self.nombres)


@dataclass(frozen=True)
class RedHopfield:
    """Red de Hopfield cargada desde disco."""

    pesos: np.ndarray
    patrones: np.ndarray
    nombres: tuple[str, ...]
    filas: int
    columnas: int

    @property
    def n_neuronas(self) -> int:
        """Cantidad de neuronas de la red."""
        return self.filas * self.columnas

    @property
    def n_patrones(self) -> int:
        """Cantidad de patrones almacenados."""
        return len(self.nombres)


@dataclass(frozen=True)
class PasoEnergia:
    """Energía registrada después de un barrido."""

    barrido: int
    energia: float
    cambios: int


@dataclass(frozen=True)
class ResultadoRecuperacion:
    """Resultado completo de una recuperación."""

    estado_inicial: np.ndarray
    estado_final: np.ndarray
    barridos: int
    actualizaciones: int
    energia_inicial: float
    energia_final: float
    historial: list[PasoEnergia]
    convergio: bool
    ciclo: bool
    motivo: str


@dataclass(frozen=True)
class Clasificacion:
    """Clasificación de un estado frente a los patrones almacenados."""

    tipo: str
    nombre: str | None
    indice: int | None
    distancia_minima: int
    nombre_mas_cercano: str


@dataclass(frozen=True)
class AnalisisRed:
    """Tablas y métricas calculadas al verificar una red."""

    estabilidad: pd.DataFrame
    estabilidad_neuronas: pd.DataFrame
    energia: pd.DataFrame
    solapamiento: np.ndarray
    hamming: np.ndarray
    resumen_ortogonalidad: dict[str, object]
    capacidad: dict[str, object]


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    if camino.is_absolute():
        return camino
    return RAIZ / camino


def _validar_estado(estado: np.ndarray, n_neuronas: int) -> np.ndarray:
    """Valida un estado binario y devuelve una copia entera."""
    arreglo = np.asarray(estado)
    if arreglo.shape != (n_neuronas,):
        raise ValueError(
            f"El estado debe tener forma ({n_neuronas},), recibido {arreglo.shape}."
        )
    if not np.isin(arreglo, [-1, 1]).all():
        raise ValueError("El estado debe contener solo 1 y -1.")
    return arreglo.astype(np.int8, copy=True)


def cargar_dataset(ruta: str | Path) -> Dataset:
    """Carga y valida un dataset de patrones binarios.

    Args:
        ruta: Ruta al CSV del dataset.

    Returns:
        Dataset con nombres, patrones y forma.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato o los valores no son válidos.
    """
    camino = Path(ruta)
    if not camino.exists():
        raise FileNotFoundError(f"No existe el dataset: {camino}")

    datos = pd.read_csv(camino)
    requeridas = {"nombre", "filas", "columnas"}
    if not requeridas.issubset(datos.columns):
        raise ValueError(f"El dataset debe tener las columnas {sorted(requeridas)}.")
    if datos.empty:
        raise ValueError("El dataset no puede estar vacío.")
    if datos[["filas", "columnas"]].isna().any().any():
        raise ValueError("Las columnas filas y columnas no pueden tener nulos.")

    filas = datos["filas"].to_numpy()
    columnas = datos["columnas"].to_numpy()
    if not np.all(filas == filas[0]) or not np.all(columnas == columnas[0]):
        raise ValueError("Todas las filas deben declarar la misma forma.")

    filas_int = int(filas[0])
    columnas_int = int(columnas[0])
    if filas_int <= 0 or columnas_int <= 0:
        raise ValueError("Las filas y las columnas deben ser positivas.")

    columnas_pixeles = [columna for columna in datos.columns if columna.startswith("p")]
    esperadas = filas_int * columnas_int
    if len(columnas_pixeles) != esperadas:
        raise ValueError(
            "La cantidad de columnas de píxeles no coincide con filas × columnas: "
            f"hay {len(columnas_pixeles)} y se esperaban {esperadas}."
        )

    nombres_crudos = datos["nombre"].tolist()
    if any(pd.isna(nombre) or not str(nombre).strip() for nombre in nombres_crudos):
        raise ValueError("Los nombres de los patrones no pueden estar vacíos.")
    nombres = tuple(str(nombre) for nombre in nombres_crudos)
    if len(set(nombres)) != len(nombres):
        raise ValueError("Los nombres de los patrones deben ser únicos.")

    valores = datos[columnas_pixeles].to_numpy()
    if not np.isin(valores, [-1, 1]).all():
        raise ValueError("Los patrones deben contener solo 1 y -1.")

    return Dataset(
        nombres=nombres,
        patrones=valores.astype(np.int8),
        filas=filas_int,
        columnas=columnas_int,
    )


def calcular_pesos(patrones: np.ndarray) -> np.ndarray:
    """Calcula la matriz de pesos con la regla de Hebb.

    ``W = (1/N) Xᵀ X`` con diagonal nula y matriz simétrica.
    """
    arreglo = np.asarray(patrones, dtype=float)
    if arreglo.ndim != 2 or arreglo.shape[0] == 0 or arreglo.shape[1] == 0:
        raise ValueError("Los patrones deben ser una matriz no vacía (p, N).")
    if not np.isin(arreglo, [-1.0, 1.0]).all():
        raise ValueError("Los patrones deben contener solo 1 y -1.")

    n_neuronas = arreglo.shape[1]
    pesos = arreglo.T @ arreglo / n_neuronas
    np.fill_diagonal(pesos, 0.0)
    return (pesos + pesos.T) / 2.0


def calcular_energia(pesos: np.ndarray, estado: np.ndarray) -> float:
    """Calcula la energía de Lyapunov ``-1/2 Sᵀ W S``."""
    if pesos.ndim != 2 or pesos.shape[0] != pesos.shape[1]:
        raise ValueError("La matriz de pesos debe ser cuadrada.")
    vector = _validar_estado(estado, pesos.shape[0])
    return float(-0.5 * vector.astype(float) @ pesos @ vector.astype(float))


def calcular_campo_local(pesos: np.ndarray, estado: np.ndarray) -> np.ndarray:
    """Calcula el campo local de todas las neuronas."""
    if pesos.ndim != 2 or pesos.shape[0] != pesos.shape[1]:
        raise ValueError("La matriz de pesos debe ser cuadrada.")
    vector = _validar_estado(estado, pesos.shape[0])
    return pesos @ vector.astype(float)


def actualizar_neurona(campo: float, estado_actual: int) -> int:
    """Aplica la función de activación con empate conservador."""
    if estado_actual not in (-1, 1):
        raise ValueError("El estado de una neurona debe ser 1 o -1.")
    if campo > 0.0:
        return 1
    if campo < 0.0:
        return -1
    return int(estado_actual)


def recuperar(
    pesos: np.ndarray,
    estado_inicial: np.ndarray,
    *,
    modo: str = "asincronica",
    orden: str = "secuencial",
    max_barridos: int = 100,
    semilla: int = 0,
) -> ResultadoRecuperacion:
    """Ejecuta la dinámica hasta un estado estable, un ciclo o el límite.

    Args:
        pesos: Matriz de pesos cuadrada.
        estado_inicial: Estado binario de partida.
        modo: ``asincronica`` o ``sincronica``.
        orden: ``secuencial`` o ``aleatorio`` para el modo asincrónico.
        max_barridos: Cantidad máxima de barridos.
        semilla: Semilla para el orden aleatorio.

    Returns:
        Resultado con el estado final, la energía por barrido y el motivo de
        detención.
    """
    if pesos.ndim != 2 or pesos.shape[0] != pesos.shape[1]:
        raise ValueError("La matriz de pesos debe ser cuadrada.")
    if modo not in MODOS:
        raise ValueError(f"El modo debe ser uno de {MODOS}.")
    if orden not in ORDENES:
        raise ValueError(f"El orden debe ser uno de {ORDENES}.")
    if not isinstance(max_barridos, int) or isinstance(max_barridos, bool):
        raise ValueError("max_barridos debe ser un entero.")
    if max_barridos < 1:
        raise ValueError("max_barridos debe ser mayor o igual que 1.")
    if not isinstance(semilla, int) or isinstance(semilla, bool):
        raise ValueError("La semilla debe ser un entero.")
    if semilla < 0:
        raise ValueError("La semilla debe ser mayor o igual que 0.")

    n_neuronas = pesos.shape[0]
    estado = _validar_estado(estado_inicial, n_neuronas)
    rng = np.random.default_rng(semilla)
    energia_inicial = calcular_energia(pesos, estado)
    historial = [PasoEnergia(barrido=0, energia=energia_inicial, cambios=0)]
    vistos = {estado.tobytes()}

    convergio = False
    ciclo = False
    motivo = "max_barridos"
    barridos = 0

    for barrido in range(1, max_barridos + 1):
        if modo == "asincronica":
            if orden == "aleatorio":
                orden_actual = rng.permutation(n_neuronas)
            else:
                orden_actual = np.arange(n_neuronas)

            cambios = 0
            for indice in orden_actual:
                campo = float(pesos[indice] @ estado)
                nuevo = actualizar_neurona(campo, int(estado[indice]))
                if nuevo != estado[indice]:
                    estado[indice] = nuevo
                    cambios += 1
        else:
            campos = pesos @ estado.astype(float)
            nuevo_estado = np.where(
                campos > 0.0,
                1,
                np.where(campos < 0.0, -1, estado),
            ).astype(np.int8)
            cambios = int(np.count_nonzero(nuevo_estado != estado))
            estado = nuevo_estado

        barridos = barrido
        energia = calcular_energia(pesos, estado)
        historial.append(PasoEnergia(barrido=barrido, energia=energia, cambios=cambios))

        if cambios == 0:
            convergio = True
            motivo = "estable"
            break

        if modo == "sincronica":
            clave = estado.tobytes()
            if clave in vistos:
                ciclo = True
                motivo = "ciclo"
                break
            vistos.add(clave)

    return ResultadoRecuperacion(
        estado_inicial=_validar_estado(estado_inicial, n_neuronas),
        estado_final=estado.copy(),
        barridos=barridos,
        actualizaciones=barridos * n_neuronas,
        energia_inicial=energia_inicial,
        energia_final=historial[-1].energia,
        historial=historial,
        convergio=convergio,
        ciclo=ciclo,
        motivo=motivo,
    )


def calcular_distancias_hamming(
    estado: np.ndarray,
    patrones: np.ndarray,
) -> np.ndarray:
    """Calcula la distancia de Hamming del estado a cada patrón."""
    arreglo = np.asarray(patrones, dtype=int)
    if arreglo.ndim != 2 or arreglo.shape[0] == 0:
        raise ValueError("Los patrones deben ser una matriz no vacía (p, N).")
    vector = _validar_estado(estado, arreglo.shape[1])
    productos = arreglo @ vector.astype(int)
    return np.rint((arreglo.shape[1] - productos) / 2.0).astype(int)


def clasificar_estado(
    estado: np.ndarray,
    patrones: np.ndarray,
    nombres: Sequence[str],
    nombre_origen: str | None = None,
) -> Clasificacion:
    """Clasifica un estado estable frente a los patrones almacenados."""
    arreglo = np.asarray(patrones, dtype=int)
    if arreglo.ndim != 2 or arreglo.shape[0] == 0:
        raise ValueError("Los patrones deben ser una matriz no vacía (p, N).")
    if len(nombres) != arreglo.shape[0]:
        raise ValueError("La cantidad de nombres no coincide con los patrones.")

    distancias = calcular_distancias_hamming(estado, arreglo)
    indice = int(np.argmin(distancias))
    distancia = int(distancias[indice])
    nombre_cercano = str(nombres[indice])

    if distancia == 0:
        if nombre_origen is None or nombre_origen == nombre_cercano:
            tipo = "almacenado_correcto"
        else:
            tipo = "almacenado_incorrecto"
        return Clasificacion(
            tipo=tipo,
            nombre=nombre_cercano,
            indice=indice,
            distancia_minima=0,
            nombre_mas_cercano=nombre_cercano,
        )

    return Clasificacion(
        tipo="espurio",
        nombre=None,
        indice=None,
        distancia_minima=distancia,
        nombre_mas_cercano=nombre_cercano,
    )


def calcular_solapamiento(patrones: np.ndarray) -> np.ndarray:
    """Calcula la matriz de solapamiento ``X Xᵀ / N``."""
    arreglo = np.asarray(patrones, dtype=float)
    if arreglo.ndim != 2 or arreglo.shape[0] == 0 or arreglo.shape[1] == 0:
        raise ValueError("Los patrones deben ser una matriz no vacía (p, N).")
    if not np.isin(arreglo, [-1.0, 1.0]).all():
        raise ValueError("Los patrones deben contener solo 1 y -1.")
    return arreglo @ arreglo.T / arreglo.shape[1]


def calcular_hamming(patrones: np.ndarray) -> np.ndarray:
    """Calcula la matriz de distancias de Hamming entre patrones."""
    arreglo = np.asarray(patrones, dtype=int)
    if arreglo.ndim != 2 or arreglo.shape[0] == 0 or arreglo.shape[1] == 0:
        raise ValueError("Los patrones deben ser una matriz no vacía (p, N).")
    if not np.isin(arreglo, [-1, 1]).all():
        raise ValueError("Los patrones deben contener solo 1 y -1.")
    productos = arreglo @ arreglo.T
    return np.rint((arreglo.shape[1] - productos) / 2.0).astype(int)


def resumir_ortogonalidad(
    solapamiento: np.ndarray,
    hamming: np.ndarray,
) -> dict[str, object]:
    """Resume la ortogonalidad de los pares de patrones."""
    if solapamiento.ndim != 2 or solapamiento.shape[0] != solapamiento.shape[1]:
        raise ValueError("La matriz de solapamiento debe ser cuadrada.")
    if hamming.shape != solapamiento.shape:
        raise ValueError("Las matrices de solapamiento y Hamming deben coincidir.")

    cantidad = solapamiento.shape[0]
    if cantidad < 2:
        return {
            "solapamiento_abs_medio": 0.0,
            "solapamiento_abs_max": 0.0,
            "hamming_min": 0,
            "hamming_medio": 0.0,
            "pares_ortogonales": 0,
        }

    indices = np.triu_indices(cantidad, k=1)
    solapamientos = np.abs(solapamiento[indices])
    distancias = hamming[indices]
    return {
        "solapamiento_abs_medio": float(solapamientos.mean()),
        "solapamiento_abs_max": float(solapamientos.max()),
        "hamming_min": int(distancias.min()),
        "hamming_medio": float(distancias.mean()),
        "pares_ortogonales": int(np.isclose(solapamientos, 0.0, atol=1e-12).sum()),
    }


def calcular_capacidad(n_neuronas: int, n_patrones: int) -> dict[str, object]:
    """Calcula la carga de la red y el límite teórico del 15 por ciento."""
    if n_neuronas <= 0:
        raise ValueError("La cantidad de neuronas debe ser positiva.")
    if n_patrones <= 0:
        raise ValueError("La cantidad de patrones debe ser positiva.")
    limite = 0.15 * n_neuronas
    return {
        "neuronas": int(n_neuronas),
        "patrones": int(n_patrones),
        "carga": n_patrones / n_neuronas,
        "limite_teorico": limite,
        "excede_limite": bool(n_patrones > limite),
    }


def calcular_margenes_estabilidad(
    pesos: np.ndarray,
    patron: np.ndarray,
) -> np.ndarray:
    """Calcula el margen ``ξ_i h_i`` de cada neurona."""
    campos = calcular_campo_local(pesos, patron)
    vector = _validar_estado(patron, pesos.shape[0])
    return vector.astype(float) * campos


def analizar_red(
    dataset: Dataset,
    pesos: np.ndarray,
    *,
    modo: str,
    orden: str,
    max_barridos: int,
    semilla: int,
) -> AnalisisRed:
    """Verifica los patrones almacenados y calcula las tablas del análisis."""
    if pesos.shape != (dataset.n_neuronas, dataset.n_neuronas):
        raise ValueError("La forma de la matriz de pesos no coincide con el dataset.")

    filas_estabilidad: list[dict[str, object]] = []
    filas_neuronas: list[dict[str, object]] = []
    filas_energia: list[dict[str, object]] = []

    for nombre, patron in zip(dataset.nombres, dataset.patrones, strict=True):
        resultado = recuperar(
            pesos,
            patron,
            modo=modo,
            orden=orden,
            max_barridos=max_barridos,
            semilla=semilla,
        )
        margenes = calcular_margenes_estabilidad(pesos, patron)
        campos = calcular_campo_local(pesos, patron)
        punto_fijo = bool(np.all(margenes >= 0.0))
        clasificacion = clasificar_estado(
            resultado.estado_final,
            dataset.patrones,
            dataset.nombres,
            nombre_origen=nombre,
        )

        if resultado.ciclo:
            tipo_resultado = "ciclo"
            nombre_final = ""
        elif not resultado.convergio:
            tipo_resultado = "no_converge"
            nombre_final = ""
        else:
            tipo_resultado = clasificacion.tipo
            nombre_final = clasificacion.nombre or ""

        filas_estabilidad.append(
            {
                "nombre": nombre,
                "punto_fijo": punto_fijo,
                "convergio": resultado.convergio,
                "ciclo": resultado.ciclo,
                "tipo_resultado": tipo_resultado,
                "nombre_final": nombre_final,
                "distancia_hamming": clasificacion.distancia_minima,
                "margen_minimo": float(margenes.min()),
                "neuronas_inestables": int((margenes < 0.0).sum()),
                "neuronas_empate": int((margenes == 0.0).sum()),
                "barridos": resultado.barridos,
                "actualizaciones": resultado.actualizaciones,
                "energia_inicial": resultado.energia_inicial,
                "energia_final": resultado.energia_final,
            }
        )

        for indice, (xi, campo, margen) in enumerate(
            zip(patron, campos, margenes, strict=True)
        ):
            filas_neuronas.append(
                {
                    "nombre": nombre,
                    "neurona": indice,
                    "xi": int(xi),
                    "campo_local": float(campo),
                    "margen": float(margen),
                }
            )

        for paso in resultado.historial:
            filas_energia.append(
                {
                    "nombre": nombre,
                    "barrido": paso.barrido,
                    "energia": paso.energia,
                    "cambios": paso.cambios,
                }
            )

    solapamiento = calcular_solapamiento(dataset.patrones)
    hamming = calcular_hamming(dataset.patrones)

    return AnalisisRed(
        estabilidad=pd.DataFrame(filas_estabilidad),
        estabilidad_neuronas=pd.DataFrame(filas_neuronas),
        energia=pd.DataFrame(filas_energia),
        solapamiento=solapamiento,
        hamming=hamming,
        resumen_ortogonalidad=resumir_ortogonalidad(solapamiento, hamming),
        capacidad=calcular_capacidad(dataset.n_neuronas, dataset.n_patrones),
    )


def escribir_dataset(dataset: Dataset, ruta: str | Path) -> None:
    """Guarda un dataset en el formato CSV del proyecto."""
    cantidad_pixeles = dataset.n_neuronas
    ancho = max(4, len(str(cantidad_pixeles - 1)))
    columnas_pixeles = [f"p{indice:0{ancho}d}" for indice in range(cantidad_pixeles)]
    registros: list[dict[str, object]] = []
    for nombre, patron in zip(dataset.nombres, dataset.patrones, strict=True):
        registro: dict[str, object] = {
            "nombre": nombre,
            "filas": dataset.filas,
            "columnas": dataset.columnas,
        }
        registro.update(
            {
                columna: int(valor)
                for columna, valor in zip(
                    columnas_pixeles, patron.reshape(-1), strict=True
                )
            }
        )
        registros.append(registro)

    camino = Path(ruta)
    camino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(camino, index=False)


def guardar_red(
    directorio: str | Path,
    dataset: Dataset,
    pesos: np.ndarray,
) -> None:
    """Guarda pesos, patrones y metadatos de la red."""
    camino = Path(directorio)
    camino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        camino / "pesos.npz",
        pesos=pesos,
        patrones=dataset.patrones,
        nombres=np.asarray(dataset.nombres, dtype=str),
        filas=dataset.filas,
        columnas=dataset.columnas,
    )
    escribir_dataset(dataset, camino / "patrones.csv")


def cargar_red(directorio: str | Path) -> RedHopfield:
    """Carga una red guardada con :func:`guardar_red`."""
    ruta = Path(directorio) / "pesos.npz"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la red: {ruta}")

    with np.load(ruta, allow_pickle=False) as datos:
        pesos = np.asarray(datos["pesos"], dtype=float)
        patrones = np.asarray(datos["patrones"], dtype=np.int8)
        nombres = tuple(str(nombre) for nombre in datos["nombres"].tolist())
        filas = int(datos["filas"])
        columnas = int(datos["columnas"])

    if pesos.shape != (filas * columnas, filas * columnas):
        raise ValueError("La forma de los pesos no coincide con filas × columnas.")
    if patrones.shape != (len(nombres), filas * columnas):
        raise ValueError("La forma de los patrones no coincide con los metadatos.")

    return RedHopfield(
        pesos=pesos,
        patrones=patrones,
        nombres=nombres,
        filas=filas,
        columnas=columnas,
    )


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
    raise TypeError(f"No se puede escribir en TOML: {valor!r}")


def guardar_config(config: dict, ruta: str | Path) -> None:
    """Guarda una configuración de secciones planas en formato TOML."""
    camino = Path(ruta)
    camino.parent.mkdir(parents=True, exist_ok=True)
    lineas: list[str] = []
    for seccion in ("datos", "modelo", "salida"):
        if seccion not in config:
            continue
        lineas.append(f"[{seccion}]")
        for clave, valor in config[seccion].items():
            lineas.append(f"{clave} = {_formatear_valor_toml(valor)}")
        lineas.append("")
    camino.write_text("\n".join(lineas).rstrip() + "\n", encoding="utf-8")


def cargar_config(ruta: str | Path) -> dict:
    """Carga un archivo TOML de configuración."""
    camino = Path(ruta)
    if not camino.exists():
        raise FileNotFoundError(f"No existe la configuración: {camino}")
    with camino.open("rb") as archivo:
        return tomllib.load(archivo)


def validar_config(config: dict) -> None:
    """Valida las secciones y los valores principales de la configuración."""
    if not isinstance(config, dict):
        raise ValueError("La configuración debe ser un diccionario.")
    for seccion in ("datos", "modelo", "salida"):
        if seccion not in config or not isinstance(config[seccion], dict):
            raise ValueError(f"Falta la sección obligatoria: {seccion}.")

    ruta = config["datos"].get("ruta")
    if not isinstance(ruta, str) or not ruta:
        raise ValueError("datos.ruta debe ser un texto no vacío.")

    modelo = config["modelo"]
    if modelo.get("modo") not in MODOS:
        raise ValueError(f"modelo.modo debe ser uno de {MODOS}.")
    if modelo.get("orden") not in ORDENES:
        raise ValueError(f"modelo.orden debe ser uno de {ORDENES}.")

    semilla = modelo.get("semilla")
    if not isinstance(semilla, int) or isinstance(semilla, bool) or semilla < 0:
        raise ValueError("modelo.semilla debe ser un entero mayor o igual que 0.")

    max_barridos = modelo.get("max_barridos")
    if (
        not isinstance(max_barridos, int)
        or isinstance(max_barridos, bool)
        or max_barridos < 1
    ):
        raise ValueError("modelo.max_barridos debe ser un entero mayor o igual que 1.")

    directorio = config["salida"].get("directorio")
    if not isinstance(directorio, str):
        raise ValueError("salida.directorio debe ser un texto.")


def resolver_config(cli: argparse.Namespace) -> dict:
    """Resuelve la configuración con precedencia CLI > TOML > valores por defecto."""
    ruta_config = Path(getattr(cli, "config", RUTA_CONFIG))
    if not ruta_config.is_absolute():
        ruta_config = RAIZ / ruta_config
    if not ruta_config.exists():
        raise FileNotFoundError(f"No existe la configuración: {ruta_config}")

    config = _fusionar(CONFIG_POR_DEFECTO, cargar_config(ruta_config))

    datos_cli = getattr(cli, "datos", None)
    if datos_cli is not None:
        config["datos"]["ruta"] = str(datos_cli)

    salida_cli = getattr(cli, "salida", None)
    if salida_cli is not None:
        config["salida"]["directorio"] = str(salida_cli)

    for clave in ("modo", "orden", "semilla", "max_barridos"):
        valor = getattr(cli, clave, None)
        if valor is not None:
            config["modelo"][clave] = valor

    ruta_datos = resolver_ruta(config["datos"]["ruta"])
    config["datos"]["ruta"] = str(ruta_datos)

    directorio = config["salida"].get("directorio", "")
    if not directorio:
        directorio = str(RAIZ / "resultados" / "ej2" / "hopfield" / ruta_datos.stem)
    else:
        directorio = str(resolver_ruta(directorio))
    config["salida"]["directorio"] = directorio

    validar_config(config)
    return config


def construir_resumen(
    config: dict,
    dataset: Dataset,
    analisis: AnalisisRed,
) -> str:
    """Construye el resumen textual de una corrida."""
    puntos_fijos = int(analisis.estabilidad["punto_fijo"].sum())
    tipos = analisis.estabilidad["tipo_resultado"].value_counts().to_dict()
    lineas = [
        "Red de Hopfield — Ejercicio 2.2",
        f"Dataset: {config['datos']['ruta']}",
        f"Patrones: {', '.join(dataset.nombres)}",
        f"Filas: {dataset.filas}",
        f"Columnas: {dataset.columnas}",
        f"Neuronas: {dataset.n_neuronas}",
        f"Cantidad de patrones: {dataset.n_patrones}",
        f"Modo: {config['modelo']['modo']}",
        f"Orden: {config['modelo']['orden']}",
        f"Semilla: {config['modelo']['semilla']}",
        f"Máximo de barridos: {config['modelo']['max_barridos']}",
        "",
        "Capacidad:",
        f"  carga p/N = {analisis.capacidad['carga']:.6f}",
        f"  límite teórico = {analisis.capacidad['limite_teorico']:.6f}",
        f"  excede límite = {analisis.capacidad['excede_limite']}",
        "",
        "Ortogonalidad:",
        f"  solapamiento absoluto medio = "
        f"{analisis.resumen_ortogonalidad['solapamiento_abs_medio']:.6f}",
        f"  solapamiento absoluto máximo = "
        f"{analisis.resumen_ortogonalidad['solapamiento_abs_max']:.6f}",
        f"  Hamming mínimo = {analisis.resumen_ortogonalidad['hamming_min']}",
        f"  Hamming medio = {analisis.resumen_ortogonalidad['hamming_medio']:.6f}",
        f"  pares ortogonales = {analisis.resumen_ortogonalidad['pares_ortogonales']}",
        "",
        f"Puntos fijos: {puntos_fijos} de {dataset.n_patrones}",
        f"Tipos de resultado: {tipos}",
        "",
        "Detalle por patrón:",
    ]
    for _, fila in analisis.estabilidad.iterrows():
        lineas.append(
            f"  {fila['nombre']}: {fila['tipo_resultado']}, "
            f"punto_fijo={fila['punto_fijo']}, "
            f"barridos={fila['barridos']}, "
            f"energía_final={fila['energia_final']:.6f}, "
            f"margen_mínimo={fila['margen_minimo']:.6f}"
        )
    return "\n".join(lineas) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el entrenamiento y guarda la red y sus tablas."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena una red de Hopfield con un dataset de patrones binarios "
            "y guarda la red y las tablas de análisis."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG,
        help="Archivo TOML de configuración.",
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=None,
        help="Dataset CSV de patrones; pisa datos.ruta del TOML.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Directorio de salida; pisa salida.directorio del TOML.",
    )
    parser.add_argument(
        "--modo",
        choices=MODOS,
        default=None,
        help="Dinámica de actualización.",
    )
    parser.add_argument(
        "--orden",
        choices=ORDENES,
        default=None,
        help="Orden de actualización en modo asincrónico.",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=None,
        help="Semilla para el orden aleatorio.",
    )
    parser.add_argument(
        "--max-barridos",
        type=int,
        default=None,
        help="Cantidad máxima de barridos.",
    )
    argumentos = parser.parse_args(argv)

    config = resolver_config(argumentos)
    dataset = cargar_dataset(config["datos"]["ruta"])
    pesos = calcular_pesos(dataset.patrones)
    analisis = analizar_red(
        dataset,
        pesos,
        modo=config["modelo"]["modo"],
        orden=config["modelo"]["orden"],
        max_barridos=config["modelo"]["max_barridos"],
        semilla=config["modelo"]["semilla"],
    )

    directorio = Path(config["salida"]["directorio"])
    directorio.mkdir(parents=True, exist_ok=True)

    guardar_config(config, directorio / "config_usada.toml")
    guardar_red(directorio, dataset, pesos)
    analisis.estabilidad.to_csv(directorio / "estabilidad.csv", index=False)
    analisis.estabilidad_neuronas.to_csv(
        directorio / "estabilidad_neuronas.csv", index=False
    )
    analisis.energia.to_csv(directorio / "energia_entrenamiento.csv", index=False)
    pd.DataFrame(
        analisis.solapamiento,
        index=dataset.nombres,
        columns=dataset.nombres,
    ).to_csv(directorio / "solapamiento.csv", index_label="nombre")
    pd.DataFrame(
        analisis.hamming,
        index=dataset.nombres,
        columns=dataset.nombres,
    ).to_csv(directorio / "hamming.csv", index_label="nombre")
    pd.DataFrame([analisis.resumen_ortogonalidad]).to_csv(
        directorio / "ortogonalidad_resumen.csv", index=False
    )
    pd.DataFrame([analisis.capacidad]).to_csv(directorio / "capacidad.csv", index=False)
    (directorio / "resumen.txt").write_text(
        construir_resumen(config, dataset, analisis),
        encoding="utf-8",
    )

    print(f"Red guardada en {directorio}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
