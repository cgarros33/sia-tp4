#!/usr/bin/env python3
"""Runner de una red de Hopfield entrenada (TP4, ejercicio 2.2).

El script carga una red guardada por ``hopfield.py``, aplica ruido opcional a
los patrones de consulta, ejecuta la recuperación y guarda las tablas de
resultados. El modo gráfico imprime cada estado final como texto.

Uso:
    python3 src/ej2/hopfield-runner.py \
        --red resultados/ej2/hopfield/letras_5x5 \
        --datos datos/hopfield/letras_5x5.csv \
        --ruido 0.4 --semilla 0 --repeticiones 3 --graphic
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

RUTA_RED_POR_DEFECTO = Path("resultados") / "ej2" / "hopfield"


def _cargar_motor() -> ModuleType:
    """Carga el motor de Hopfield que está en la misma carpeta."""
    ruta = Path(__file__).resolve().parent / "hopfield.py"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el motor de Hopfield: {ruta}")
    especificacion = importlib.util.spec_from_file_location(
        "hopfield_motor_runner", ruta
    )
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


motor = _cargar_motor()


@dataclass(frozen=True)
class ClasificacionConsulta:
    """Clasificación de una consulta después de la recuperación."""

    tipo: str
    nombre: str | None
    distancia_minima: int
    nombre_mas_cercano: str


def cargar_config_red(directorio: str | Path) -> dict:
    """Carga y valida la configuración guardada con la red."""
    ruta = Path(directorio) / "config_usada.toml"
    config = motor.cargar_config(ruta)
    motor.validar_config(config)
    return config


def aplicar_ruido(
    patron: np.ndarray,
    ruido: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Invierte exactamente ``round(ruido * N)`` bits distintos del patrón."""
    vector = np.asarray(patron)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("El patrón debe ser un vector no vacío.")
    if not np.isin(vector, [-1, 1]).all():
        raise ValueError("El patrón debe contener solo 1 y -1.")
    if not isinstance(ruido, (int, float)) or isinstance(ruido, bool):
        raise ValueError("El ruido debe ser un número entre 0 y 1.")
    if not 0.0 <= float(ruido) <= 1.0:
        raise ValueError("El ruido debe estar entre 0 y 1.")

    resultado = vector.astype(np.int8, copy=True)
    cantidad = int(round(float(ruido) * vector.size))
    if cantidad > 0:
        indices = rng.choice(vector.size, size=cantidad, replace=False)
        resultado[indices] *= -1
    return resultado


def renderizar_matriz(estado: np.ndarray, filas: int, columnas: int) -> str:
    """Renderiza un estado como texto con ``*`` para 1 y espacio para -1."""
    if filas <= 0 or columnas <= 0:
        raise ValueError("Las filas y las columnas deben ser positivas.")
    vector = np.asarray(estado)
    if vector.size != filas * columnas:
        raise ValueError(
            f"El estado tiene {vector.size} valores y se esperaban {filas * columnas}."
        )
    if not np.isin(vector, [-1, 1]).all():
        raise ValueError("El estado debe contener solo 1 y -1.")

    matriz = vector.reshape(filas, columnas)
    return "\n".join(
        "".join("*" if valor == 1 else " " for valor in fila) for fila in matriz
    )


def clasificar_consulta(
    resultado: Any,
    red: Any,
    nombre_origen: str,
) -> ClasificacionConsulta:
    """Clasifica el resultado de una consulta, incluyendo ciclos y no convergencia."""
    clasificacion = motor.clasificar_estado(
        resultado.estado_final,
        red.patrones,
        red.nombres,
        nombre_origen=nombre_origen,
    )
    if resultado.ciclo:
        return ClasificacionConsulta(
            tipo="ciclo",
            nombre=None,
            distancia_minima=clasificacion.distancia_minima,
            nombre_mas_cercano=clasificacion.nombre_mas_cercano,
        )
    if not resultado.convergio:
        return ClasificacionConsulta(
            tipo="no_converge",
            nombre=None,
            distancia_minima=clasificacion.distancia_minima,
            nombre_mas_cercano=clasificacion.nombre_mas_cercano,
        )
    return ClasificacionConsulta(
        tipo=clasificacion.tipo,
        nombre=clasificacion.nombre,
        distancia_minima=clasificacion.distancia_minima,
        nombre_mas_cercano=clasificacion.nombre_mas_cercano,
    )


def _dataframe_estados(
    registros: list[tuple[str, np.ndarray]],
    filas: int,
    columnas: int,
) -> pd.DataFrame:
    """Convierte una lista de estados en una tabla CSV ancha."""
    cantidad_pixeles = filas * columnas
    ancho = max(4, len(str(cantidad_pixeles - 1)))
    columnas_pixeles = [f"p{indice:0{ancho}d}" for indice in range(cantidad_pixeles)]

    filas_tabla: list[dict[str, object]] = []
    for consulta, estado in registros:
        vector = np.asarray(estado)
        if vector.size != cantidad_pixeles:
            raise ValueError(
                f"El estado {consulta} tiene {vector.size} valores y se "
                f"esperaban {cantidad_pixeles}."
            )
        registro: dict[str, object] = {
            "consulta": consulta,
            "filas": filas,
            "columnas": columnas,
        }
        registro.update(
            {
                columna: int(valor)
                for columna, valor in zip(
                    columnas_pixeles, vector.reshape(-1), strict=True
                )
            }
        )
        filas_tabla.append(registro)

    return pd.DataFrame(
        filas_tabla,
        columns=["consulta", "filas", "columnas", *columnas_pixeles],
    )


def construir_resumen(
    red_dir: Path,
    ruta_consulta: Path,
    config: dict,
    consultas: pd.DataFrame,
    ruido: float,
    semilla: int,
    repeticiones: int,
) -> str:
    """Construye el resumen textual de una corrida del runner."""
    tipos = consultas["tipo_resultado"].value_counts().to_dict()
    lineas = [
        "Consultas de Hopfield — Ejercicio 2.2",
        f"Red: {red_dir}",
        f"Dataset de consulta: {ruta_consulta}",
        f"Modo: {config['modelo']['modo']}",
        f"Orden: {config['modelo']['orden']}",
        f"Máximo de barridos: {config['modelo']['max_barridos']}",
        f"Ruido: {ruido}",
        f"Semilla: {semilla}",
        f"Repeticiones: {repeticiones}",
        f"Cantidad de consultas: {len(consultas)}",
        f"Tipos de resultado: {tipos}",
        "",
        "Detalle por consulta:",
    ]
    for _, fila in consultas.iterrows():
        lineas.append(
            f"  {fila['consulta']}: {fila['tipo_resultado']}, "
            f"nombre_final={fila['nombre_final'] or '-'}, "
            f"distancia_hamming={fila['distancia_hamming']}, "
            f"barridos={fila['barridos']}, "
            f"energía_final={fila['energia_final']:.6f}"
        )
    return "\n".join(lineas) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta las consultas y guarda sus resultados."""
    parser = argparse.ArgumentParser(
        description=(
            "Consulta una red de Hopfield entrenada y guarda los resultados "
            "de recuperación."
        )
    )
    parser.add_argument(
        "--red",
        type=Path,
        required=True,
        help="Directorio de la red entrenada.",
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=None,
        help="Dataset de consulta; por defecto usa patrones.csv de la red.",
    )
    parser.add_argument(
        "--ruido",
        type=float,
        default=0.0,
        help="Fracción de bits a invertir, entre 0 y 1.",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=0,
        help="Semilla para el ruido aleatorio.",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=1,
        help="Cantidad de versiones ruidosas por patrón.",
    )
    parser.add_argument(
        "--graphic",
        "--grafico",
        dest="graphic",
        action="store_true",
        help="Imprime cada estado final como texto.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Directorio de salida; por defecto <red>/consultas.",
    )
    argumentos = parser.parse_args(argv)

    if not 0.0 <= argumentos.ruido <= 1.0:
        parser.error("--ruido debe estar entre 0 y 1.")
    if argumentos.repeticiones < 1:
        parser.error("--repeticiones debe ser mayor o igual que 1.")
    if argumentos.semilla < 0:
        parser.error("--semilla debe ser mayor o igual que 0.")

    red_dir = motor.resolver_ruta(argumentos.red)
    config = cargar_config_red(red_dir)
    red = motor.cargar_red(red_dir)

    ruta_consulta = (
        motor.resolver_ruta(argumentos.datos)
        if argumentos.datos is not None
        else red_dir / "patrones.csv"
    )
    consulta = motor.cargar_dataset(ruta_consulta)

    if (consulta.filas, consulta.columnas) != (red.filas, red.columnas):
        raise ValueError(
            "Las dimensiones de la consulta no coinciden con las de la red: "
            f"consulta {consulta.filas}×{consulta.columnas}, "
            f"red {red.filas}×{red.columnas}."
        )

    salida = (
        motor.resolver_ruta(argumentos.salida)
        if argumentos.salida is not None
        else red_dir / "consultas"
    )
    salida.mkdir(parents=True, exist_ok=True)

    modo = config["modelo"]["modo"]
    orden = config["modelo"]["orden"]
    max_barridos = config["modelo"]["max_barridos"]
    semilla_red = config["modelo"]["semilla"]
    rng = np.random.default_rng(argumentos.semilla)

    registros: list[dict[str, object]] = []
    energias: list[dict[str, object]] = []
    estados_iniciales: list[tuple[str, np.ndarray]] = []
    estados_finales: list[tuple[str, np.ndarray]] = []

    for indice_patron, (nombre, patron) in enumerate(
        zip(consulta.nombres, consulta.patrones, strict=True)
    ):
        for repeticion in range(argumentos.repeticiones):
            estado_inicial = aplicar_ruido(patron, argumentos.ruido, rng)
            resultado = motor.recuperar(
                red.pesos,
                estado_inicial,
                modo=modo,
                orden=orden,
                max_barridos=max_barridos,
                semilla=semilla_red,
            )
            clasificacion = clasificar_consulta(resultado, red, nombre)
            nombre_consulta = (
                nombre if argumentos.repeticiones == 1 else f"{nombre}_rep{repeticion}"
            )

            registros.append(
                {
                    "consulta": nombre_consulta,
                    "nombre_origen": nombre,
                    "indice_patron": indice_patron,
                    "repeticion": repeticion,
                    "ruido": argumentos.ruido,
                    "semilla": argumentos.semilla,
                    "tipo_resultado": clasificacion.tipo,
                    "nombre_final": clasificacion.nombre or "",
                    "distancia_hamming": clasificacion.distancia_minima,
                    "nombre_mas_cercano": clasificacion.nombre_mas_cercano,
                    "barridos": resultado.barridos,
                    "actualizaciones": resultado.actualizaciones,
                    "energia_inicial": resultado.energia_inicial,
                    "energia_final": resultado.energia_final,
                    "convergio": resultado.convergio,
                    "ciclo": resultado.ciclo,
                    "motivo": resultado.motivo,
                }
            )

            for paso in resultado.historial:
                energias.append(
                    {
                        "consulta": nombre_consulta,
                        "barrido": paso.barrido,
                        "energia": paso.energia,
                        "cambios": paso.cambios,
                    }
                )

            estados_iniciales.append((nombre_consulta, estado_inicial))
            estados_finales.append((nombre_consulta, resultado.estado_final))

            if argumentos.graphic:
                print(f"consulta: {nombre_consulta}")
                print(
                    f"origen: {nombre} | tipo: {clasificacion.tipo} | "
                    f"nombre: {clasificacion.nombre or '-'} | "
                    f"barridos: {resultado.barridos} | "
                    f"energía: {resultado.energia_final:.6f}"
                )
                print(
                    renderizar_matriz(
                        resultado.estado_final,
                        red.filas,
                        red.columnas,
                    )
                )
                print()

    tabla_consultas = pd.DataFrame(registros)
    tabla_energia = pd.DataFrame(energias)
    tabla_iniciales = _dataframe_estados(estados_iniciales, red.filas, red.columnas)
    tabla_finales = _dataframe_estados(estados_finales, red.filas, red.columnas)

    tabla_consultas.to_csv(salida / "consultas.csv", index=False)
    tabla_energia.to_csv(salida / "energia.csv", index=False)
    tabla_iniciales.to_csv(salida / "estados_iniciales.csv", index=False)
    tabla_finales.to_csv(salida / "estados_finales.csv", index=False)
    (salida / "resumen.txt").write_text(
        construir_resumen(
            red_dir,
            ruta_consulta,
            config,
            tabla_consultas,
            argumentos.ruido,
            argumentos.semilla,
            argumentos.repeticiones,
        ),
        encoding="utf-8",
    )

    print(f"Consultas guardadas en {salida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
