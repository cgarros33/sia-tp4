#!/usr/bin/env python3
"""Barrido secuencial de hiperparámetros para la red de Kohonen.

El script lee una configuración base y una configuración de barrido, entrena
cada candidato con las semillas indicadas, elige el mejor valor de cada paso y
guarda los resultados y las figuras en ``resultados/ej1/kohonen_sweep/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CONFIG_BARRIDO = Path(__file__).resolve().with_name("kohonen-sweep.toml")

PARAMETROS_VALIDOS = (
    "grilla",
    "topologia",
    "distancia",
    "inicializacion",
    "radio_inicial_relativo",
    "decaimiento_radio",
    "eta_inicial",
    "decaimiento_eta",
    "iteraciones",
)
CRITERIOS_VALIDOS = ("minimo", "codo")
OPCIONES_FIGURAS = ("a", "b", "c")


def _cargar_kohonen():
    """Carga el motor de Kohonen desde el archivo vecino."""
    ruta = Path(__file__).resolve().with_name("kohonen.py")
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el motor de Kohonen: {ruta}")
    especificacion = importlib.util.spec_from_file_location(
        "kohonen_para_barrido",
        ruta,
    )
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar el motor de Kohonen: {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


kohonen = _cargar_kohonen()


@dataclass
class ResultadoPaso:
    """Resultado de barrer un parámetro o un par de parámetros."""

    parametro: str
    resultados: pd.DataFrame
    resumen: pd.DataFrame
    mejor_valor: object
    qe_media: float


def cargar_config_barrido(ruta: Path) -> dict:
    """Lee y valida la configuración del barrido."""
    with Path(ruta).open("rb") as archivo:
        config = tomllib.load(archivo)
    validar_config_barrido(config)
    return config


def validar_config_barrido(config: dict) -> None:
    """Valida el orden, los valores y las secciones opcionales del barrido."""
    if not isinstance(config, dict):
        raise ValueError("La configuración del barrido debe ser un diccionario.")

    barrido = config.get("barrido")
    if not isinstance(barrido, dict):
        raise ValueError("Falta la sección [barrido].")

    orden = barrido.get("orden")
    if not isinstance(orden, list) or not orden:
        raise ValueError("barrido.orden debe ser una lista no vacía.")
    if len(set(orden)) != len(orden):
        raise ValueError("barrido.orden no puede tener parámetros repetidos.")

    for parametro in orden:
        if parametro not in PARAMETROS_VALIDOS:
            raise ValueError(f"Parámetro de barrido desconocido: {parametro}")

    semillas = barrido.get("semillas")
    if (
        not isinstance(semillas, list)
        or not semillas
        or not all(
            isinstance(semilla, int) and not isinstance(semilla, bool)
            for semilla in semillas
        )
    ):
        raise ValueError("barrido.semillas debe ser una lista de enteros.")

    criterio = barrido.get("criterio_grilla", "minimo")
    if criterio not in CRITERIOS_VALIDOS:
        raise ValueError(f"Criterio de grilla inválido: {criterio}")

    tolerancia = barrido.get("tolerancia_codo", 0.05)
    if not isinstance(tolerancia, (int, float)) or tolerancia < 0.0:
        raise ValueError("barrido.tolerancia_codo debe ser no negativa.")

    valores = config.get("valores")
    if not isinstance(valores, dict):
        raise ValueError("Falta la sección [valores].")

    for parametro in orden:
        candidatos = valores.get(parametro)
        if not isinstance(candidatos, list) or not candidatos:
            raise ValueError(f"valores.{parametro} debe ser una lista no vacía.")

    acoplamientos = config.get("acoplamientos", {})
    if not isinstance(acoplamientos, dict):
        raise ValueError("La sección [acoplamientos] debe ser una tabla.")
    pares = acoplamientos.get("pares", [])
    if not isinstance(pares, list):
        raise ValueError("acoplamientos.pares debe ser una lista.")
    for par in pares:
        if (
            not isinstance(par, list)
            or len(par) != 2
            or any(parametro not in PARAMETROS_VALIDOS for parametro in par)
        ):
            raise ValueError("Cada acoplamiento debe ser un par de parámetros válidos.")
        for parametro in par:
            if not isinstance(valores.get(parametro), list) or not valores[parametro]:
                raise ValueError(
                    f"Falta la lista de valores para el acoplamiento: {parametro}"
                )
    if not isinstance(acoplamientos.get("adoptar_solo_si_mejora", True), bool):
        raise ValueError("acoplamientos.adoptar_solo_si_mejora debe ser booleano.")

    figuras = config.get("figuras", {})
    if not isinstance(figuras, dict):
        raise ValueError("La sección [figuras] debe ser una tabla.")
    opciones = figuras.get("opciones", list(OPCIONES_FIGURAS))
    if (
        not isinstance(opciones, list)
        or not opciones
        or any(opcion not in OPCIONES_FIGURAS for opcion in opciones)
    ):
        raise ValueError(
            f"figuras.opciones debe ser una sublista de {OPCIONES_FIGURAS}."
        )

    if "base" in config:
        base = config["base"]
        if not isinstance(base, dict) or not isinstance(base.get("archivo"), str):
            raise ValueError("base.archivo debe ser un texto.")

    if "salida" in config:
        salida = config["salida"]
        if not isinstance(salida, dict):
            raise ValueError("La sección [salida] debe ser una tabla.")
        if not isinstance(salida.get("directorio"), str):
            raise ValueError("salida.directorio debe ser un texto.")
        if "directorio_final" in salida and not isinstance(
            salida["directorio_final"], str
        ):
            raise ValueError("salida.directorio_final debe ser un texto.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos del barrido."""
    parser = argparse.ArgumentParser(
        description="Barrido secuencial de hiperparámetros de Kohonen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_BARRIDO,
        help="Archivo TOML de configuración del barrido.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Directorio donde guardar los resultados del barrido.",
    )
    parser.add_argument(
        "--semillas",
        type=kohonen.lista_semillas,
        default=None,
        help="Semillas separadas por comas.",
    )
    return parser.parse_args(argv)


def agregar_resultados(resultados: pd.DataFrame) -> pd.DataFrame:
    """Agrega las métricas por valor o por par de valores."""
    columnas_grupo = [
        columna
        for columna in resultados.columns
        if columna not in {"semilla", "qe", "te", "unidades_muertas", "ocupacion"}
    ]
    agregado = resultados.groupby(
        columnas_grupo,
        dropna=False,
        sort=False,
    ).agg(
        qe_media=("qe", "mean"),
        qe_desvio=(
            "qe",
            lambda serie: float(np.std(serie, ddof=1)) if len(serie) > 1 else 0.0,
        ),
        te_media=("te", "mean"),
        te_desvio=(
            "te",
            lambda serie: float(np.std(serie, ddof=1)) if len(serie) > 1 else 0.0,
        ),
        muertas_media=("unidades_muertas", "mean"),
        ocupacion_media=("ocupacion", "mean"),
        n=("qe", "size"),
    )
    return agregado.reset_index()


def _valor_python(valor: object) -> object:
    """Convierte escalares de NumPy a tipos nativos de Python."""
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def seleccionar_mejor(
    resumen: pd.DataFrame,
    criterio: str = "minimo",
    tolerancia: float = 0.05,
) -> object:
    """Selecciona el mejor valor de un resumen de un solo parámetro."""
    if criterio not in CRITERIOS_VALIDOS:
        raise ValueError(f"Criterio desconocido: {criterio}")
    if resumen.empty:
        raise ValueError("No hay resultados para seleccionar.")
    if "valor" not in resumen.columns or "qe_media" not in resumen.columns:
        raise ValueError("El resumen debe tener las columnas valor y qe_media.")

    qe_minimo = float(resumen["qe_media"].min())
    if criterio == "minimo":
        fila = resumen.loc[resumen["qe_media"].idxmin()]
        return _valor_python(fila["valor"])

    limite = qe_minimo * (1.0 + tolerancia)
    candidatos = resumen[resumen["qe_media"] <= limite]
    if candidatos.empty:
        candidatos = resumen
    candidatos = candidatos.sort_values("valor")
    return _valor_python(candidatos.iloc[0]["valor"])


def actualizar_config(config: dict, parametro: str, valor: object) -> dict:
    """Devuelve una copia de la configuración con un parámetro actualizado."""
    if parametro not in PARAMETROS_VALIDOS:
        raise ValueError(f"Parámetro desconocido: {parametro}")
    resultado = deepcopy(config)
    resultado["modelo"][parametro] = valor
    return resultado


def _evaluar_candidato(
    datos: np.ndarray,
    config: dict,
    semillas: Sequence[int],
    entrenador: Callable[[np.ndarray, dict, int], dict],
) -> list[dict]:
    """Entrena un candidato con todas las semillas y devuelve sus métricas."""
    filas: list[dict] = []
    for semilla in semillas:
        metricas = entrenador(datos, config["modelo"], int(semilla))
        filas.append(
            {
                "semilla": int(semilla),
                "qe": float(metricas["qe"]),
                "te": float(metricas["te"]),
                "unidades_muertas": int(metricas["unidades_muertas"]),
                "ocupacion": float(metricas["ocupacion"]),
            }
        )
    return filas


def evaluar_paso(
    parametro: str,
    valores: Sequence[object],
    config: dict,
    datos: np.ndarray,
    semillas: Sequence[int],
    entrenador: Callable[[np.ndarray, dict, int], dict],
    criterio: str | None = None,
    tolerancia: float = 0.05,
) -> ResultadoPaso:
    """Evalúa un parámetro manteniendo el resto de la configuración fija."""
    if parametro not in PARAMETROS_VALIDOS:
        raise ValueError(f"Parámetro desconocido: {parametro}")

    candidatos = list(dict.fromkeys(list(valores) + [config["modelo"][parametro]]))
    if criterio is None:
        criterio = "codo" if parametro == "grilla" else "minimo"

    filas: list[dict] = []
    for valor in candidatos:
        config_candidato = actualizar_config(config, parametro, valor)
        for fila in _evaluar_candidato(datos, config_candidato, semillas, entrenador):
            filas.append({"parametro": parametro, "valor": valor, **fila})

    resultados = pd.DataFrame(filas)
    resumen = agregar_resultados(resultados)
    mejor_valor = seleccionar_mejor(resumen, criterio, tolerancia)
    fila_mejor = resumen.loc[resumen["valor"] == mejor_valor].iloc[0]

    return ResultadoPaso(
        parametro=parametro,
        resultados=resultados,
        resumen=resumen,
        mejor_valor=mejor_valor,
        qe_media=float(fila_mejor["qe_media"]),
    )


def evaluar_acoplamiento(
    parametro_a: str,
    valores_a: Sequence[object],
    parametro_b: str,
    valores_b: Sequence[object],
    config: dict,
    datos: np.ndarray,
    semillas: Sequence[int],
    entrenador: Callable[[np.ndarray, dict, int], dict],
) -> ResultadoPaso:
    """Evalúa el producto cartesiano de dos parámetros acoplados."""
    for parametro in (parametro_a, parametro_b):
        if parametro not in PARAMETROS_VALIDOS:
            raise ValueError(f"Parámetro desconocido: {parametro}")
    if parametro_a == parametro_b:
        raise ValueError("Un acoplamiento debe usar dos parámetros distintos.")

    candidatos_a = list(
        dict.fromkeys(list(valores_a) + [config["modelo"][parametro_a]])
    )
    candidatos_b = list(
        dict.fromkeys(list(valores_b) + [config["modelo"][parametro_b]])
    )

    filas: list[dict] = []
    for valor_a in candidatos_a:
        for valor_b in candidatos_b:
            config_candidato = actualizar_config(config, parametro_a, valor_a)
            config_candidato = actualizar_config(
                config_candidato,
                parametro_b,
                valor_b,
            )
            for fila in _evaluar_candidato(
                datos, config_candidato, semillas, entrenador
            ):
                filas.append(
                    {
                        "parametro_a": parametro_a,
                        "valor_a": valor_a,
                        "parametro_b": parametro_b,
                        "valor_b": valor_b,
                        **fila,
                    }
                )

    resultados = pd.DataFrame(filas)
    resumen = agregar_resultados(resultados)
    fila_mejor = resumen.loc[resumen["qe_media"].idxmin()]
    mejor_valor = (
        _valor_python(fila_mejor["valor_a"]),
        _valor_python(fila_mejor["valor_b"]),
    )

    return ResultadoPaso(
        parametro=f"{parametro_a} x {parametro_b}",
        resultados=resultados,
        resumen=resumen,
        mejor_valor=mejor_valor,
        qe_media=float(fila_mejor["qe_media"]),
    )


def debe_adoptar(qe_nuevo: float, qe_actual: float) -> bool:
    """Indica si un acoplamiento mejora la QE actual."""
    return bool(qe_nuevo < qe_actual)


def _qe_de_config(
    config: dict,
    datos: np.ndarray,
    semillas: Sequence[int],
    entrenador: Callable[[np.ndarray, dict, int], dict],
) -> float:
    """Calcula la QE media de una configuración completa."""
    filas = _evaluar_candidato(datos, config, semillas, entrenador)
    return float(np.mean([fila["qe"] for fila in filas]))


def _guardar_paso(
    resultado: ResultadoPaso,
    indice: int,
    ruta_salida: Path,
) -> None:
    """Guarda los CSV crudos y agregados de un paso."""
    nombre = f"paso_{indice:02d}_{resultado.parametro.replace(' ', '_')}"
    resultado.resultados.to_csv(
        ruta_salida / f"{nombre}.csv",
        index=False,
        float_format="%.8f",
    )
    resultado.resumen.to_csv(
        ruta_salida / f"{nombre}_resumen.csv",
        index=False,
        float_format="%.8f",
    )


def _indice_valor(valores: Sequence[object], buscado: object) -> int | None:
    """Devuelve la posición del valor buscado en una secuencia."""
    for indice, valor in enumerate(valores):
        if valor == buscado:
            return indice
        if (
            isinstance(valor, (int, float, np.number))
            and not isinstance(valor, bool)
            and isinstance(buscado, (int, float, np.number))
            and not isinstance(buscado, bool)
            and np.isclose(float(valor), float(buscado))
        ):
            return indice
    return None


def _subrayar_etiqueta_elegida(
    figura: Figure,
    eje: Axes,
    indice: int | None,
) -> None:
    """Subraya la etiqueta del eje x correspondiente al valor elegido."""
    if indice is None:
        return

    figura.canvas.draw()
    etiquetas = eje.get_xticklabels()
    if indice < 0 or indice >= len(etiquetas):
        return

    etiqueta = etiquetas[indice]
    renderizador = figura.canvas.get_renderer()
    caja = etiqueta.get_window_extent(renderer=renderizador)
    inversa = figura.transFigure.inverted()
    x0, y0 = inversa.transform((caja.x0, caja.y0))
    x1, _ = inversa.transform((caja.x1, caja.y0))

    linea = Line2D(
        [x0, x1],
        [y0 - 0.006, y0 - 0.006],
        transform=figura.transFigure,
        color="black",
        linewidth=1.2,
        clip_on=False,
    )
    figura.add_artist(linea)


def _graficar_paso(
    resultado: ResultadoPaso,
    indice: int,
    ruta: Path,
) -> None:
    """Genera la figura individual de un paso (opción A)."""
    resumen = resultado.resumen
    valores = resumen["valor"].tolist()
    figura, eje = plt.subplots(figsize=(9, 5))

    if pd.api.types.is_numeric_dtype(resumen["valor"]):
        eje.errorbar(
            resumen["valor"],
            resumen["qe_media"],
            yerr=resumen["qe_desvio"],
            marker="o",
            capsize=4,
            color="#4C72B0",
        )
        eje.scatter(
            resultado.resultados["valor"],
            resultado.resultados["qe"],
            s=14,
            color="#C44E52",
            alpha=0.55,
            label="Semillas",
        )
        eje.set_xticks(resumen["valor"])
        eje.set_xticklabels([str(valor) for valor in valores])
    else:
        posiciones = np.arange(len(resumen))
        eje.bar(
            posiciones,
            resumen["qe_media"],
            yerr=resumen["qe_desvio"],
            capsize=4,
            color="#4C72B0",
            alpha=0.9,
        )
        for posicion, valor in zip(posiciones, valores):
            semillas = resultado.resultados[resultado.resultados["valor"] == valor]
            eje.scatter(
                np.full(len(semillas), posicion),
                semillas["qe"],
                s=14,
                color="#C44E52",
                alpha=0.55,
                label="Semillas" if posicion == 0 else None,
            )
        eje.set_xticks(posiciones)
        eje.set_xticklabels([str(valor) for valor in valores])

    indice_elegido = _indice_valor(valores, resultado.mejor_valor)
    eje.tick_params(axis="x", labelsize=8)
    eje.set_xlabel(resultado.parametro)
    eje.set_ylabel("QE media")
    eje.set_title(
        f"Paso {indice}: {resultado.parametro} — elegido {resultado.mejor_valor}",
        fontsize=12,
    )
    eje.grid(alpha=0.25)
    eje.legend(fontsize=8)
    figura.tight_layout()
    _subrayar_etiqueta_elegida(figura, eje, indice_elegido)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _graficar_progreso(resumen_pasos: pd.DataFrame, ruta: Path) -> None:
    """Genera el gráfico acumulado del avance del barrido."""
    figura, eje = plt.subplots(figsize=(10, 5))
    posiciones = np.arange(len(resumen_pasos))
    eje.errorbar(
        posiciones,
        resumen_pasos["qe_media"],
        yerr=resumen_pasos.get("qe_desvio", 0.0),
        marker="o",
        capsize=4,
        color="#4C72B0",
    )
    eje.set_xticks(posiciones)
    etiquetas = [
        f"{fila.parametro}\n{fila.valor_elegido}" for fila in resumen_pasos.itertuples()
    ]
    eje.set_xticklabels(etiquetas, rotation=30, ha="right", fontsize=8)
    eje.set_ylabel("QE media")
    eje.set_title("Progreso del barrido secuencial", fontsize=13)
    eje.grid(alpha=0.25)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _graficar_opcion_a(
    resultados_pasos: Sequence[ResultadoPaso],
    resumen_pasos: pd.DataFrame,
    ruta: Path,
) -> None:
    """Genera la opción A: una figura por paso y el progreso."""
    ruta.mkdir(parents=True, exist_ok=True)
    for indice, resultado in enumerate(resultados_pasos, start=1):
        _graficar_paso(
            resultado,
            indice,
            ruta / f"fig_paso_{indice:02d}_{resultado.parametro}.png",
        )
    _graficar_progreso(resumen_pasos, ruta / "fig_progreso.png")


def _graficar_opcion_b(
    resultados_pasos: Sequence[ResultadoPaso],
    acoplamientos: Sequence[ResultadoPaso],
    ruta: Path,
) -> None:
    """Genera la opción B: dashboard compacto y legible."""
    ruta.mkdir(parents=True, exist_ok=True)
    cantidad = len(resultados_pasos)
    columnas = min(3, cantidad)
    filas = int(np.ceil(cantidad / columnas))
    figura, ejes = plt.subplots(
        filas,
        columnas,
        figsize=(5.5 * columnas, 4.0 * filas),
        squeeze=False,
    )

    for eje, resultado in zip(ejes.flat, resultados_pasos):
        resumen = resultado.resumen
        posiciones = np.arange(len(resumen))
        eje.bar(
            posiciones,
            resumen["qe_media"],
            yerr=resumen["qe_desvio"],
            capsize=3,
            color="#4C72B0",
            alpha=0.9,
        )
        eje.set_xticks(posiciones)
        eje.set_xticklabels(
            [str(valor) for valor in resumen["valor"]],
            rotation=30,
            fontsize=7,
        )
        eje.set_title(
            f"{resultado.parametro}: {resultado.mejor_valor}",
            fontsize=9,
        )
        eje.tick_params(axis="y", labelsize=7)
        eje.grid(axis="y", alpha=0.2)

    for eje in list(ejes.flat)[cantidad:]:
        eje.axis("off")

    figura.suptitle("Barrido secuencial de hiperparámetros", fontsize=14)
    figura.tight_layout(rect=(0, 0, 1, 0.95))
    figura.savefig(
        ruta / "fig_dashboard_secuencial.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figura)

    if acoplamientos:
        figura, ejes = plt.subplots(
            1,
            len(acoplamientos),
            figsize=(7.0 * len(acoplamientos), 5.5),
            squeeze=False,
        )
        for eje, resultado in zip(ejes.flat, acoplamientos):
            _dibujar_heatmap_acoplamiento(eje, resultado)
        figura.suptitle("Acoplamientos fuertes", fontsize=14)
        figura.tight_layout(rect=(0, 0, 1, 0.94))
        figura.savefig(
            ruta / "fig_dashboard_acoplamientos.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figura)


def _graficar_opcion_c(
    resultados_pasos: Sequence[ResultadoPaso],
    resumen_pasos: pd.DataFrame,
    ruta: Path,
) -> None:
    """Genera la opción C: boxplots por paso y progreso."""
    ruta.mkdir(parents=True, exist_ok=True)
    for indice, resultado in enumerate(resultados_pasos, start=1):
        figura, eje = plt.subplots(figsize=(9, 5))
        grupos = [
            grupo["qe"].to_numpy()
            for _, grupo in resultado.resultados.groupby("valor", sort=False)
        ]
        etiquetas = [
            str(valor) for valor in resultado.resultados["valor"].drop_duplicates()
        ]
        eje.boxplot(grupos, tick_labels=etiquetas)
        eje.set_xlabel(resultado.parametro)
        eje.set_ylabel("QE por semilla")
        eje.set_title(
            f"Paso {indice}: {resultado.parametro} — elegido {resultado.mejor_valor}",
            fontsize=12,
        )
        eje.grid(axis="y", alpha=0.25)
        figura.tight_layout()
        figura.savefig(
            ruta / f"fig_boxplot_{indice:02d}_{resultado.parametro}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figura)
    _graficar_progreso(resumen_pasos, ruta / "fig_progreso.png")


def _dibujar_heatmap_acoplamiento(eje: plt.Axes, resultado: ResultadoPaso) -> None:
    """Dibuja el heatmap de un acoplamiento."""
    resumen = resultado.resumen
    tabla = resumen.pivot(
        index="valor_b",
        columns="valor_a",
        values="qe_media",
    )
    imagen = eje.imshow(tabla.to_numpy(), cmap="viridis", aspect="auto")
    eje.set_xticks(np.arange(tabla.shape[1]))
    eje.set_yticks(np.arange(tabla.shape[0]))
    eje.set_xticklabels([str(valor) for valor in tabla.columns], rotation=30)
    eje.set_yticklabels([str(valor) for valor in tabla.index])
    eje.set_xlabel(resumen["parametro_a"].iloc[0])
    eje.set_ylabel(resumen["parametro_b"].iloc[0])
    eje.set_title(str(resultado.mejor_valor), fontsize=9)
    for fila in range(tabla.shape[0]):
        for columna in range(tabla.shape[1]):
            valor = tabla.iloc[fila, columna]
            if pd.notna(valor):
                eje.text(
                    columna,
                    fila,
                    f"{valor:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
    return imagen


def _graficar_acoplamiento(
    resultado: ResultadoPaso,
    ruta: Path,
) -> None:
    """Guarda el heatmap de un acoplamiento."""
    ruta.mkdir(parents=True, exist_ok=True)
    figura, eje = plt.subplots(figsize=(8, 6))
    imagen = _dibujar_heatmap_acoplamiento(eje, resultado)
    figura.colorbar(imagen, ax=eje, label="QE media")
    figura.tight_layout()
    figura.savefig(ruta / "fig_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(figura)


def _guardar_figuras(
    resultados_pasos: Sequence[ResultadoPaso],
    resumen_pasos: pd.DataFrame,
    acoplamientos: Sequence[ResultadoPaso],
    opciones: Sequence[str],
    ruta_salida: Path,
) -> None:
    """Genera las opciones de visualización pedidas."""
    if "a" in opciones:
        _graficar_opcion_a(
            resultados_pasos,
            resumen_pasos,
            ruta_salida / "figuras" / "opcion_a",
        )
    if "b" in opciones:
        _graficar_opcion_b(
            resultados_pasos,
            acoplamientos,
            ruta_salida / "figuras" / "opcion_b",
        )
    if "c" in opciones:
        _graficar_opcion_c(
            resultados_pasos,
            resumen_pasos,
            ruta_salida / "figuras" / "opcion_c",
        )


def ejecutar_barrido(
    config: dict,
    datos: np.ndarray,
    config_barrido: dict,
    entrenador: Callable[[np.ndarray, dict, int], dict],
    ruta_salida: Path,
) -> tuple[dict, pd.DataFrame]:
    """Ejecuta el barrido secuencial y los acoplamientos fuertes."""
    validar_config_barrido(config_barrido)
    ruta_salida = Path(ruta_salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    config_actual = deepcopy(config)
    orden = config_barrido["barrido"]["orden"]
    semillas = config_barrido["barrido"]["semillas"]
    criterio_grilla = config_barrido["barrido"].get("criterio_grilla", "minimo")
    tolerancia = float(config_barrido["barrido"].get("tolerancia_codo", 0.05))

    resultados_pasos: list[ResultadoPaso] = []
    filas_resumen: list[dict] = []

    for indice, parametro in enumerate(orden, start=1):
        valores = config_barrido["valores"][parametro]
        resultado = evaluar_paso(
            parametro,
            valores,
            config_actual,
            datos,
            semillas,
            entrenador,
            criterio=criterio_grilla if parametro == "grilla" else "minimo",
            tolerancia=tolerancia,
        )
        _guardar_paso(resultado, indice, ruta_salida)
        config_actual = actualizar_config(
            config_actual,
            parametro,
            resultado.mejor_valor,
        )
        filas_resumen.append(
            {
                "paso": indice,
                "parametro": parametro,
                "valor_elegido": resultado.mejor_valor,
                "qe_media": resultado.qe_media,
                "qe_desvio": float(
                    resultado.resumen.loc[
                        resultado.resumen["valor"] == resultado.mejor_valor,
                        "qe_desvio",
                    ].iloc[0]
                ),
            }
        )
        resultados_pasos.append(resultado)

    qe_actual = _qe_de_config(config_actual, datos, semillas, entrenador)
    acoplamientos: list[ResultadoPaso] = []
    filas_acoplamientos: list[dict] = []
    seccion_acoplamientos = config_barrido.get("acoplamientos", {})
    pares = seccion_acoplamientos.get("pares", [])
    adoptar_solo_si_mejora = bool(
        seccion_acoplamientos.get("adoptar_solo_si_mejora", True)
    )

    for parametro_a, parametro_b in pares:
        resultado = evaluar_acoplamiento(
            parametro_a,
            config_barrido["valores"][parametro_a],
            parametro_b,
            config_barrido["valores"][parametro_b],
            config_actual,
            datos,
            semillas,
            entrenador,
        )
        carpeta = ruta_salida / "acoplamientos" / f"{parametro_a}_x_{parametro_b}"
        carpeta.mkdir(parents=True, exist_ok=True)
        resultado.resultados.to_csv(
            carpeta / "resultados.csv",
            index=False,
            float_format="%.8f",
        )
        resultado.resumen.to_csv(
            carpeta / "resumen.csv",
            index=False,
            float_format="%.8f",
        )
        _graficar_acoplamiento(resultado, carpeta)
        acoplamientos.append(resultado)

        qe_base = qe_actual
        adoptado = False
        if not adoptar_solo_si_mejora or debe_adoptar(resultado.qe_media, qe_actual):
            config_actual = actualizar_config(
                config_actual,
                parametro_a,
                resultado.mejor_valor[0],
            )
            config_actual = actualizar_config(
                config_actual,
                parametro_b,
                resultado.mejor_valor[1],
            )
            qe_actual = resultado.qe_media
            adoptado = True

        filas_acoplamientos.append(
            {
                "parametro_a": parametro_a,
                "parametro_b": parametro_b,
                "mejor_valor_a": resultado.mejor_valor[0],
                "mejor_valor_b": resultado.mejor_valor[1],
                "qe_media": resultado.qe_media,
                "qe_base": qe_base,
                "adoptado": adoptado,
            }
        )

    resumen_pasos = pd.DataFrame(filas_resumen)
    resumen_pasos.to_csv(
        ruta_salida / "resumen_pasos.csv",
        index=False,
        float_format="%.8f",
    )

    if filas_acoplamientos:
        pd.DataFrame(filas_acoplamientos).to_csv(
            ruta_salida / "acoplamientos" / "resumen_acoplamientos.csv",
            index=False,
            float_format="%.8f",
        )

    directorio_final = config_barrido.get("salida", {}).get(
        "directorio_final",
        config_actual["salida"]["directorio"],
    )
    config_actual["salida"]["directorio"] = directorio_final
    kohonen.guardar_config(
        config_actual,
        ruta_salida / "config_optima.toml",
    )
    kohonen.guardar_config(
        config_barrido,
        ruta_salida / "config_barrido.toml",
    )

    lineas_registro = [
        "Registro del barrido — TP4, ejercicio 1.2 (Red de Kohonen)",
        f"Orden: {orden}",
        f"Semillas: {semillas}",
        f"Criterio de grilla: {criterio_grilla}",
        f"Tolerancia del codo: {tolerancia}",
        f"Pares de acoplamiento: {pares}",
        f"Configuración final: {config_actual['modelo']}",
        f"QE final: {qe_actual:.8f}",
        "",
        "Archivos generados:",
        "  config_barrido.toml",
        "  config_optima.toml",
        "  resumen_pasos.csv",
        "  paso_*.csv",
        "  acoplamientos/",
        "  figuras/opcion_a/",
        "  figuras/opcion_b/",
        "  figuras/opcion_c/",
    ]
    (ruta_salida / "registro.txt").write_text(
        "\n".join(lineas_registro) + "\n",
        encoding="utf-8",
    )

    opciones = config_barrido.get("figuras", {}).get("opciones", ["a", "b", "c"])
    _guardar_figuras(
        resultados_pasos,
        resumen_pasos,
        acoplamientos,
        opciones,
        ruta_salida,
    )

    return config_actual, resumen_pasos


def _entrenador_real(datos: np.ndarray, modelo: dict, semilla: int) -> dict:
    """Adapta el motor de Kohonen a la interfaz del barrido."""
    resultado = kohonen.entrenar_som(datos, modelo, semilla)
    return {
        "qe": resultado.qe,
        "te": resultado.te,
        "unidades_muertas": resultado.unidades_muertas,
        "ocupacion": resultado.ocupacion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del barrido."""
    args = parse_args(argv)
    config_barrido = cargar_config_barrido(args.config)

    if args.salida is not None:
        config_barrido["salida"]["directorio"] = str(args.salida)
    if args.semillas is not None:
        config_barrido["barrido"]["semillas"] = list(args.semillas)

    base = kohonen.cargar_config(
        kohonen.resolver_ruta(config_barrido["base"]["archivo"])
    )
    datos = kohonen.cargar_datos(kohonen.resolver_ruta(base["datos"]["ruta"]))
    columnas = kohonen.columnas_entrada(
        datos,
        base["datos"]["columnas_excluidas"],
    )
    matriz = datos[columnas].to_numpy(dtype=float)
    estandarizada, _, _ = kohonen.estandarizar(matriz)

    ruta_salida = kohonen.resolver_ruta(config_barrido["salida"]["directorio"])

    inicio = time.perf_counter()
    config_final, resumen = ejecutar_barrido(
        base,
        estandarizada,
        config_barrido,
        _entrenador_real,
        ruta_salida,
    )
    tiempo_total = time.perf_counter() - inicio

    print("=" * 78)
    print("TP4 — Ejercicio 1.2: barrido de hiperparámetros de Kohonen")
    print("=" * 78)
    print(f"Resultados: {ruta_salida}")
    print(f"Tiempo total: {tiempo_total:.3f} s")
    print(f"Configuración final: {config_final['modelo']}")
    print("Resumen de pasos:")
    print(resumen.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
