#!/usr/bin/env python3
"""Análisis de la recuperación de códigos QR con Hopfield (TP4, ejercicio 2.1).

Script de análisis: solo lee las redes guardadas por ``hopfield.py`` y las
consultas guardadas por ``hopfield-runner.py``; no calcula pesos ni recupera.

Lee, para cada red, las carpetas ``consultas/<deterioro>_<nivel>_semilla_<s>/``
del barrido y las carpetas ``consultas/detalle_<deterioro>_<nivel>/`` con
energía por actualización, y genera tablas y figuras de correlación entre
patrones, tasa de recuperación, barridos, energía y ejemplos visuales.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SALIDA = RAIZ / "resultados" / "ej2" / "hopfield" / "analisis_qr"
REDES_POR_DEFECTO = [
    "hebb=resultados/ej2/hopfield/qr_hebb",
    "pseudoinversa=resultados/ej2/hopfield/qr_pseudoinversa",
]
PATRON_EJEMPLO = "food_menu_qr"

DETERIOROS = ("ruido", "esquina", "manchas")
NOMBRES_DETERIORO = {
    "ruido": "Ruido aleatorio",
    "esquina": "Esquina rota",
    "manchas": "Manchas 3×3",
}
NOMBRES_REGLA = {"hebb": "Hebb", "pseudoinversa": "Pseudo-inversa"}
COLORES_REGLA = {"hebb": "#C44E52", "pseudoinversa": "#4C72B0"}
PATRON_CARPETA = re.compile(r"^(ruido|esquina|manchas)_([0-9.]+)_semilla_(\d+)$")
PATRON_DETALLE = re.compile(r"^detalle_(ruido|esquina|manchas)_([0-9.]+)$")
CASOS_ENERGIA = (("esquina", 0.25), ("manchas", 0.25), ("ruido", 0.4))
COLORES_CASO = {"esquina": "#DD8452", "manchas": "#55A868", "ruido": "#8172B3"}
TASAS = ("correcto", "incorrecto", "espurio", "otro")


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    if camino.is_absolute():
        return camino
    return RAIZ / camino


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analiza la recuperación de QR deteriorados con Hopfield.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--red",
        action="append",
        default=None,
        help="Red a analizar como regla=directorio. Se puede repetir.",
    )
    parser.add_argument("--patron", default=PATRON_EJEMPLO)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    return parser.parse_args(argv)


def leer_carpeta(nombre: str) -> tuple[str, float, int] | None:
    """Devuelve ``(deterioro, nivel, semilla)`` si es una carpeta del barrido."""
    coincidencia = PATRON_CARPETA.match(nombre)
    if coincidencia is None:
        return None
    deterioro, nivel, semilla = coincidencia.groups()
    return deterioro, float(nivel), int(semilla)


def tasas(consultas: pd.DataFrame) -> dict[str, float]:
    """Fracción de cada tipo de resultado y medias de barridos y distancias."""
    tipos = consultas["tipo_resultado"]
    correcto = float(np.mean(tipos == "almacenado_correcto"))
    incorrecto = float(np.mean(tipos == "almacenado_incorrecto"))
    espurio = float(np.mean(tipos == "espurio"))
    return {
        "correcto": correcto,
        "incorrecto": incorrecto,
        "espurio": espurio,
        "otro": 1.0 - correcto - incorrecto - espurio,
        "barridos": float(consultas["barridos"].mean()),
        "hamming_inicial": float(consultas["hamming_inicial"].mean()),
        "hamming_final": float(consultas["distancia_hamming"].mean()),
    }


def estado_mezcla(patrones: np.ndarray) -> np.ndarray:
    """Voto mayoritario de los patrones (con empate, 1)."""
    return np.where(np.asarray(patrones).sum(axis=0) >= 0, 1, -1)


def pixeles_comunes(patrones: np.ndarray) -> np.ndarray:
    """Máscara de los píxeles que valen lo mismo en todos los patrones."""
    arreglo = np.asarray(patrones)
    return np.all(arreglo == arreglo[0], axis=0)


def _leer_estados(ruta: Path) -> pd.DataFrame:
    tabla = pd.read_csv(ruta)
    return tabla.set_index(tabla.columns[0])


def _pixeles(tabla: pd.DataFrame) -> list[str]:
    return [c for c in tabla.columns if c.startswith("p") and c[1:].isdigit()]


def cargar_red(directorio: Path) -> dict:
    """Lee configuración, patrones y estabilidad de una red guardada."""
    with (directorio / "config_usada.toml").open("rb") as archivo:
        config = tomllib.load(archivo)
    patrones = _leer_estados(directorio / "patrones.csv")
    return {
        "config": config,
        "nombres": list(patrones.index),
        "filas": int(patrones["filas"].iloc[0]),
        "columnas": int(patrones["columnas"].iloc[0]),
        "patrones": patrones[_pixeles(patrones)].to_numpy(dtype=int),
        "estabilidad": pd.read_csv(directorio / "estabilidad.csv"),
        "capacidad": pd.read_csv(directorio / "capacidad_subconjuntos.csv"),
    }


def tabla_recuperacion(regla: str, directorio: Path) -> pd.DataFrame:
    """Tasas por deterioro, nivel y semilla de una red."""
    filas = []
    for carpeta in sorted((directorio / "consultas").iterdir()):
        clave = leer_carpeta(carpeta.name)
        if clave is None:
            continue
        deterioro, nivel, semilla = clave
        consultas = pd.read_csv(carpeta / "consultas.csv")
        filas.append(
            {
                "regla": regla,
                "deterioro": deterioro,
                "nivel": nivel,
                "semilla": semilla,
                "consultas": len(consultas),
                **tasas(consultas),
            }
        )
    return pd.DataFrame(filas)


def resumir(por_semilla: pd.DataFrame) -> pd.DataFrame:
    """Media y desvío muestral sobre semillas."""
    metricas = [*TASAS, "barridos", "hamming_inicial", "hamming_final"]
    agrupado = por_semilla.groupby(["regla", "deterioro", "nivel"], sort=True)
    resumen = agrupado[metricas].agg(["mean", lambda serie: serie.std(ddof=1)])
    resumen.columns = [
        f"{metrica}_{'media' if estadistico == 'mean' else 'desvio'}"
        for metrica, estadistico in resumen.columns
    ]
    resumen["semillas"] = agrupado.size()
    return resumen.reset_index()


def _imagen(eje: plt.Axes, estado: np.ndarray, filas: int, columnas: int) -> None:
    # Negro (1) se dibuja negro, como en el QR impreso.
    eje.imshow(
        np.asarray(estado).reshape(filas, columnas),
        cmap="gray_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    eje.set_xticks([])
    eje.set_yticks([])


def _figura_correlacion(red: dict, solapamiento: np.ndarray, ruta: Path) -> None:
    nombres = [nombre.replace("_menu_qr", "") for nombre in red["nombres"]]
    comunes = pixeles_comunes(red["patrones"])
    mezcla = estado_mezcla(red["patrones"])
    filas, columnas = red["filas"], red["columnas"]

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.8))
    imagen = ejes[0].imshow(solapamiento, cmap="viridis", vmin=0, vmax=1)
    for i, j in np.ndindex(solapamiento.shape):
        ejes[0].text(
            j,
            i,
            f"{solapamiento[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black" if solapamiento[i, j] > 0.6 else "white",
        )
    ejes[0].set_xticks(range(len(nombres)), nombres, rotation=30, ha="right")
    ejes[0].set_yticks(range(len(nombres)), nombres)
    ejes[0].set_title("Solapamiento ξᵘ·ξᵛ / N entre los QR", fontsize=10)
    fig.colorbar(imagen, ax=ejes[0], fraction=0.046)

    mapa = np.where(comunes, 0.25, 1.0).reshape(filas, columnas)
    ejes[1].imshow(mapa, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ejes[1].set_xticks([])
    ejes[1].set_yticks([])
    ejes[1].set_title(
        f"Píxeles idénticos en los {len(nombres)} QR (gris):\n"
        f"{comunes.sum()} de {comunes.size} ({comunes.mean():.1%})",
        fontsize=10,
    )

    _imagen(ejes[2], mezcla, filas, columnas)
    distancias = (red["patrones"] != mezcla).sum(axis=1)
    ejes[2].set_title(
        "Estado mezcla (voto mayoritario)\n"
        f"Hamming a cada QR: {', '.join(str(d) for d in distancias)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _figura_por_nivel(
    resumen: pd.DataFrame,
    metricas: Sequence[tuple[str, str]],
    ylabel: str,
    ruta,
    reglas: Sequence[str] | None = None,
    piso: float | None = None,
) -> None:
    fig, ejes = plt.subplots(1, len(DETERIOROS), figsize=(15, 4.5), sharey=True)
    for eje, deterioro in zip(ejes, DETERIOROS, strict=True):
        datos = resumen[resumen["deterioro"] == deterioro]
        if reglas is not None:
            datos = datos[datos["regla"].isin(reglas)]
        for regla, grupo in datos.groupby("regla"):
            for metrica, estilo in metricas:
                eje.errorbar(
                    grupo["nivel"],
                    grupo[f"{metrica}_media"],
                    yerr=grupo[f"{metrica}_desvio"],
                    color=COLORES_REGLA.get(regla, "0.3"),
                    linestyle=estilo,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    capsize=4,
                    label=NOMBRES_REGLA.get(regla, regla)
                    if len(metricas) == 1
                    else f"{NOMBRES_REGLA.get(regla, regla)} — {metrica}",
                )
        eje.set_title(NOMBRES_DETERIORO[deterioro], fontsize=16)
        eje.set_xlabel("Nivel (fracción de píxeles afectados)", fontsize=13)
        eje.tick_params(labelsize=12)
        eje.grid(alpha=0.3)
        if piso is not None:
            eje.axhline(piso, color="black", ls=":", lw=1.5)
            eje.set_ylim(piso - 0.15, 3.0)
    if piso is not None:
        ejes[0].annotate(
            "mínimo posible: 2 barridos",
            (0.05, piso),
            xytext=(0, -16),
            textcoords="offset points",
            fontsize=11,
        )
    ejes[0].set_ylabel(ylabel, fontsize=13)
    contenedores, etiquetas = ejes[-1].get_legend_handles_labels()
    lineas = [
        Line2D(
            [],
            [],
            color=c[0].get_color(),
            linestyle=c[0].get_linestyle(),
            linewidth=2.5,
        )
        for c in contenedores
    ]
    ejes[-1].legend(lineas, etiquetas, fontsize=11, handlelength=3.5)
    fig.suptitle("Media ± desvío sobre 3 semillas (5 QR × 5 repeticiones)", fontsize=13)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _detalles(directorio: Path) -> dict[tuple[str, float], Path]:
    encontrados = {}
    for carpeta in sorted((directorio / "consultas").iterdir()):
        coincidencia = PATRON_DETALLE.match(carpeta.name)
        if coincidencia is not None:
            deterioro, nivel = coincidencia.groups()
            encontrados[(deterioro, float(nivel))] = carpeta
    return encontrados


def _figura_energia(redes: dict[str, dict], patron: str, ruta: Path) -> None:
    """Energía por paso, normalizada: 0 es la energía del QR original."""
    fig, ejes = plt.subplots(1, len(redes), figsize=(15, 4.8), sharex=True, sharey=True)
    ejes = np.atleast_1d(ejes)
    for eje, (regla, red) in zip(ejes, redes.items(), strict=True):
        n = red["filas"] * red["columnas"]
        referencia = (
            red["estabilidad"].set_index("nombre").loc[patron, "energia_inicial"]
        )
        for caso in CASOS_ENERGIA:
            energia = pd.read_csv(red["detalles"][caso] / "energia_actualizaciones.csv")
            energia = energia[energia["consulta"] == patron]
            relativa = (energia["energia"] - referencia) / abs(referencia)
            eje.plot(
                energia["actualizacion"] / n,
                relativa,
                color=COLORES_CASO[caso[0]],
                linewidth=2.5,
                label=f"{NOMBRES_DETERIORO[caso[0]]} {caso[1]:g}",
            )
        final = relativa.iloc[-1]
        eje.axhline(0, color="black", ls=":", lw=1.5)
        eje.annotate(
            "energía del QR original",
            (2.0, 0),
            xytext=(0, 6),
            textcoords="offset points",
            ha="right",
            fontsize=12,
        )
        eje.annotate(
            f"estado final: {final:+.1%} respecto del QR",
            (2.0, final),
            xytext=(0, -16),
            textcoords="offset points",
            ha="right",
            fontsize=12,
        )
        eje.set_ylim(-0.15, 1.05)
        eje.set_title(NOMBRES_REGLA.get(regla, regla), fontsize=16)
        eje.set_xlabel("Barridos (actualizaciones / N)", fontsize=13)
        eje.tick_params(labelsize=12)
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("Energía relativa (H − H_QR) / |H_QR|", fontsize=13)
    ejes[0].legend(fontsize=11, handlelength=2.5)
    fig.suptitle(f"Consulta: {patron}, semilla 0", fontsize=13)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _figura_ejemplos(redes: dict[str, dict], patron: str, ruta: Path) -> None:
    """Consultas deterioradas de cada caso, dispuestas en una fila."""
    casos = sorted(set.intersection(*(set(r["detalles"]) for r in redes.values())))
    primera = next(iter(redes.values()))
    filas, columnas = primera["filas"], primera["columnas"]
    original = primera["patrones"][primera["nombres"].index(patron)]

    fig, ejes = plt.subplots(1, 4, figsize=(20, 5.4))
    for eje, caso in zip(ejes, casos, strict=True):
        iniciales = _leer_estados(primera["detalles"][caso] / "estados_iniciales.csv")
        inicial = iniciales.loc[patron, _pixeles(iniciales)].to_numpy()
        _imagen(eje, inicial, filas, columnas)
        cambiados = int((inicial != original).sum())
        eje.set_title(
            f"{NOMBRES_DETERIORO[caso[0]]} {caso[1]:g}\n({cambiados} píxeles distintos)",
            fontsize=15,
        )
    fig.tight_layout(w_pad=2)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tabla_capacidad(redes: dict[str, dict]) -> pd.DataFrame:
    """Subconjuntos con todos sus patrones estables, por regla y cantidad."""
    filas = []
    for regla, red in redes.items():
        for cantidad, grupo in red["capacidad"].groupby("cantidad"):
            filas.append(
                {
                    "regla": regla,
                    "cantidad": int(cantidad),
                    "subconjuntos": len(grupo),
                    "subconjuntos_estables": int(grupo["todos_estables"].sum()),
                    "fraccion_estables": float(grupo["todos_estables"].mean()),
                    "margen_minimo": float(grupo["margen_minimo"].min()),
                }
            )
    return pd.DataFrame(filas)


def _figura_capacidad(capacidad: pd.DataFrame, ruta: Path) -> None:
    reglas = list(dict.fromkeys(capacidad["regla"]))
    cantidades = sorted(capacidad["cantidad"].unique())
    ancho = 0.8 / len(reglas)
    fig, eje = plt.subplots(figsize=(8, 4.5))
    for desplazamiento, regla in enumerate(reglas):
        datos = capacidad[capacidad["regla"] == regla].set_index("cantidad")
        posiciones = (
            np.array(cantidades) + (desplazamiento - (len(reglas) - 1) / 2) * ancho
        )
        barras = eje.bar(
            posiciones,
            datos.loc[cantidades, "fraccion_estables"],
            width=ancho,
            color=COLORES_REGLA.get(regla, "0.5"),
            label=NOMBRES_REGLA.get(regla, regla),
        )
        etiquetas = [
            f"{fila.subconjuntos_estables}/{fila.subconjuntos}"
            for fila in datos.loc[cantidades].itertuples()
        ]
        eje.bar_label(barras, etiquetas, fontsize=8)
    eje.set_xticks(cantidades)
    eje.set_xlabel("Cantidad de QR almacenados")
    eje.set_ylabel("Fracción de subconjuntos con todos estables")
    eje.set_ylim(0, 1.15)
    eje.set_title(
        "Capacidad práctica: ¿todos los QR almacenados son puntos fijos?\n"
        "Se prueban todos los subconjuntos de cada tamaño",
        fontsize=10,
    )
    eje.legend()
    eje.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _resumen_texto(
    redes: dict[str, dict],
    resumen: pd.DataFrame,
    solapamiento: np.ndarray,
    capacidad: pd.DataFrame,
) -> str:
    primera = next(iter(redes.values()))
    comunes = pixeles_comunes(primera["patrones"])
    fuera_diagonal = solapamiento[~np.eye(len(solapamiento), dtype=bool)]
    lineas = [
        "Recuperación de QR deteriorados con Hopfield — TP4, ejercicio 2.1",
        "",
        f"Patrones: {len(primera['nombres'])} de "
        f"{primera['filas']}×{primera['columnas']} = "
        f"{primera['filas'] * primera['columnas']} neuronas",
        f"Píxeles idénticos en todos los QR: {comunes.sum()} ({comunes.mean():.1%})",
        f"Solapamiento entre pares: {fuera_diagonal.min():.3f} a "
        f"{fuera_diagonal.max():.3f}",
        "",
        "Puntos fijos por regla:",
    ]
    for regla, red in redes.items():
        estabilidad = red["estabilidad"]
        lineas.append(
            f"  {regla}: {int(estabilidad['punto_fijo'].sum())} de {len(estabilidad)}"
        )
    lineas += ["", "Subconjuntos con todos los QR estables, por cantidad:"]
    for fila in capacidad.itertuples():
        lineas.append(
            f"  {fila.regla}: {fila.cantidad} QR -> "
            f"{fila.subconjuntos_estables} de {fila.subconjuntos}"
        )
    lineas += ["", "Tasa de recuperación correcta (media ± desvío sobre semillas):"]
    for fila in resumen.itertuples():
        lineas.append(
            f"  {fila.regla:<14} {fila.deterioro:<8} nivel {fila.nivel:<5g} "
            f"correcto {fila.correcto_media:.3f} ± {fila.correcto_desvio:.3f} | "
            f"espurio {fila.espurio_media:.3f} | "
            f"incorrecto {fila.incorrecto_media:.3f} | "
            f"barridos {fila.barridos_media:.2f}"
        )
    return "\n".join(lineas) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    redes: dict[str, dict] = {}
    tablas = []
    for especificacion in args.red or REDES_POR_DEFECTO:
        regla, _, directorio = especificacion.partition("=")
        if not directorio:
            raise ValueError(f"Red inválida, se espera regla=dir: {regla}")
        ruta = resolver_ruta(directorio)
        redes[regla] = cargar_red(ruta)
        redes[regla]["detalles"] = _detalles(ruta)
        tablas.append(tabla_recuperacion(regla, ruta))

    primera = next(iter(redes.values()))
    patron = args.patron if args.patron in primera["nombres"] else primera["nombres"][0]
    patrones = primera["patrones"].astype(float)
    solapamiento = patrones @ patrones.T / patrones.shape[1]

    por_semilla = pd.concat(tablas, ignore_index=True)
    resumen = resumir(por_semilla)

    salida = resolver_ruta(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    por_semilla.to_csv(
        salida / "recuperacion_por_semilla.csv", index=False, float_format="%.6f"
    )
    resumen.to_csv(salida / "recuperacion.csv", index=False, float_format="%.6f")
    pd.DataFrame(
        solapamiento, index=primera["nombres"], columns=primera["nombres"]
    ).to_csv(salida / "correlacion.csv", index_label="nombre", float_format="%.6f")

    _figura_correlacion(primera, solapamiento, salida / "fig_correlacion.png")
    _figura_por_nivel(
        resumen,
        [("correcto", "-"), ("espurio", "--")],
        "Fracción de consultas",
        salida / "fig_recuperacion.png",
        reglas=("pseudoinversa",),
    )
    _figura_por_nivel(
        resumen,
        [("barridos", "-")],
        "Barridos hasta estabilizar",
        salida / "fig_barridos.png",
        piso=2,
    )
    capacidad = tabla_capacidad(redes)
    capacidad.to_csv(salida / "capacidad.csv", index=False, float_format="%.6f")
    _figura_capacidad(capacidad, salida / "fig_capacidad.png")
    _figura_energia(redes, patron, salida / "fig_energia.png")
    _figura_ejemplos(redes, patron, salida / "fig_ejemplos.png")

    texto = _resumen_texto(redes, resumen, solapamiento, capacidad)
    (salida / "resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Resultados: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
