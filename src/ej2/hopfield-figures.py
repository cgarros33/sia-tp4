#!/usr/bin/env python3
"""Figuras de la recuperación de QR con Hopfield (TP4, ejercicio 2.1).

Script de análisis: solo lee los CSV y NPZ que guardan el entrenamiento, el
runner y el barrido de ruido; no calcula pesos ni recupera consultas. Todas las
figuras se pueden regenerar sin volver a ejecutar la red.

Uso:
    python src/ej2/hopfield-figures.py
    python src/ej2/hopfield-figures.py --solo modelos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

RAIZ = Path(__file__).resolve().parents[2]
RUTA_RESULTADOS = RAIZ / "resultados" / "ej2" / "hopfield" / "qr_37x37"
DPI = 150
MODELOS = ("aleatorio", "esquina", "finder", "rayones", "manchas", "bloques")
NOMBRES_MODELO = {
    "aleatorio": "Ruido aleatorio",
    "esquina": "Esquina rota",
    "finder": "Finder borrado",
    "rayones": "Rayones",
    "manchas": "Manchas",
    "bloques": "Bloques borrados",
    "combinaciones": "Combinaciones",
}
EJES_NIVEL = {
    "aleatorio": "Fracción de bits invertidos",
    "esquina": "Lado L del cuadrado borrado (píxeles)",
    "finder": "Finders de 7 × 7 borrados",
    "rayones": "Cantidad de líneas",
    "manchas": "Manchas (cantidad × radio)",
    "bloques": "Fracción de bloques de 6 × 6 borrados",
    "combinaciones": "Paso de la escalera",
}
# Paleta categórica de referencia, en orden fijo, y marcadores como segunda
# codificación para que la identidad no dependa solo del color.
CATEGORICOS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
MARCADORES = ("o", "s", "^", "D", "v", "P", "X", "*")
COLORES_REGLA = {"pseudoinversa": CATEGORICOS[0], "hebb": CATEGORICOS[1]}
MARCADORES_REGLA = {"pseudoinversa": "o", "hebb": "s"}
NOMBRES_REGLA = {"hebb": "Hebb", "pseudoinversa": "Pseudo-inversa"}
TIPOS = {
    "almacenado_correcto": ("Correcto", "exactitud"),
    "almacenado_incorrecto": ("Incorrecto", "tasa_incorrecto"),
    "espurio": ("Espurio", "tasa_espurio"),
    "ciclo": ("Ciclo", "tasa_ciclo"),
    "no_converge": ("No converge", "tasa_no_converge"),
}
TINTA_SECUNDARIA = "#52514e"
GRIS = "#8f8e88"
# Rampa ordinal azul: el paso más claro todavía se distingue de la superficie.
RAMPA = LinearSegmentedColormap.from_list(
    "azul", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.edgecolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "font.size": 9,
        "axes.titlesize": 10,
    }
)


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    return camino if camino.is_absolute() else RAIZ / camino


def leer(ruta: Path) -> pd.DataFrame:
    """Lee un CSV de resultados conservando las etiquetas de nivel como texto."""
    return pd.read_csv(ruta, dtype={"nivel": str})


def nombre_corto(nombre: str) -> str:
    """Nombre del QR sin el sufijo común."""
    return nombre.replace("_menu_qr", "")


def _etiquetas(tabla: pd.DataFrame) -> list[str]:
    """Etiquetas de nivel en el orden del barrido."""
    niveles = (
        tabla[["orden_nivel", "nivel"]].drop_duplicates().sort_values("orden_nivel")
    )
    return niveles["nivel"].tolist()


def _por_regla(tabla: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Grupos por regla, ordenados por nivel."""
    return [
        (regla, grupo.sort_values("orden_nivel"))
        for regla, grupo in tabla.groupby("regla", sort=True)
    ]


def _eje_niveles(eje: plt.Axes, etiquetas: Sequence[str], titulo: str) -> None:
    eje.set_xticks(
        range(len(etiquetas)), etiquetas, rotation=45 if len(etiquetas) > 8 else 0
    )
    eje.set_xlabel(titulo)


def _lineas(
    eje: plt.Axes, resumen: pd.DataFrame, metrica: str, etiquetas: Sequence[str]
) -> None:
    """Una línea por regla con media ± desvío sobre semillas."""
    posiciones = {etiqueta: indice for indice, etiqueta in enumerate(etiquetas)}
    for regla, grupo in _por_regla(resumen):
        eje.errorbar(
            grupo["nivel"].map(posiciones),
            grupo[f"{metrica}_media"],
            yerr=grupo[f"{metrica}_desvio"].fillna(0.0),
            color=COLORES_REGLA.get(regla, GRIS),
            marker=MARCADORES_REGLA.get(regla, "o"),
            capsize=3,
            label=NOMBRES_REGLA.get(regla, regla),
        )


def _nota(fig: plt.Figure, resumen: pd.DataFrame) -> None:
    semillas = int(resumen["semillas"].max())
    por_semilla = int(round(resumen["consultas"].max() / semillas))
    fig.text(
        0.01,
        0.005,
        f"Media ± desvío muestral sobre {semillas} semillas "
        f"({por_semilla} consultas por semilla y regla).",
        fontsize=7,
        color=TINTA_SECUNDARIA,
    )


def _guardar(fig: plt.Figure, ruta: Path) -> None:
    # Margen inferior para la nota y superior para el título general.
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)


def nivel_referencia(resumen: pd.DataFrame) -> str:
    """Menor nivel en que la pseudo-inversa cae por debajo de 0,95; si no, el central."""
    pseudo = resumen[resumen["regla"] == "pseudoinversa"].sort_values("orden_nivel")
    if pseudo.empty:
        pseudo = resumen.sort_values("orden_nivel").drop_duplicates("orden_nivel")
    debajo = pseudo[pseudo["exactitud_media"] < 0.95]
    if not debajo.empty:
        return str(debajo["nivel"].iloc[0])
    return str(pseudo["nivel"].iloc[len(pseudo) // 2])


def _fig_exactitud(resumen, etiquetas, modelo, ruta) -> None:
    fig, eje = plt.subplots(figsize=(7, 4.2))
    _lineas(eje, resumen, "exactitud", etiquetas)
    eje.set_ylim(-0.05, 1.05)
    eje.set_ylabel("Exactitud (recupera el QR de origen)")
    _eje_niveles(eje, etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    eje.set_title(f"{NOMBRES_MODELO.get(modelo, modelo)}: exactitud por nivel")
    eje.legend()
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _fig_composicion(resumen, etiquetas, modelo, ruta) -> None:
    grupos = _por_regla(resumen)
    presentes = [
        tipo
        for tipo, (_, tasa) in TIPOS.items()
        if (resumen[f"{tasa}_media"] > 0).any()
    ]
    fig, ejes = plt.subplots(
        1, len(grupos), figsize=(6 * len(grupos), 4.2), sharey=True
    )
    ejes = np.atleast_1d(ejes)
    for eje, (regla, grupo) in zip(ejes, grupos, strict=True):
        base = np.zeros(len(grupo))
        for indice, tipo in enumerate(TIPOS):
            if tipo not in presentes:
                continue
            nombre, tasa = TIPOS[tipo]
            valores = grupo[f"{tasa}_media"].to_numpy()
            eje.bar(
                range(len(grupo)),
                valores,
                bottom=base,
                width=0.8,
                color=CATEGORICOS[indice],
                edgecolor="white",
                linewidth=1,
                label=nombre,
            )
            base += valores
        eje.set_ylim(0, 1.02)
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
        eje.grid(axis="x", visible=False)
        _eje_niveles(eje, etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    ejes[0].set_ylabel("Proporción de consultas")
    ejes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{NOMBRES_MODELO.get(modelo, modelo)}: composición de resultados")
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _fig_iteraciones(resumen, etiquetas, modelo, ruta) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    for eje, (metrica, titulo) in zip(
        ejes,
        (
            ("barridos", "Barridos hasta estabilizar"),
            ("cambios_totales", "Bits cambiados"),
        ),
        strict=True,
    ):
        _lineas(eje, resumen, metrica, etiquetas)
        eje.set_ylabel(titulo)
        eje.set_ylim(bottom=0)
        _eje_niveles(eje, etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    ejes[0].legend()
    fig.suptitle(
        f"{NOMBRES_MODELO.get(modelo, modelo)}: iteraciones de la recuperación"
    )
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _fig_energia(resumen, energia, etiquetas, modelo, ruta) -> None:
    grupos = _por_regla(energia)
    fig, ejes = plt.subplots(1, len(grupos) + 1, figsize=(5 * (len(grupos) + 1), 4.2))
    colores = {
        etiqueta: RAMPA(indice / max(len(etiquetas) - 1, 1))
        for indice, etiqueta in enumerate(etiquetas)
    }
    for eje, (regla, grupo) in zip(ejes[:-1], grupos, strict=True):
        for etiqueta in etiquetas:
            datos = grupo[grupo["nivel"] == etiqueta].sort_values("barrido")
            eje.plot(
                datos["barrido"],
                datos["energia_por_neurona_media"],
                color=colores[etiqueta],
                marker="o",
                markersize=4,
                label=etiqueta,
            )
        eje.set_title(f"{NOMBRES_REGLA.get(regla, regla)}: E/N media por barrido")
        eje.set_xlabel("Barrido")
        eje.xaxis.set_major_locator(MaxNLocator(integer=True))
        eje.set_ylabel("E / N")
    ejes[0].legend(title="Nivel", fontsize=7, ncol=2)
    _lineas(ejes[-1], resumen, "delta_energia_por_neurona", etiquetas)
    ejes[-1].set_title("ΔE / N (final − inicial)")
    ejes[-1].set_ylabel("ΔE / N")
    _eje_niveles(ejes[-1], etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    ejes[-1].legend()
    fig.suptitle(
        f"{NOMBRES_MODELO.get(modelo, modelo)}: energía (no comparable entre reglas)"
    )
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _fig_hamming(resumen, etiquetas, modelo, ruta) -> None:
    fig, eje = plt.subplots(figsize=(7, 4.2))
    _lineas(eje, resumen, "hamming_final_normalizado", etiquetas)
    referencia = resumen.sort_values("orden_nivel").drop_duplicates("orden_nivel")
    eje.plot(
        range(len(referencia)),
        referencia["fraccion_afectada_media"],
        color=GRIS,
        linestyle="--",
        linewidth=1.5,
        label="Fracción afectada de la consulta",
    )
    eje.set_ylim(bottom=0)
    eje.set_ylabel("Hamming final al origen / N")
    _eje_niveles(eje, etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    eje.set_title(f"{NOMBRES_MODELO.get(modelo, modelo)}: error de reconstrucción")
    eje.legend()
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _fig_confusion(resumen, consultas, modelo, ruta) -> None:
    referencia = nivel_referencia(resumen)
    datos = consultas[consultas["nivel"] == referencia]
    nombres = (
        datos[["indice_patron", "nombre_origen"]]
        .drop_duplicates()
        .sort_values("indice_patron")["nombre_origen"]
        .tolist()
    )
    final = np.where(
        datos["tipo_resultado"].isin(["almacenado_correcto", "almacenado_incorrecto"]),
        datos["nombre_final"].fillna(""),
        np.where(datos["tipo_resultado"] == "espurio", "espurio", "otro"),
    )
    columnas = [*nombres, "espurio"] + (["otro"] if (final == "otro").any() else [])
    datos = datos.assign(final=final)
    grupos = _por_regla(datos)
    fig, ejes = plt.subplots(1, len(grupos), figsize=(5.5 * len(grupos), 4.8))
    ejes = np.atleast_1d(ejes)
    for eje, (regla, grupo) in zip(ejes, grupos, strict=True):
        matriz = (
            pd.crosstab(grupo["nombre_origen"], grupo["final"])
            .reindex(index=nombres, columns=columnas, fill_value=0)
            .to_numpy()
        )
        maximo = max(int(matriz.sum(axis=1).max()), 1)
        eje.imshow(matriz, cmap=SECUENCIAL, vmin=0, vmax=maximo)
        for (i, j), valor in np.ndenumerate(matriz):
            eje.text(
                j,
                i,
                str(valor),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if valor > maximo / 2 else "black",
            )
        eje.set_xticks(
            range(len(columnas)),
            [nombre_corto(c) for c in columnas],
            rotation=30,
            ha="right",
        )
        eje.set_yticks(range(len(nombres)), [nombre_corto(n) for n in nombres])
        eje.set_xlabel("Estado final")
        eje.set_ylabel("QR de origen")
        eje.grid(visible=False)
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
    fig.suptitle(
        f"{NOMBRES_MODELO.get(modelo, modelo)}: confusión en el nivel {referencia} "
        "(todas las semillas y repeticiones)"
    )
    _guardar(fig, ruta)


def _fig_por_patron(resumen, por_patron, etiquetas, modelo, ruta) -> None:
    grupos = _por_regla(por_patron)
    nombres = sorted(por_patron["nombre_origen"].unique())
    posiciones = {etiqueta: indice for indice, etiqueta in enumerate(etiquetas)}
    fig, ejes = plt.subplots(
        1, len(grupos), figsize=(6 * len(grupos), 4.2), sharey=True
    )
    ejes = np.atleast_1d(ejes)
    for eje, (regla, grupo) in zip(ejes, grupos, strict=True):
        for indice, nombre in enumerate(nombres):
            datos = grupo[grupo["nombre_origen"] == nombre].sort_values("orden_nivel")
            eje.plot(
                datos["nivel"].map(posiciones),
                datos["exactitud_media"],
                color=CATEGORICOS[indice % len(CATEGORICOS)],
                marker=MARCADORES[indice % len(MARCADORES)],
                label=nombre_corto(nombre),
            )
        eje.set_ylim(-0.05, 1.05)
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
        _eje_niveles(eje, etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    ejes[0].set_ylabel("Exactitud media")
    ejes[-1].legend(title="QR")
    fig.suptitle(f"{NOMBRES_MODELO.get(modelo, modelo)}: exactitud por QR")
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _fig_boxplot_barridos(consultas, etiquetas, modelo, ruta) -> None:
    grupos = _por_regla(consultas)
    ancho = 0.8 / len(grupos)
    fig, eje = plt.subplots(figsize=(8, 4.2))
    for indice, (regla, grupo) in enumerate(grupos):
        desplazamiento = (indice - (len(grupos) - 1) / 2) * ancho
        valores = [
            grupo.loc[grupo["nivel"] == e, "barridos"].to_numpy() for e in etiquetas
        ]
        color = COLORES_REGLA.get(regla, GRIS)
        eje.boxplot(
            valores,
            positions=np.arange(len(etiquetas)) + desplazamiento,
            widths=ancho * 0.85,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color},
            medianprops={"color": color, "linewidth": 2},
            whiskerprops={"color": color},
            capprops={"color": color},
            flierprops={"markeredgecolor": color, "markersize": 4},
        )
    eje.legend(
        handles=[
            Patch(
                facecolor=COLORES_REGLA.get(r, GRIS),
                alpha=0.5,
                label=NOMBRES_REGLA.get(r, r),
            )
            for r, _ in grupos
        ]
    )
    eje.set_ylabel("Barridos hasta estabilizar")
    _eje_niveles(eje, etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    eje.set_title(f"{NOMBRES_MODELO.get(modelo, modelo)}: distribución de barridos")
    _guardar(fig, ruta)


def _fig_afectacion(consultas, etiquetas, modelo, ruta) -> None:
    # La consulta es la misma para todas las reglas: alcanza con una.
    datos = consultas[consultas["regla"] == consultas["regla"].iloc[0]]
    fig, eje = plt.subplots(figsize=(7, 4.2))
    valores = [
        datos.loc[datos["nivel"] == e, "fraccion_afectada"].to_numpy()
        for e in etiquetas
    ]
    eje.boxplot(
        valores,
        positions=range(len(etiquetas)),
        widths=0.5,
        patch_artist=True,
        boxprops={
            "facecolor": CATEGORICOS[0],
            "alpha": 0.35,
            "edgecolor": CATEGORICOS[0],
        },
        medianprops={"color": CATEGORICOS[0], "linewidth": 2},
        whiskerprops={"color": CATEGORICOS[0]},
        capprops={"color": CATEGORICOS[0]},
        flierprops={"markeredgecolor": CATEGORICOS[0], "markersize": 4},
    )
    nominal = [
        datos.loc[datos["nivel"] == e, "fraccion_nominal"].mean() for e in etiquetas
    ]
    eje.plot(
        range(len(etiquetas)),
        nominal,
        color=GRIS,
        linestyle="none",
        marker="x",
        markersize=8,
        label="Zona dañada (incluye píxeles que ya eran blancos)",
    )
    eje.legend(
        handles=[
            Patch(facecolor=CATEGORICOS[0], alpha=0.5, label="Fracción afectada real"),
            *eje.get_legend_handles_labels()[0],
        ]
    )
    eje.set_ylim(bottom=0)
    eje.set_ylabel("Fracción de los N píxeles")
    _eje_niveles(eje, etiquetas, EJES_NIVEL.get(modelo, "Nivel"))
    eje.set_title(f"{NOMBRES_MODELO.get(modelo, modelo)}: daño nominal y real")
    _guardar(fig, ruta)


def figuras_modelo(directorio: Path, modelo: str) -> None:
    """Las nueve figuras de un modelo de ruido o de un paso de la escalera."""
    resumen = leer(directorio / "resumen_niveles.csv")
    consultas = leer(directorio / "consultas.csv")
    por_patron = leer(directorio / "resumen_por_patron.csv")
    energia = leer(directorio / "resumen_energia.csv")
    etiquetas = _etiquetas(resumen)
    figuras = directorio / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    _fig_exactitud(resumen, etiquetas, modelo, figuras / "fig_exactitud.png")
    _fig_composicion(
        resumen, etiquetas, modelo, figuras / "fig_composicion_errores.png"
    )
    _fig_iteraciones(resumen, etiquetas, modelo, figuras / "fig_iteraciones.png")
    _fig_energia(resumen, energia, etiquetas, modelo, figuras / "fig_energia.png")
    _fig_hamming(resumen, etiquetas, modelo, figuras / "fig_hamming.png")
    _fig_confusion(resumen, consultas, modelo, figuras / "fig_confusion.png")
    _fig_por_patron(
        resumen, por_patron, etiquetas, modelo, figuras / "fig_exactitud_por_patron.png"
    )
    _fig_boxplot_barridos(
        consultas, etiquetas, modelo, figuras / "fig_boxplot_barridos.png"
    )
    _fig_afectacion(consultas, etiquetas, modelo, figuras / "fig_afectacion_real.png")


def _etiquetas_pasos(combinaciones: Path) -> dict[str, str]:
    """Etiqueta de cada paso con el componente que agrega."""
    etiquetas = {}
    for carpeta in combinaciones.glob("paso_*"):
        with (carpeta / "config_usada.toml").open("rb") as archivo:
            componentes = tomllib.load(archivo)["corrida"]["componentes"]
        prefijo = "+" if len(componentes) > 1 else ""
        etiquetas[carpeta.name] = f"{len(componentes)}: {prefijo}{componentes[-1]}"
    return etiquetas


def _fig_mapa_combinaciones(por_patron, etiquetas, ruta) -> None:
    grupos = _por_regla(por_patron)
    nombres = sorted(por_patron["nombre_origen"].unique())
    fig, ejes = plt.subplots(
        1,
        len(grupos),
        figsize=(1.1 * len(etiquetas) * len(grupos) + 3, 3.8),
        layout="constrained",
    )
    ejes = np.atleast_1d(ejes)
    for eje, (regla, grupo) in zip(ejes, grupos, strict=True):
        matriz = (
            grupo.pivot(
                index="nombre_origen", columns="nivel", values="exactitud_media"
            )
            .reindex(index=nombres, columns=etiquetas)
            .to_numpy()
        )
        imagen = eje.imshow(matriz, cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto")
        for (i, j), valor in np.ndenumerate(matriz):
            eje.text(
                j,
                i,
                f"{valor:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if valor > 0.5 else "black",
            )
        eje.set_xticks(range(len(etiquetas)), etiquetas, rotation=30, ha="right")
        eje.set_yticks(range(len(nombres)), [nombre_corto(n) for n in nombres])
        eje.set_xlabel("Paso de la escalera")
        eje.grid(visible=False)
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
    fig.colorbar(imagen, ax=ejes.tolist(), label="Exactitud media", shrink=0.85)
    fig.suptitle("Combinaciones: exactitud por QR y paso (media sobre semillas)")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)


def figuras_combinaciones(combinaciones: Path) -> None:
    """Las figuras de cada paso y las de la escalera consolidada."""
    for paso in combinaciones.glob("paso_*"):
        if (paso / "resumen_niveles.csv").exists():
            figuras_modelo(paso, "combinaciones")
    consolidado = combinaciones / "consolidado"
    if not (consolidado / "resumen_niveles.csv").exists():
        return
    etiquetas_pasos = _etiquetas_pasos(combinaciones)

    def renombrar(tabla: pd.DataFrame) -> pd.DataFrame:
        return tabla.assign(nivel=tabla["nivel"].map(etiquetas_pasos))

    resumen = renombrar(leer(consolidado / "resumen_niveles.csv"))
    energia = renombrar(leer(consolidado / "resumen_energia.csv"))
    por_patron = renombrar(leer(consolidado / "resumen_por_patron.csv"))
    etiquetas = _etiquetas(resumen)
    figuras = consolidado / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    modelo = "combinaciones"
    _fig_exactitud(
        resumen, etiquetas, modelo, figuras / "fig_combinaciones_exactitud.png"
    )
    _fig_iteraciones(
        resumen, etiquetas, modelo, figuras / "fig_combinaciones_iteraciones.png"
    )
    _fig_energia(
        resumen, energia, etiquetas, modelo, figuras / "fig_combinaciones_energia.png"
    )
    _fig_hamming(resumen, etiquetas, modelo, figuras / "fig_combinaciones_hamming.png")
    _fig_mapa_combinaciones(
        por_patron, etiquetas, figuras / "fig_mapa_calor_exactitud_combinaciones.png"
    )


def _pixeles(tabla: pd.DataFrame) -> list[str]:
    return [c for c in tabla.columns if c.startswith("p") and c[1:].isdigit()]


def _leer_estados(ruta: Path) -> pd.DataFrame:
    """Lee un CSV de estados (patrones o consultas) indexado por su nombre."""
    tabla = pd.read_csv(ruta)
    return tabla.set_index(tabla.columns[0])


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
    eje.grid(visible=False)


def _matriz_anotada(eje, matriz, nombres, formato, titulo, vmin, vmax) -> None:
    eje.imshow(matriz, cmap=SECUENCIAL, vmin=vmin, vmax=vmax)
    medio = (vmin + vmax) / 2
    for (i, j), valor in np.ndenumerate(matriz):
        eje.text(
            j,
            i,
            format(valor, formato),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if valor > medio else "black",
        )
    cortos = [nombre_corto(n) for n in nombres]
    eje.set_xticks(range(len(nombres)), cortos, rotation=30, ha="right")
    eje.set_yticks(range(len(nombres)), cortos)
    eje.grid(visible=False)
    eje.set_title(titulo)


def comparacion_entrenamiento(raiz: Path, reglas: Sequence[str]) -> pd.DataFrame:
    """Une energía, estabilidad, condición y carga de las redes entrenadas."""
    tablas = []
    for regla in reglas:
        directorio = raiz / regla
        tabla = pd.read_csv(directorio / "energia_estados.csv")
        condicion = pd.read_csv(directorio / "condicion.csv").iloc[0]
        capacidad = pd.read_csv(directorio / "capacidad.csv").iloc[0]
        tabla.insert(0, "regla", regla)
        tabla["rango"] = int(condicion["rango"])
        tabla["numero_condicion"] = float(condicion["numero_condicion"])
        tabla["carga"] = float(capacidad["carga"])
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def figuras_entrenamiento(raiz: Path) -> None:
    """Comparación de entrenamiento entre las reglas: tabla y cinco figuras."""
    reglas = [
        regla
        for regla in ("hebb", "pseudoinversa")
        if (raiz / regla / "energia_estados.csv").exists()
    ]
    if not reglas:
        return
    destino = raiz / "comparacion_entrenamiento"
    destino.mkdir(parents=True, exist_ok=True)
    tabla = comparacion_entrenamiento(raiz, reglas)
    tabla.to_csv(destino / "comparacion_entrenamiento.csv", index=False)

    patrones = _leer_estados(raiz / reglas[0] / "patrones.csv")
    nombres = patrones.index.tolist()
    filas, columnas = int(patrones["filas"].iloc[0]), int(patrones["columnas"].iloc[0])
    matriz_patrones = patrones[_pixeles(patrones)].to_numpy()

    solapamiento = pd.read_csv(raiz / reglas[0] / "solapamiento.csv", index_col=0)
    hamming = pd.read_csv(raiz / reglas[0] / "hamming.csv", index_col=0)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6))
    _matriz_anotada(
        ejes[0], solapamiento.to_numpy(), nombres, ".2f", "Solapamiento ξᵘ·ξᵛ / N", 0, 1
    )
    _matriz_anotada(
        ejes[1],
        hamming.to_numpy(),
        nombres,
        "d",
        f"Distancia de Hamming (N = {filas * columnas})",
        0,
        int(hamming.to_numpy().max()),
    )
    fig.suptitle("Correlación entre los QR almacenados")
    _guardar(fig, destino / "fig_solapamiento_entrenamiento.png")

    limpias = raiz / "hebb" / "consultas" / "limpias" / "estados_finales.csv"
    if limpias.exists():
        finales = _leer_estados(limpias)
        espurio = finales[_pixeles(finales)].to_numpy()
        iguales = bool((espurio == espurio[0]).all())
        fig, ejes = plt.subplots(
            1, len(nombres) + 1, figsize=(2.4 * (len(nombres) + 1), 3.2)
        )
        _imagen(ejes[0], espurio[0], filas, columnas)
        ejes[0].set_title(
            "Estado final de Hebb\n"
            + ("(igual para las 5 consultas)" if iguales else "(primera consulta)"),
            fontsize=8,
        )
        for eje, nombre, patron in zip(ejes[1:], nombres, matriz_patrones, strict=True):
            _imagen(eje, patron, filas, columnas)
            distancia = int((patron != espurio[0]).sum())
            eje.set_title(
                f"{nombre_corto(nombre)}\nHamming al espurio: {distancia}", fontsize=8
            )
        fig.suptitle("Hebb: las consultas limpias caen en un estado espurio")
        _guardar(fig, destino / "fig_estado_espurio.png")

    fig, ejes = plt.subplots(1, len(reglas), figsize=(6 * len(reglas), 4.2))
    ejes = np.atleast_1d(ejes)
    for eje, regla in zip(ejes, reglas, strict=True):
        datos = tabla[tabla["regla"] == regla]
        colores = [
            GRIS if estado == "mezcla" else COLORES_REGLA.get(regla, GRIS)
            for estado in datos["estado"]
        ]
        posiciones = np.arange(len(datos))
        eje.bar(posiciones, datos["energia_por_neurona"], color=colores, width=0.7)
        for posicion, (valor, fijo) in zip(
            posiciones,
            zip(datos["energia_por_neurona"], datos["punto_fijo"], strict=True),
            strict=True,
        ):
            eje.annotate(
                "punto fijo" if fijo else "inestable",
                (posicion, valor),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=TINTA_SECUNDARIA,
            )
        eje.set_xticks(
            posiciones, [nombre_corto(e) for e in datos["estado"]], rotation=30
        )
        eje.set_ylabel("E / N")
        eje.set_ylim(top=0, bottom=float(datos["energia_por_neurona"].min()) * 1.2)
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
    fig.suptitle(
        "Energía de los patrones y del estado mezcla sign(Σξ) (no comparable entre reglas)"
    )
    _guardar(fig, destino / "fig_energia_entrenamiento.png")

    fig, ejes = plt.subplots(1, len(reglas), figsize=(6 * len(reglas), 4.2))
    ejes = np.atleast_1d(ejes)
    for eje, regla in zip(ejes, reglas, strict=True):
        with np.load(raiz / regla / "pesos.npz", allow_pickle=False) as datos:
            pesos = datos["pesos"]
        valores = pesos[np.triu_indices(pesos.shape[0], k=1)]
        eje.hist(valores, bins=60, color=COLORES_REGLA.get(regla, GRIS))
        eje.set_yscale("log")
        eje.set_xlabel("Peso W_ij (i < j)")
        eje.set_ylabel("Cantidad de pares (escala log)")
        eje.set_title(
            f"{NOMBRES_REGLA.get(regla, regla)}: media {valores.mean():.2e}, "
            f"desvío {valores.std():.2e}"
        )
    fig.suptitle("Distribución de los pesos (cada panel con su escala)")
    _guardar(fig, destino / "fig_pesos_histograma.png")

    fig, eje = plt.subplots(figsize=(8, 4.4))
    ancho = 0.8 / len(reglas)
    for indice, regla in enumerate(reglas):
        capacidad = pd.read_csv(raiz / regla / "capacidad_subconjuntos.csv")
        grupos = capacidad.groupby("cantidad")["todos_estables"]
        cantidades = np.asarray(sorted(capacidad["cantidad"].unique()))
        posiciones = cantidades + (indice - (len(reglas) - 1) / 2) * ancho
        barras = eje.bar(
            posiciones,
            grupos.mean().loc[cantidades],
            width=ancho * 0.95,
            color=COLORES_REGLA.get(regla, GRIS),
            label=NOMBRES_REGLA.get(regla, regla),
        )
        eje.bar_label(
            barras,
            [f"{int(grupos.sum()[c])}/{int(grupos.size()[c])}" for c in cantidades],
            fontsize=8,
        )
    eje.set_xticks(cantidades)
    eje.set_xlabel("Cantidad de QR almacenados")
    eje.set_ylabel("Fracción de subconjuntos con todos estables")
    eje.set_ylim(0, 1.15)
    eje.legend()
    eje.set_title("Capacidad práctica: todos los subconjuntos de cada tamaño")
    _guardar(fig, destino / "fig_capacidad_subconjuntos.png")


def _rejilla(cantidad: int) -> tuple[int, int]:
    return (1, cantidad) if cantidad <= 3 else (2, int(np.ceil(cantidad / 2)))


def _paneles_modelos(resumen, metrica, etiqueta_y, titulo, ruta, limites=None) -> None:
    """Un panel por modelo con la métrica de cada regla contra el nivel nominal."""
    modelos = [m for m in MODELOS if m in set(resumen["modelo"])]
    filas, columnas = _rejilla(len(modelos))
    fig, ejes = plt.subplots(
        filas, columnas, figsize=(4.8 * columnas, 3.8 * filas), squeeze=False
    )
    for eje, modelo in zip(ejes.flat, modelos, strict=False):
        datos = resumen[resumen["modelo"] == modelo]
        etiquetas = _etiquetas(datos)
        _lineas(eje, datos, metrica, etiquetas)
        _eje_niveles(eje, etiquetas, EJES_NIVEL[modelo])
        eje.set_title(NOMBRES_MODELO[modelo])
        if limites is not None:
            eje.set_ylim(*limites)
    for eje in ejes.flat[len(modelos) :]:
        eje.set_visible(False)
    for eje in ejes[:, 0]:
        eje.set_ylabel(etiqueta_y)
    ejes.flat[0].legend()
    fig.suptitle(titulo)
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _niveles_unicos(resumen: pd.DataFrame) -> pd.DataFrame:
    """Una fila por modelo y nivel (las magnitudes de la consulta no dependen de la regla)."""
    return resumen.sort_values(["modelo", "orden_nivel"]).drop_duplicates(
        ["modelo", "orden_nivel"]
    )


def _estilo_modelo(modelo: str) -> dict:
    indice = MODELOS.index(modelo)
    return {
        "color": CATEGORICOS[indice],
        "marker": MARCADORES[indice],
        "label": NOMBRES_MODELO[modelo],
    }


def _fig_comparacion_global(resumen, ruta) -> None:
    modelos = [m for m in MODELOS if m in set(resumen["modelo"])]
    reglas = sorted(resumen["regla"].unique())
    metricas = (
        ("exactitud", "Exactitud"),
        ("hamming_final_normalizado", "Hamming final / N"),
        ("barridos", "Barridos"),
    )
    fig, ejes = plt.subplots(
        len(reglas), 3, figsize=(15, 3.8 * len(reglas)), squeeze=False, sharex=True
    )
    for fila, regla in enumerate(reglas):
        for columna, (metrica, nombre) in enumerate(metricas):
            eje = ejes[fila, columna]
            for modelo in modelos:
                datos = resumen[
                    (resumen["modelo"] == modelo) & (resumen["regla"] == regla)
                ].sort_values("orden_nivel")
                eje.errorbar(
                    datos["fraccion_afectada_media"],
                    datos[f"{metrica}_media"],
                    yerr=datos[f"{metrica}_desvio"].fillna(0.0),
                    capsize=2,
                    markersize=5,
                    **_estilo_modelo(modelo),
                )
            eje.set_title(f"{NOMBRES_REGLA.get(regla, regla)}: {nombre}")
            eje.set_ylabel(nombre)
        ejes[-1, 0].set_xlabel("Fracción afectada real")
    for eje in ejes[-1]:
        eje.set_xlabel("Fracción afectada real")
    ejes[0, 0].legend(fontsize=7)
    fig.suptitle("Comparación global: métricas contra la fracción de píxeles cambiados")
    _nota(fig, resumen)
    _guardar(fig, ruta)


FRACCION_REFERENCIA = 0.10


def _fig_comparacion_reglas(resumen, ruta) -> None:
    """Exactitud de cada regla en el nivel de cada modelo más cercano al 10% real."""
    niveles = _niveles_unicos(resumen)
    elegidos = []
    for modelo in [m for m in MODELOS if m in set(niveles["modelo"])]:
        datos = niveles[niveles["modelo"] == modelo]
        distancia = (datos["fraccion_afectada_media"] - FRACCION_REFERENCIA).abs()
        elegidos.append(datos.loc[distancia.idxmin()])
    reglas = sorted(resumen["regla"].unique())
    ancho = 0.8 / len(reglas)
    fig, eje = plt.subplots(figsize=(10, 4.6))
    for indice, regla in enumerate(reglas):
        medias, desvios = [], []
        for fila in elegidos:
            dato = resumen[
                (resumen["modelo"] == fila["modelo"])
                & (resumen["orden_nivel"] == fila["orden_nivel"])
                & (resumen["regla"] == regla)
            ].iloc[0]
            medias.append(dato["exactitud_media"])
            desvios.append(
                0.0 if pd.isna(dato["exactitud_desvio"]) else dato["exactitud_desvio"]
            )
        posiciones = np.arange(len(elegidos)) + (indice - (len(reglas) - 1) / 2) * ancho
        barras = eje.bar(
            posiciones,
            medias,
            yerr=desvios,
            width=ancho * 0.95,
            capsize=3,
            color=COLORES_REGLA.get(regla, GRIS),
            label=NOMBRES_REGLA.get(regla, regla),
        )
        eje.bar_label(barras, [f"{m:.2f}" for m in medias], fontsize=8)
    eje.set_xticks(
        range(len(elegidos)),
        [
            f"{NOMBRES_MODELO[f['modelo']]}\nnivel {f['nivel']} "
            f"({f['fraccion_afectada_media']:.1%})"
            for f in elegidos
        ],
        fontsize=8,
    )
    eje.set_ylim(0, 1.15)
    eje.set_ylabel("Exactitud")
    eje.legend()
    eje.set_title(
        f"Exactitud por regla en el nivel más cercano a {FRACCION_REFERENCIA:.0%} "
        "de píxeles cambiados"
    )
    _nota(fig, resumen)
    _guardar(fig, ruta)


def _fig_mapa_calor_global(intervalos, ruta) -> None:
    modelos = [m for m in MODELOS if m in set(intervalos["modelo"])]
    reglas = sorted(intervalos["regla"].unique())
    cantidad = int(intervalos["orden_intervalo"].max()) + 1
    etiquetas = [f"{5 * k}%–{5 * (k + 1)}%" for k in range(cantidad)]
    fig, ejes = plt.subplots(
        1,
        len(reglas),
        figsize=(0.75 * cantidad * len(reglas) + 3, 0.55 * len(modelos) + 2.4),
        layout="constrained",
    )
    ejes = np.atleast_1d(ejes)
    for eje, regla in zip(ejes, reglas, strict=True):
        matriz = np.full((len(modelos), cantidad), np.nan)
        datos = intervalos[intervalos["regla"] == regla]
        for fila in datos.itertuples():
            matriz[modelos.index(fila.modelo), int(fila.orden_intervalo)] = (
                fila.exactitud_media
            )
        imagen = eje.imshow(
            np.ma.masked_invalid(matriz), cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto"
        )
        for (i, j), valor in np.ndenumerate(matriz):
            if not np.isnan(valor):
                eje.text(
                    j,
                    i,
                    f"{valor:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if valor > 0.5 else "black",
                )
        eje.set_xticks(range(cantidad), etiquetas, rotation=45, ha="right", fontsize=7)
        eje.set_yticks(range(len(modelos)), [NOMBRES_MODELO[m] for m in modelos])
        eje.set_xlabel("Fracción afectada real")
        eje.grid(visible=False)
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
    fig.colorbar(imagen, ax=ejes.tolist(), label="Exactitud media", shrink=0.85)
    fig.suptitle(
        "Exactitud por modelo e intervalo de fracción afectada (celda vacía: sin consultas)"
    )
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)


def _ejemplo_de_referencia(directorio: Path, modelo: str) -> dict:
    """Consulta de ejemplo del nivel de referencia de un modelo."""
    resumen = leer(directorio / "resumen_niveles.csv")
    referencia = nivel_referencia(resumen)
    consultas = leer(directorio / "consultas.csv")
    with np.load(directorio / "ejemplos.npz", allow_pickle=False) as datos:
        ejemplos = {clave: datos[clave] for clave in datos.files}
    indice = ejemplos["niveles"].tolist().index(referencia)
    orden = int(ejemplos["ordenes"][indice])
    patron = str(ejemplos["patron"])
    filas_ejemplo = consultas[
        (consultas["nivel"] == referencia)
        & (consultas["semilla"] == consultas["semilla"].min())
        & (consultas["consulta"] == f"{patron}_rep0")
    ].set_index("regla")
    return {
        "modelo": modelo,
        "nivel": referencia,
        "patron": patron,
        "filas": int(ejemplos["filas"]),
        "columnas": int(ejemplos["columnas"]),
        "original": ejemplos["original"],
        "inicial": ejemplos["iniciales"][indice],
        "reglas": ejemplos["reglas"].tolist(),
        "finales": {r: ejemplos[f"finales_{r}"][indice] for r in ejemplos["reglas"]},
        "energias": {r: ejemplos[f"energia_{r}_{orden}"] for r in ejemplos["reglas"]},
        "consultas": filas_ejemplo,
    }


def _fig_ejemplos(ejemplos: Sequence[dict], ruta) -> None:
    reglas = ejemplos[0]["reglas"]
    fig, ejes = plt.subplots(
        len(ejemplos),
        2 + len(reglas),
        figsize=(2.6 * (2 + len(reglas)), 2.7 * len(ejemplos)),
        squeeze=False,
    )
    for fila_ejes, ejemplo in zip(ejes, ejemplos, strict=True):
        filas, columnas = ejemplo["filas"], ejemplo["columnas"]
        _imagen(fila_ejes[0], ejemplo["original"], filas, columnas)
        fila_ejes[0].set_title(
            f"Original: {nombre_corto(ejemplo['patron'])}", fontsize=8
        )
        cambiados = int((ejemplo["inicial"] != ejemplo["original"]).sum())
        _imagen(fila_ejes[1], ejemplo["inicial"], filas, columnas)
        fila_ejes[1].set_title(
            f"{NOMBRES_MODELO[ejemplo['modelo']]} {ejemplo['nivel']}\n"
            f"({cambiados} píxeles cambiados)",
            fontsize=8,
        )
        for eje, regla in zip(fila_ejes[2:], reglas, strict=True):
            _imagen(eje, ejemplo["finales"][regla], filas, columnas)
            dato = ejemplo["consultas"].loc[regla]
            tipo = TIPOS.get(dato["tipo_resultado"], (dato["tipo_resultado"],))[0]
            eje.set_title(
                f"{NOMBRES_REGLA.get(regla, regla)}: {tipo.lower()}\n"
                f"Hamming al origen {int(dato['hamming_final_origen'])}, "
                f"{int(dato['barridos'])} barridos",
                fontsize=8,
            )
    fig.suptitle(
        "Ejemplos en el nivel de referencia de cada modelo (semilla 0, repetición 0)"
    )
    _guardar(fig, ruta)


def _fig_trayectorias(ejemplos: Sequence[dict], ruta) -> None:
    reglas = ejemplos[0]["reglas"]
    fig, ejes = plt.subplots(1, len(reglas), figsize=(6.5 * len(reglas), 4.2))
    ejes = np.atleast_1d(ejes)
    for eje, regla in zip(ejes, reglas, strict=True):
        for ejemplo in ejemplos:
            energia = np.asarray(ejemplo["energias"][regla], dtype=float)
            n = ejemplo["filas"] * ejemplo["columnas"]
            estilo = _estilo_modelo(ejemplo["modelo"])
            eje.plot(
                np.arange(energia.size) / n,
                energia / n,
                color=estilo["color"],
                linewidth=1.5,
                label=f"{estilo['label']} {ejemplo['nivel']}",
            )
        eje.set_xlabel("Barridos (actualizaciones / N)")
        eje.set_ylabel("E / N")
        eje.set_title(
            f"{NOMBRES_REGLA.get(regla, regla)}: energía en cada actualización"
        )
    ejes[0].legend(fontsize=7)
    fig.suptitle(
        "Trayectorias de energía de las consultas de ejemplo (no comparable entre reglas)"
    )
    _guardar(fig, ruta)


def _fig_basin(consultas, ruta) -> None:
    """Exactitud por intervalo de 2,5% de Hamming inicial, consulta por consulta."""
    reglas = sorted(consultas["regla"].unique())
    modelos = [m for m in MODELOS if m in set(consultas["modelo"])]
    ancho = 0.025
    datos = consultas.assign(
        intervalo=np.floor(consultas["fraccion_afectada"] / ancho).astype(int),
        correcto=consultas["tipo_resultado"] == "almacenado_correcto",
    )
    fig, ejes = plt.subplots(
        1, len(reglas), figsize=(6.5 * len(reglas), 4.2), sharey=True
    )
    ejes = np.atleast_1d(ejes)
    for eje, regla in zip(ejes, reglas, strict=True):
        for modelo in modelos:
            grupo = datos[(datos["regla"] == regla) & (datos["modelo"] == modelo)]
            exactitud = grupo.groupby("intervalo")["correcto"].mean()
            eje.plot(
                (exactitud.index + 0.5) * ancho,
                exactitud.to_numpy(),
                markersize=5,
                **_estilo_modelo(modelo),
            )
        eje.set_ylim(-0.05, 1.05)
        eje.set_xlabel("Hamming inicial al origen / N (intervalos de 2,5%)")
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
    ejes[0].set_ylabel("Exactitud (fracción de consultas)")
    ejes[-1].legend(fontsize=7)
    fig.suptitle("Cuenca de atracción: exactitud contra la distancia inicial")
    _guardar(fig, ruta)


def _fig_afectacion_global(resumen, ruta) -> None:
    niveles = _niveles_unicos(resumen)
    fig, eje = plt.subplots(figsize=(7, 5))
    maximo = float(niveles["fraccion_nominal_media"].max()) * 1.05
    eje.plot(
        [0, maximo],
        [0, maximo],
        color=GRIS,
        linestyle="--",
        linewidth=1,
        label="real = zona",
    )
    eje.plot(
        [0, maximo],
        [0, maximo / 2],
        color=GRIS,
        linestyle=":",
        linewidth=1,
        label="real = zona / 2 (mitad de píxeles negros)",
    )
    for modelo in [m for m in MODELOS if m in set(niveles["modelo"])]:
        datos = niveles[niveles["modelo"] == modelo]
        eje.plot(
            datos["fraccion_nominal_media"],
            datos["fraccion_afectada_media"],
            markersize=5,
            **_estilo_modelo(modelo),
        )
    eje.set_xlabel("Zona dañada / N (nominal)")
    eje.set_ylabel("Píxeles cambiados / N (real)")
    eje.set_title("Daño nominal contra daño real, por modelo y nivel")
    eje.legend(fontsize=7)
    _guardar(fig, ruta)


def figuras_globales(consolidado: Path, ruido: Path) -> None:
    """Las once figuras globales desde el consolidado de los modelos."""
    resumen = leer(consolidado / "resumen_niveles.csv")
    consultas = leer(consolidado / "consultas.csv")
    intervalos = pd.read_csv(consolidado / "exactitud_por_intervalo.csv")
    figuras = consolidado / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    _paneles_modelos(
        resumen,
        "exactitud",
        "Exactitud",
        "Exactitud por modelo",
        figuras / "fig_exactitud_global.png",
        (-0.05, 1.05),
    )
    _paneles_modelos(
        resumen,
        "barridos",
        "Barridos",
        "Barridos hasta estabilizar, por modelo",
        figuras / "fig_iteraciones_global.png",
    )
    _paneles_modelos(
        resumen,
        "delta_energia_por_neurona",
        "ΔE / N",
        "Cambio de energía por neurona, por modelo (no comparable entre reglas)",
        figuras / "fig_energia_global.png",
    )
    _paneles_modelos(
        resumen,
        "hamming_final_normalizado",
        "Hamming final / N",
        "Error de reconstrucción por modelo",
        figuras / "fig_hamming_global.png",
    )
    _fig_comparacion_global(resumen, figuras / "fig_comparacion_global.png")
    _fig_comparacion_reglas(resumen, figuras / "fig_comparacion_reglas.png")
    _fig_mapa_calor_global(intervalos, figuras / "fig_mapa_calor_exactitud.png")
    ejemplos = [
        _ejemplo_de_referencia(ruido / modelo, modelo)
        for modelo in MODELOS
        if (ruido / modelo / "ejemplos.npz").exists()
    ]
    _fig_ejemplos(ejemplos, figuras / "fig_ejemplos_qr.png")
    _fig_trayectorias(ejemplos, figuras / "fig_trayectorias_energia_ejemplo.png")
    _fig_basin(consultas, figuras / "fig_basin_hamming.png")
    _fig_afectacion_global(resumen, figuras / "fig_afectacion_real_global.png")
    if (ruido / "legibilidad_patrones.csv").exists():
        _fig_lectura_ejemplos(ruido, figuras / "fig_lectura_ejemplos.png")


def _textos_leidos(ruido: Path) -> tuple[dict[str, str], str]:
    """Texto de cada QR almacenado y el prefijo común de esos textos."""
    patrones = pd.read_csv(ruido / "legibilidad_patrones.csv", keep_default_na=False)
    textos = dict(zip(patrones["estado"], patrones["texto"], strict=True))
    leidos = [t for estado, t in textos.items() if t and estado != "mezcla"]
    prefijo = os.path.commonprefix(leidos) if len(leidos) > 1 else ""
    return textos, prefijo


def _fig_lectura_ejemplos(ruido: Path, ruta: Path) -> None:
    """Consulta dañada que no se lee y estados recuperados que sí, por modelo."""
    textos, prefijo = _textos_leidos(ruido)

    def leido(lectura: str) -> str:
        if not lectura:
            return "no se lee"
        texto = textos.get(lectura, lectura)
        return "se lee: …" + texto[len(prefijo) :] if prefijo else f"se lee: {texto}"

    casos = []
    for modelo in MODELOS:
        directorio = ruido / modelo
        if not (directorio / "legibilidad.csv").exists():
            continue
        tabla = pd.read_csv(
            directorio / "legibilidad.csv", dtype={"nivel": str}, keep_default_na=False
        )
        with np.load(directorio / "ejemplos.npz", allow_pickle=False) as datos:
            ejemplos = {clave: datos[clave] for clave in datos.files}
        patron = str(ejemplos["patron"])
        ejemplo = tabla[
            (tabla["semilla"] == tabla["semilla"].min())
            & (tabla["consulta"] == f"{patron}_rep0")
        ]
        reglas = sorted(ejemplo["regla"].unique(), key=lambda r: r != "pseudoinversa")
        guia = ejemplo[ejemplo["regla"] == reglas[0]].sort_values("orden_nivel")
        # El primer nivel en que la consulta dañada no se lee y la recuperada sí;
        # si no hay, el nivel central.
        elegibles = guia[~guia["legible_inicial"] & guia["lee_origen_final"]]
        orden = int(
            (elegibles if not elegibles.empty else guia)["orden_nivel"].iloc[
                0 if not elegibles.empty else len(guia) // 2
            ]
        )
        indice = ejemplos["ordenes"].tolist().index(orden)
        filas_caso = ejemplo[ejemplo["orden_nivel"] == orden].set_index("regla")
        casos.append((modelo, ejemplos, indice, reglas, filas_caso))
    if not casos:
        return

    patrones = _leer_estados(ruido.parent / casos[0][3][-1] / "patrones.csv")
    mezcla = np.where(patrones[_pixeles(patrones)].to_numpy().sum(axis=0) >= 0, 1, -1)
    reglas = casos[0][3]
    fig, ejes = plt.subplots(
        len(casos),
        2 + len(reglas),
        figsize=(2.9 * (2 + len(reglas)), 3.0 * len(casos)),
        squeeze=False,
    )
    for fila_ejes, (modelo, ejemplos, indice, _, filas_caso) in zip(
        ejes, casos, strict=True
    ):
        filas, columnas = int(ejemplos["filas"]), int(ejemplos["columnas"])
        patron = str(ejemplos["patron"])
        primera = filas_caso.iloc[0]
        _imagen(fila_ejes[0], ejemplos["original"], filas, columnas)
        fila_ejes[0].set_title(
            f"Original\n{leido(patron if textos.get(patron) else '')}", fontsize=8
        )
        _imagen(fila_ejes[1], ejemplos["iniciales"][indice], filas, columnas)
        fila_ejes[1].set_title(
            f"{NOMBRES_MODELO[modelo]} {primera['nivel']}\n{leido(primera['lectura_inicial'])}",
            fontsize=8,
        )
        for eje, regla in zip(fila_ejes[2:], reglas, strict=True):
            final = ejemplos[f"finales_{regla}"][indice]
            _imagen(eje, final, filas, columnas)
            nota = " (estado mezcla)" if np.array_equal(final, mezcla) else ""
            eje.set_title(
                f"{NOMBRES_REGLA.get(regla, regla)}{nota}\n"
                f"{leido(filas_caso.loc[regla, 'lectura_final'])}",
                fontsize=8,
            )
    fig.suptitle(
        "Lectura con un lector de QR (OpenCV): consulta dañada y estado recuperado"
    )
    if prefijo:
        fig.text(
            0.01,
            0.005,
            f"Prefijo común de las URL: {prefijo}",
            fontsize=7,
            color=TINTA_SECUNDARIA,
        )
    _guardar(fig, ruta)


def _legibilidad_con_fraccion(consolidado: Path) -> pd.DataFrame:
    legibilidad = leer(consolidado / "legibilidad_resumen.csv")
    resumen = leer(consolidado / "resumen_niveles.csv")
    claves = ["modelo", "orden_nivel", "regla"]
    return legibilidad.merge(resumen[[*claves, "fraccion_afectada_media"]], on=claves)


def _fig_legibilidad(consolidados: Sequence[tuple[Path, str]], ruta: Path) -> None:
    """QR legibles contra la fracción afectada real: dañados y recuperados."""
    tablas = [
        (_legibilidad_con_fraccion(c), estilo)
        for c, estilo in consolidados
        if (c / "legibilidad_resumen.csv").exists()
    ]
    if not tablas:
        return
    reglas = sorted(
        pd.concat([t for t, _ in tablas])["regla"].unique(),
        key=lambda r: r != "pseudoinversa",
    )
    paneles = [("legible_inicial", reglas[0], "Consulta dañada: se lee")]
    paneles += [
        (
            "lee_origen_final",
            regla,
            f"{NOMBRES_REGLA.get(regla, regla)}: se lee el QR de origen",
        )
        for regla in reglas
    ]
    fig, ejes = plt.subplots(
        1, len(paneles), figsize=(5.6 * len(paneles), 4.4), sharey=True
    )
    for eje, (metrica, regla, titulo) in zip(ejes, paneles, strict=True):
        for tabla, estilo in tablas:
            for modelo in [m for m in MODELOS if m in set(tabla["modelo"])]:
                datos = tabla[(tabla["modelo"] == modelo) & (tabla["regla"] == regla)]
                datos = datos.sort_values("orden_nivel")
                parametros = _estilo_modelo(modelo)
                eje.errorbar(
                    datos["fraccion_afectada_media"],
                    datos[f"{metrica}_media"],
                    yerr=datos[f"{metrica}_desvio"].fillna(0.0),
                    color=parametros["color"],
                    marker=parametros["marker"],
                    markerfacecolor=parametros["color"] if estilo == "-" else "white",
                    linestyle=estilo,
                    capsize=2,
                    markersize=5,
                )
        eje.set_ylim(-0.05, 1.05)
        eje.set_xlabel("Fracción afectada real")
        eje.set_title(titulo)
    ejes[0].set_ylabel("Proporción de QR legibles")
    presentes = [m for m in MODELOS if any(m in set(t["modelo"]) for t, _ in tablas)]
    manejadores = [
        matplotlib.lines.Line2D(
            [],
            [],
            color=_estilo_modelo(m)["color"],
            marker=_estilo_modelo(m)["marker"],
            label=_estilo_modelo(m)["label"],
        )
        for m in presentes
    ]
    manejadores += [
        matplotlib.lines.Line2D(
            [], [], color=GRIS, linestyle="-", label="Niveles a priori"
        ),
        matplotlib.lines.Line2D(
            [],
            [],
            color=GRIS,
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            label="Exploratorio (posterior)",
        ),
    ]
    ejes[-1].legend(handles=manejadores, fontsize=7)
    fig.suptitle("Lectura con OpenCV: consulta dañada contra estado recuperado")
    primero = next(
        c for c, _ in consolidados if (c / "legibilidad_resumen.csv").exists()
    )
    _nota(fig, leer(primero / "resumen_niveles.csv"))
    _guardar(fig, ruta)


def tablas_quiebre(consultas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactitud por esquina borrada y según el signo del margen inicial."""
    datos = consultas.assign(
        correcto=consultas["tipo_resultado"] == "almacenado_correcto"
    )
    esquinas = datos[datos["modelo"] == "esquina"]
    esquinas = esquinas.assign(
        esquina=esquinas["parametros"].map(
            lambda texto: json.loads(texto)["esquina"]["esquina"]
        )
    )
    por_esquina = (
        esquinas.groupby(["regla", "orden_nivel", "nivel", "esquina"])
        .agg(exactitud=("correcto", "mean"), consultas=("correcto", "size"))
        .reset_index()
    )
    por_margen = (
        datos.assign(margen_positivo=datos["margen_inicial"] > 0)
        .groupby(["regla", "margen_positivo"])
        .agg(
            consultas=("correcto", "size"),
            exactitud=("correcto", "mean"),
            margen_medio=("margen_inicial", "mean"),
        )
        .reset_index()
    )
    return por_esquina, por_margen


def _par_mas_confundido(
    consultas: pd.DataFrame, patrones: pd.DataFrame
) -> tuple[str, str]:
    """Par origen → final más frecuente entre los errores; si no hay, el par más cercano.

    Un empate en la cantidad de errores se resuelve por el par más parecido.
    """
    matriz = patrones[_pixeles(patrones)].to_numpy()
    distancias = (matriz[:, None, :] != matriz[None, :, :]).sum(axis=2)
    np.fill_diagonal(distancias, distancias.max() + 1)
    nombres = patrones.index.tolist()
    errores = consultas[consultas["tipo_resultado"] == "almacenado_incorrecto"]
    if not errores.empty:
        conteo = errores.groupby(["nombre_origen", "nombre_final"]).size()
        candidatos = conteo[conteo == conteo.max()].index
        par = min(
            candidatos,
            key=lambda p: distancias[nombres.index(p[0]), nombres.index(p[1])],
        )
        return str(par[0]), str(par[1])
    i, j = np.unravel_index(np.argmin(distancias), distancias.shape)
    return str(nombres[i]), str(nombres[j])


def _fig_zonas(consultas, por_esquina, patrones, ruta) -> None:
    reglas = sorted(consultas["regla"].unique())
    regla = "pseudoinversa" if "pseudoinversa" in reglas else reglas[0]
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.8))

    datos = por_esquina[por_esquina["regla"] == regla]
    etiquetas = _etiquetas(datos)
    esquinas = sorted(datos["esquina"].unique())
    matriz = (
        datos.pivot(index="esquina", columns="nivel", values="exactitud")
        .reindex(index=esquinas, columns=etiquetas)
        .to_numpy()
    )
    ejes[0].imshow(matriz, cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto")
    for (i, j), valor in np.ndenumerate(matriz):
        ejes[0].text(
            j,
            i,
            f"{valor:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if valor > 0.5 else "black",
        )
    ejes[0].set_xticks(range(len(etiquetas)), etiquetas)
    ejes[0].set_yticks(range(len(esquinas)), [e.replace("_", " ") for e in esquinas])
    ejes[0].set_xlabel("Lado L de la esquina borrada")
    ejes[0].grid(visible=False)
    ejes[0].set_title(f"{NOMBRES_REGLA.get(regla, regla)}: exactitud por esquina")

    margen = consultas["margen_inicial"].clip(-60, 60)
    intervalos = np.floor(margen / 10) * 10 + 5
    datos_margen = consultas.assign(
        intervalo=intervalos,
        correcto=consultas["tipo_resultado"] == "almacenado_correcto",
    )
    for nombre_regla, grupo in datos_margen.groupby("regla"):
        exactitud = grupo.groupby("intervalo")["correcto"].mean()
        ejes[1].plot(
            exactitud.index,
            exactitud.to_numpy(),
            color=COLORES_REGLA.get(nombre_regla, GRIS),
            marker=MARCADORES_REGLA.get(nombre_regla, "o"),
            label=NOMBRES_REGLA.get(nombre_regla, nombre_regla),
        )
    ejes[1].axvline(0, color=GRIS, linestyle="--", linewidth=1)
    ejes[1].set_ylim(-0.05, 1.05)
    ejes[1].set_xlabel("Margen inicial: d(otro QR más cercano) − d(origen), en bits")
    ejes[1].set_ylabel("Exactitud")
    ejes[1].set_title("Exactitud según el margen de la consulta dañada")
    ejes[1].legend()

    # El par más confundido al borrar esquinas, para leerlo junto al mapa.
    esquina = consultas[
        (consultas["regla"] == regla) & (consultas["modelo"] == "esquina")
    ]
    origen, otro = _par_mas_confundido(esquina, patrones)
    filas, columnas = int(patrones["filas"].iloc[0]), int(patrones["columnas"].iloc[0])
    a = patrones.loc[origen, _pixeles(patrones)].to_numpy().reshape(filas, columnas)
    b = patrones.loc[otro, _pixeles(patrones)].to_numpy().reshape(filas, columnas)
    imagen = np.ones((filas, columnas, 3))
    imagen[a == 1] = (0.82, 0.82, 0.80)
    imagen[a != b] = matplotlib.colors.to_rgb(CATEGORICOS[1])
    ejes[2].imshow(imagen, interpolation="nearest")
    ejes[2].set_xticks([])
    ejes[2].set_yticks([])
    ejes[2].grid(visible=False)
    ejes[2].set_title(
        f"Píxeles que distinguen {nombre_corto(origen)} de {nombre_corto(otro)}: "
        f"{int((a != b).sum())} (naranja)",
        fontsize=9,
    )
    fig.suptitle("Barrido exploratorio: la caída depende de qué zona se borra")
    _guardar(fig, ruta)


def _fig_quiebre_vs_afectada(a_priori, exploratorio, ruta) -> None:
    tablas = [
        (t, estilo)
        for t, estilo in ((a_priori, "-"), (exploratorio, "--"))
        if t is not None
    ]
    reglas = sorted(pd.concat([t for t, _ in tablas])["regla"].unique())
    fig, ejes = plt.subplots(
        1, len(reglas), figsize=(7 * len(reglas), 4.8), sharey=True
    )
    ejes = np.atleast_1d(ejes)
    for eje, regla in zip(ejes, reglas, strict=True):
        for tabla, estilo in tablas:
            for modelo in [m for m in MODELOS if m in set(tabla["modelo"])]:
                datos = tabla[
                    (tabla["modelo"] == modelo) & (tabla["regla"] == regla)
                ].sort_values("orden_nivel")
                parametros = _estilo_modelo(modelo)
                eje.errorbar(
                    datos["fraccion_afectada_media"],
                    datos["exactitud_media"],
                    yerr=datos["exactitud_desvio"].fillna(0.0),
                    color=parametros["color"],
                    marker=parametros["marker"],
                    markerfacecolor=parametros["color"] if estilo == "-" else "white",
                    linestyle=estilo,
                    capsize=2,
                    markersize=5,
                )
        eje.set_ylim(-0.05, 1.05)
        eje.set_xlabel("Fracción afectada real")
        eje.set_title(NOMBRES_REGLA.get(regla, regla))
    ejes[0].set_ylabel("Exactitud")
    presentes = [m for m in MODELOS if any(m in set(t["modelo"]) for t, _ in tablas)]
    manejadores = [
        matplotlib.lines.Line2D(
            [],
            [],
            color=_estilo_modelo(m)["color"],
            marker=_estilo_modelo(m)["marker"],
            label=_estilo_modelo(m)["label"],
        )
        for m in presentes
    ]
    manejadores += [
        matplotlib.lines.Line2D(
            [], [], color=GRIS, linestyle="-", label="Niveles a priori"
        ),
        matplotlib.lines.Line2D(
            [],
            [],
            color=GRIS,
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            label="Exploratorio (posterior)",
        ),
    ]
    ejes[-1].legend(handles=manejadores, fontsize=7)
    fig.suptitle(
        "Exactitud contra fracción afectada real: niveles a priori y exploratorios"
    )
    _guardar(fig, ruta)


def figuras_quiebre(raiz: Path) -> None:
    """Figuras y tablas del barrido exploratorio del quiebre estructural."""
    quiebre = raiz / "ruido_quiebre"
    for modelo in MODELOS:
        if (quiebre / modelo / "resumen_niveles.csv").exists():
            figuras_modelo(quiebre / modelo, modelo)
    consolidado = quiebre / "consolidado"
    if not (consolidado / "resumen_niveles.csv").exists():
        return
    consultas = leer(consolidado / "consultas.csv")
    por_esquina, por_margen = tablas_quiebre(consultas)
    por_esquina.to_csv(consolidado / "exactitud_por_esquina.csv", index=False)
    por_margen.to_csv(consolidado / "exactitud_por_margen.csv", index=False)
    figuras = consolidado / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    a_priori = raiz / "ruido" / "consolidado" / "resumen_niveles.csv"
    _fig_quiebre_vs_afectada(
        leer(a_priori) if a_priori.exists() else None,
        leer(consolidado / "resumen_niveles.csv"),
        figuras / "fig_quiebre_exactitud_vs_afectada.png",
    )
    _fig_legibilidad(
        [(raiz / "ruido" / "consolidado", "-"), (consolidado, "--")],
        figuras / "fig_legibilidad.png",
    )
    redes = [
        r for r in ("hebb", "pseudoinversa") if (raiz / r / "patrones.csv").exists()
    ]
    if not por_esquina.empty and redes:
        _fig_zonas(
            consultas,
            por_esquina,
            _leer_estados(raiz / redes[0] / "patrones.csv"),
            figuras / "fig_quiebre_zonas.png",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Genera las figuras de todos los resultados que existan."""
    parser = argparse.ArgumentParser(
        description="Figuras de la recuperación de QR; solo lee resultados."
    )
    parser.add_argument("--raiz", type=Path, default=RUTA_RESULTADOS)
    parser.add_argument(
        "--solo",
        choices=("entrenamiento", "modelos", "combinaciones", "global", "quiebre"),
        default=None,
    )
    argumentos = parser.parse_args(argv)
    raiz = resolver_ruta(argumentos.raiz)

    if argumentos.solo in (None, "entrenamiento"):
        figuras_entrenamiento(raiz)
        print(
            f"Figuras de entrenamiento en {raiz / 'comparacion_entrenamiento'}",
            file=sys.stderr,
        )

    if argumentos.solo in (None, "modelos"):
        for modelo in MODELOS:
            directorio = raiz / "ruido" / modelo
            if (directorio / "resumen_niveles.csv").exists():
                figuras_modelo(directorio, modelo)
                print(
                    f"Figuras de {modelo} en {directorio / 'figuras'}", file=sys.stderr
                )
    if argumentos.solo in (None, "combinaciones"):
        combinaciones = raiz / "ruido" / "combinaciones"
        if combinaciones.exists():
            figuras_combinaciones(combinaciones)
            print(f"Figuras de combinaciones en {combinaciones}", file=sys.stderr)
    if argumentos.solo in (None, "global"):
        consolidado = raiz / "ruido" / "consolidado"
        if (consolidado / "resumen_niveles.csv").exists():
            figuras_globales(consolidado, raiz / "ruido")
            print(f"Figuras globales en {consolidado / 'figuras'}", file=sys.stderr)
    if argumentos.solo in (None, "quiebre") and (raiz / "ruido_quiebre").exists():
        figuras_quiebre(raiz)
        print(
            f"Figuras del barrido exploratorio en {raiz / 'ruido_quiebre'}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
