#!/usr/bin/env python3
"""Decisión para las campañas de marketing (TP4, ejercicio 1.4).

Script de análisis: solo lee resultados de 1.1 (cargas del PCA) y 1.2 (pesos y
asignaciones de Kohonen); no entrena nada.

Supuesto central: solo compra/adopta el público argentino, así que el alcance
útil de un creador es ``estimated_argentina_reach × % de posts del tópico``.
Guardados + visitas al perfil se toman como señales de intención de compra.

Genera cinco figuras:

- ``fig_mapa_regiones``: tópico dominante, tamaño y audiencia argentina de cada
  neurona del mapa (dónde está cada cliente).
- ``fig_enfoques``: para cada cliente, sus creadores según tamaño frente a
  costo por argentino alcanzado (masivo), alcance a no seguidores
  (descubrimiento) e intención (profundidad).
- ``fig_followers``: qué se elige al ordenar solo por followers.
- ``fig_comparacion``: selección propuesta frente al top por followers.
- ``fig_actualizacion``: cuántos creadores cambian de región si los datos
  varían de un mes a otro, sin reentrenar.

La propuesta depende de la región del mapa, que cambia con la semilla; se
reporta como media ± desvío sobre semillas.
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ / "src" / "ej1" / "campanas.toml"
RUTA_SALIDA = RAIZ / "resultados" / "ej1" / "campanas"

TOPICOS = ["topic_football_pct", "topic_dogs_pct", "topic_makeup_pct", "topic_art_pct"]
NOMBRES_TOPICO = {
    "topic_football_pct": "Fútbol",
    "topic_dogs_pct": "Perros",
    "topic_makeup_pct": "Maquillaje",
    "topic_art_pct": "Arte",
}
# Paleta categórica validada para daltonismo; el marcador es la codificación
# secundaria.
COLOR_TOPICO = {
    "topic_football_pct": "#1baf7a",
    "topic_dogs_pct": "#eda100",
    "topic_makeup_pct": "#d95926",
    "topic_art_pct": "#2a78d6",
}
MARCADOR_TOPICO = {
    "topic_football_pct": "s",
    "topic_dogs_pct": "D",
    "topic_makeup_pct": "^",
    "topic_art_pct": "o",
}
ESTRATEGIAS = {"followers": "Top por followers", "propuesta": "Nuestra propuesta"}
COLOR_ESTRATEGIA = {"followers": "#8f8e88", "propuesta": "#6250d6"}
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRILLA = "#e4e3df"
# Secuencial de un solo tono para la audiencia argentina, sin el extremo blanco.
MAPA_ARGENTINA = LinearSegmentedColormap.from_list(
    "argentina", plt.get_cmap("Purples")(np.linspace(0.25, 1.0, 256))
)
TICKS_FOLLOWERS = [1e4, 1e5, 1e6]
TOP_FOLLOWERS_FIGURA = 10

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlecolor": TINTA,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "axes.grid": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    return camino if camino.is_absolute() else RAIZ / camino


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Decisión para las campañas (ejercicio 1.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=RUTA_CONFIG)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    parser.add_argument("--repeticiones", type=int, default=None)
    return parser.parse_args(argv)


# --- Métricas y selección ---------------------------------------------------


def _z(serie: pd.Series) -> pd.Series:
    desvio = serie.std(ddof=1)
    return (serie - serie.mean()) / desvio if desvio > 0 else serie * 0.0


def agregar_metricas(datos: pd.DataFrame, topico: str) -> pd.DataFrame:
    """Agrega alcance argentino en el tópico, su costo y la intención."""
    datos = datos.copy()
    datos["argentinos_topico"] = (
        datos["estimated_argentina_reach"] * datos[topico] / 100
    )
    datos["usd_por_mil_argentinos"] = (
        datos["price_usd"] / datos["argentinos_topico"] * 1000
    )
    datos["intencion"] = datos["saves_per_1000"] + datos["profile_visits_per_1000"]
    return datos


def puntaje(datos: pd.DataFrame, pesos: Mapping[str, float]) -> pd.Series:
    """Promedio ponderado de z-scores; un peso negativo penaliza la métrica."""
    total = sum(abs(peso) for peso in pesos.values())
    return sum(peso * _z(datos[m]) for m, peso in pesos.items()) / total


def topico_dominante(
    datos: pd.DataFrame, asignaciones: pd.DataFrame, topicos: Sequence[str]
) -> pd.Series:
    """Tópico de mayor promedio entre los creadores de cada neurona."""
    unidas = asignaciones[["creator_id", "neurona"]].merge(datos, on="creator_id")
    return unidas.groupby("neurona")[list(topicos)].mean().idxmax(axis=1)


def seleccionar(puntajes: pd.Series, candidatos: Sequence[str], n: int) -> list[str]:
    """Los ``n`` candidatos de mayor puntaje, en orden."""
    return list(puntajes.loc[list(candidatos)].sort_values(ascending=False).index[:n])


def seleccionar_con_presupuesto(
    orden: Sequence[str], precios: pd.Series, presupuesto: float
) -> list[str]:
    """Recorre ``orden`` y agrega a cada creador que todavía entra en el presupuesto."""
    elegidos, gasto = [], 0.0
    for creador in orden:
        precio = float(precios[creador])
        if gasto + precio <= presupuesto:
            elegidos.append(creador)
            gasto += precio
    return elegidos


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Proporción de creadores en común entre dos selecciones."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def metricas_seleccion(
    datos: pd.DataFrame, ids: Sequence[str], topico: str
) -> dict[str, float]:
    """Resultado de contratar a ``ids`` (``datos`` con las métricas del cliente)."""
    elegidos = datos.loc[list(ids)]
    costo = float(elegidos["price_usd"].sum())
    argentinos = float(elegidos["argentinos_topico"].sum())
    intencion = float(
        (elegidos["intencion"] / 1000 * elegidos["argentinos_topico"]).sum()
    )
    return {
        "creadores": float(len(elegidos)),
        "costo_total": costo,
        "argentinos_topico": argentinos,
        "usd_por_mil_argentinos": costo / argentinos * 1000,
        "intencion_cada_100_usd": intencion / costo * 100,
        "afinidad_media": float(elegidos[topico].mean()),
        "audiencia_argentina_media": float(elegidos["argentina_audience_pct"].mean()),
    }


# --- Ubicación en el mapa y actualización mensual ----------------------------


def ubicar(puntos: np.ndarray, pesos: np.ndarray, distancia: str) -> np.ndarray:
    """Índice de la neurona ganadora de cada punto."""
    if distancia == "exponencial":
        puntos = puntos / np.linalg.norm(puntos, axis=1, keepdims=True)
        pesos = pesos / np.linalg.norm(pesos, axis=1, keepdims=True)
    distancias = ((puntos[:, None, :] - pesos[None, :, :]) ** 2).sum(axis=2)
    return distancias.argmin(axis=1)


def perturbar(
    datos: pd.DataFrame, sigma: float, generador: np.random.Generator
) -> pd.DataFrame:
    """Multiplica cada valor por ``1 + sigma·ε`` y recorta a rangos válidos."""
    ruido = generador.normal(size=datos.shape)
    perturbados = (datos * (1 + sigma * ruido)).clip(lower=0)
    porcentajes = [c for c in datos.columns if c.endswith("_pct")]
    perturbados[porcentajes] = perturbados[porcentajes].clip(upper=100)
    return perturbados


class Proyector:
    """Lleva variables originales a la entrada del mapa (z-score → PCs)."""

    def __init__(self, datos: pd.DataFrame, cargas: pd.DataFrame, pcs: list[str]):
        self.variables = list(cargas.index)
        matriz = datos[self.variables].to_numpy(dtype=float)
        self.media = matriz.mean(axis=0)
        self.desvio = matriz.std(axis=0, ddof=1)
        self.cargas = cargas[pcs].to_numpy(dtype=float)

    def __call__(self, datos: pd.DataFrame) -> np.ndarray:
        matriz = datos[self.variables].to_numpy(dtype=float)
        return ((matriz - self.media) / self.desvio) @ self.cargas


# --- Figuras -----------------------------------------------------------------


def _guardar(fig: plt.Figure, ruta: Path, extra: Sequence = ()) -> None:
    # Con una lista vacía matplotlib dejaría afuera el título general.
    adicionales = {"bbox_extra_artists": list(extra)} if extra else {}
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white", **adicionales)
    plt.close(fig)


def _miles(valor: float) -> str:
    if valor >= 1e6:
        return f"{valor / 1e6:g} M".replace(".", ",")
    return f"{valor / 1e3:.0f} mil"


def _coma(valor: float, formato: str = "+.2f") -> str:
    """Número con coma decimal."""
    return format(valor, formato).replace(".", ",")


def _eje_followers(eje: plt.Axes) -> None:
    eje.set_xscale("log")
    eje.set_xlim(7e3, 1.3e6)
    eje.xaxis.set_major_locator(FixedLocator(TICKS_FOLLOWERS))
    eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _miles(v)))
    eje.xaxis.set_minor_formatter(NullFormatter())


def _tinte(color: str, intensidad: float) -> tuple[float, float, float]:
    """Mezcla ``color`` con blanco; 0 es blanco y 1 el color puro."""
    return tuple(1 - intensidad * (1 - c) for c in to_rgb(color))


def _figura_mapa_regiones(
    datos: pd.DataFrame, asignadas: pd.DataFrame, k: int, ruta: Path
) -> None:
    unidas = asignadas.merge(datos, on="creator_id")
    grupos = unidas.groupby(["fila", "columna"])
    medias = grupos[TOPICOS + ["followers", "argentina_audience_pct"]].mean()
    conteos = grupos.size()

    fig, eje = plt.subplots(figsize=(9, 8.8))
    for fila, columna in np.ndindex(k, k):
        esquina = (columna - 0.5, fila - 0.5)
        if (fila, columna) not in medias.index:
            eje.add_patch(
                Rectangle(esquina, 1, 1, facecolor="#f1f0ec", edgecolor="white", lw=4)
            )
            eje.text(
                columna,
                fila,
                "sin creadores",
                ha="center",
                va="center",
                fontsize=10,
                color=TINTA_SECUNDARIA,
            )
            continue
        valores = medias.loc[(fila, columna)]
        topico = valores[TOPICOS].idxmax()
        dominio = valores[topico]
        relleno = _tinte(COLOR_TOPICO[topico], 0.2 + 0.5 * dominio / 100)
        eje.add_patch(
            Rectangle(esquina, 1, 1, facecolor=relleno, edgecolor="white", lw=4)
        )
        eje.text(
            columna,
            fila - 0.22,
            f"{NOMBRES_TOPICO[topico]} {dominio:.0f}%",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=TINTA,
        )
        mixta = "mixta · " if dominio < 40 else ""
        eje.text(
            columna,
            fila + 0.14,
            f"{mixta}{_miles(valores['followers'])} seg.\n"
            f"{valores['argentina_audience_pct']:.0f}% arg. · "
            f"n={conteos[(fila, columna)]}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=TINTA_SECUNDARIA,
        )
    eje.set_xlim(-0.5, k - 0.5)
    eje.set_ylim(k - 0.5, -0.5)
    eje.set_xticks(range(k))
    eje.set_yticks(range(k))
    eje.set_xlabel("Columna de la grilla")
    eje.set_ylabel("Fila de la grilla")
    eje.grid(False)
    eje.set_aspect("equal")
    eje.legend(
        handles=[
            Patch(color=COLOR_TOPICO[t], label=NOMBRES_TOPICO[t]) for t in TOPICOS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
    )
    eje.set_title(
        "Mapa de Kohonen: tópico dominante de cada neurona\n"
        "(color más intenso = más especializada; mixta = tópico dominante < 40%;\n"
        "seguidores y audiencia argentina medios de sus creadores)",
    )
    _guardar(fig, ruta)


def _figura_enfoques(
    tablas: Mapping[str, pd.DataFrame], clientes: Mapping[str, dict], ruta: Path
) -> None:
    filas = [
        (
            "usd_por_mil_argentinos",
            "MASIVO\nUSD por mil argentinos\nalcanzados en el tópico\n(↓ mejor)",
            True,
        ),
        (
            "non_follower_reach_pct",
            "DESCUBRIMIENTO\n% del alcance que\nno sigue al creador\n(↑ mejor)",
            False,
        ),
        (
            "intencion",
            "PROFUNDIDAD\nguardados + visitas\nal perfil cada mil\nalcanzados (↑ mejor)",
            False,
        ),
    ]
    normal = Normalize(20, 95)
    fig, ejes = plt.subplots(
        len(filas),
        len(clientes),
        figsize=(3.6 * len(clientes) + 2.6, 11.5),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for columna, (clave, cliente) in enumerate(clientes.items()):
        tabla = tablas[clave]
        for fila, (metrica, etiqueta, logaritmica) in enumerate(filas):
            eje = ejes[fila, columna]
            eje.scatter(
                tabla["followers"],
                tabla[metrica],
                c=tabla["argentina_audience_pct"],
                cmap=MAPA_ARGENTINA,
                norm=normal,
                s=80,
                edgecolor=TINTA_SECUNDARIA,
                lw=0.6,
                zorder=2,
            )
            y = np.log(tabla[metrica]) if logaritmica else tabla[metrica]
            r = np.corrcoef(np.log(tabla["followers"]), y)[0, 1]
            eje.text(
                0.03,
                0.96,
                f"r = {_coma(r)}",
                transform=eje.transAxes,
                va="top",
                fontsize=10,
                color=TINTA_SECUNDARIA,
            )
            if logaritmica:
                eje.set_yscale("log")
                eje.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200]))
                eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                eje.yaxis.set_minor_formatter(NullFormatter())
                ordenados = tabla.sort_values(metrica)
                destacados = pd.concat([ordenados.head(3), ordenados.tail(1)])
                puntos = list(
                    destacados.sort_values(metrica, ascending=False).itertuples()
                )
                # Los de la derecha se etiquetan hacia la izquierda. Si dos
                # etiquetas quedan pegadas: si apuntan al mismo lado sube la de
                # arriba; si apuntan a lados opuestos baja la de abajo.
                izquierdas = [c.followers > 2.5e5 for c in puntos]
                alturas = [-3] * len(puntos)
                for i in range(1, len(puntos)):
                    arriba, abajo = puntos[i - 1], puntos[i]
                    if (
                        abs(np.log(abajo.followers / arriba.followers)) < 0.7
                        and abs(
                            np.log(getattr(abajo, metrica) / getattr(arriba, metrica))
                        )
                        < 0.25
                    ):
                        if izquierdas[i - 1] == izquierdas[i]:
                            alturas[i - 1] = 8
                        else:
                            alturas[i] = -14
                for creador, altura, izquierda in zip(
                    puntos, alturas, izquierdas, strict=True
                ):
                    eje.annotate(
                        creador.name,
                        (creador.followers, getattr(creador, metrica)),
                        xytext=(-7 if izquierda else 7, altura),
                        textcoords="offset points",
                        ha="right" if izquierda else "left",
                        fontsize=8.5,
                        color=TINTA,
                    )
            _eje_followers(eje)
            if columna == 0:
                eje.set_ylabel(etiqueta, fontsize=10.5, color=TINTA)
        ejes[0, columna].set_title(f"{cliente['nombre']} (n={len(tabla)})", fontsize=13)
        ejes[-1, columna].set_xlabel("Followers (escala log)")
    # Lugar arriba de cada fila para el texto de la correlación.
    for fila, (metrica, _, logaritmica) in enumerate(filas):
        valores = pd.concat([tabla[metrica] for tabla in tablas.values()])
        if logaritmica:
            ejes[fila, 0].set_ylim(valores.min() * 0.8, valores.max() * 1.8)
        else:
            rango = valores.max() - valores.min()
            ejes[fila, 0].set_ylim(
                valores.min() - 0.06 * rango, valores.max() + 0.22 * rango
            )
    barra = fig.colorbar(
        plt.cm.ScalarMappable(norm=normal, cmap=MAPA_ARGENTINA),
        ax=ejes,
        shrink=0.5,
        pad=0.015,
    )
    barra.set_label("% de la audiencia en Argentina")
    if len(clientes) > 1:
        fig.suptitle(
            "¿Masivo, descubrimiento o profundidad? Cada punto es un creador cuyo "
            "tema principal es el del cliente\n"
            "r: correlación con log(followers). Etiquetas: los 3 más baratos por "
            "argentino y el más caro",
            fontsize=13,
        )
    _guardar(fig, ruta)


def _figura_followers(datos: pd.DataFrame, top: Sequence[str], ruta: Path) -> None:
    fig, eje = plt.subplots(figsize=(12, 7))
    dominante = datos[TOPICOS].idxmax(axis=1)

    def tamano(precio):
        return 25 + 500 * precio / datos["price_usd"].max()

    for topico in TOPICOS:
        grupo = datos[dominante == topico]
        eje.scatter(
            grupo["followers"],
            grupo["argentina_audience_pct"],
            s=tamano(grupo["price_usd"]),
            color=COLOR_TOPICO[topico],
            marker=MARCADOR_TOPICO[topico],
            alpha=0.85,
            edgecolor="white",
            lw=1,
            zorder=2,
        )
    elegidos = datos.loc[list(top)]
    eje.scatter(
        elegidos["followers"],
        elegidos["argentina_audience_pct"],
        s=tamano(elegidos["price_usd"]) * 1.5 + 80,
        facecolor="none",
        edgecolor=TINTA,
        lw=1.3,
        zorder=3,
    )
    # El creador más barato y el del top con menos audiencia argentina.
    extremos = [
        datos["price_usd"].idxmin(),
        elegidos["argentina_audience_pct"].idxmin(),
    ]
    for creador in datos.loc[extremos].itertuples():
        derecha = creador.followers > 2e5
        eje.annotate(
            f"{creador.name}\n{creador.argentina_audience_pct:.0f}% arg. · "
            f"USD {creador.price_usd:,.0f}".replace(",", "."),
            (creador.followers, creador.argentina_audience_pct),
            xytext=(0, -62) if derecha else (14, -4),
            textcoords="offset points",
            ha="center" if derecha else "left",
            fontsize=9.5,
            color=TINTA,
            arrowprops={"arrowstyle": "-", "color": TINTA_SECUNDARIA, "lw": 0.8},
        )
    _eje_followers(eje)
    eje.set_xlabel("Followers (escala log)")
    eje.set_ylabel("% de la audiencia en Argentina")
    eje.set_ylim(0, 100)

    temas = dominante.loc[list(top)].value_counts()
    composicion = ", ".join(
        f"{temas.get(t, 0)} {NOMBRES_TOPICO[t].lower()}" for t in TOPICOS
    )
    r = {
        v: datos["followers"].corr(datos[v])
        for v in ["price_usd", "argentina_audience_pct", "saves_per_1000"]
    }
    eje.set_title(
        f"¿Qué se elige ordenando solo por followers? Top {len(top)}: {composicion}\n"
        f"r(followers, precio) = {_coma(r['price_usd'])} · "
        f"r(followers, % arg.) = {_coma(r['argentina_audience_pct'])} · "
        f"r(followers, guardados) = {_coma(r['saves_per_1000'])}",
    )
    marcas = [
        Line2D(
            [],
            [],
            ls="",
            marker=MARCADOR_TOPICO[t],
            color=COLOR_TOPICO[t],
            markersize=9,
            label=NOMBRES_TOPICO[t],
        )
        for t in TOPICOS
    ]
    marcas.append(
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=TINTA,
            markersize=13,
            label=f"Top {len(top)} por followers",
        )
    )
    leyenda = eje.legend(
        handles=marcas,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9.5,
        title="Tema principal",
        title_fontsize=9.5,
    )
    eje.add_artist(leyenda)
    tamanos = [
        eje.scatter(
            [], [], s=tamano(p), color="#c9c8c2", label=f"USD {p:,}".replace(",", ".")
        )
        for p in (100, 1000, 5000)
    ]
    eje.legend(
        handles=tamanos,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9.5,
        title="Precio por post",
        title_fontsize=9.5,
        labelspacing=1.3,
        borderpad=1,
    )
    _guardar(fig, ruta, extra=(leyenda,))


def _figura_comparacion(
    resumen: pd.DataFrame,
    clientes: Mapping[str, dict],
    presupuesto: float,
    ruta: Path,
) -> None:
    con_presupuesto = (
        f"Miles de argentinos alcanzados en el\ntópico con USD {presupuesto:,.0f} "
        "(↑ mejor)"
    ).replace(",", ".")
    paneles = [
        (
            "usd_por_mil_argentinos",
            1,
            "USD por mil argentinos alcanzados\nen el tópico (↓ mejor)",
        ),
        (
            "intencion_cada_100_usd",
            1,
            "Guardados + visitas de argentinos\ndel tópico cada 100 USD (↑ mejor)",
        ),
        ("argentinos_mismo_presupuesto", 1000, con_presupuesto),
    ]
    claves = list(clientes)
    x = np.arange(len(claves))
    ancho = 0.38
    fig, ejes = plt.subplots(1, len(paneles), figsize=(17, 5.6))
    for eje, (metrica, escala, titulo) in zip(ejes, paneles, strict=True):
        for desplazamiento, estrategia in zip(
            (-ancho / 2, ancho / 2), ESTRATEGIAS, strict=True
        ):
            filas = [resumen.loc[(c, estrategia), metrica] for c in claves]
            medias = np.array([f["media"] for f in filas]) / escala
            desvios = np.array([f["desvio"] for f in filas]) / escala
            barras = eje.bar(
                x + desplazamiento,
                medias,
                ancho,
                yerr=desvios if estrategia == "propuesta" else None,
                capsize=3,
                color=COLOR_ESTRATEGIA[estrategia],
                edgecolor="white",
                lw=2,
                label=ESTRATEGIAS[estrategia],
            )
            eje.bar_label(barras, fmt="{:.0f}", padding=3, fontsize=10, color=TINTA)
            if metrica != "argentinos_mismo_presupuesto" or estrategia != "propuesta":
                continue
            for posicion, clave in zip(x, claves, strict=True):
                gasto = resumen.loc[
                    (clave, estrategia), ("gasto_mismo_presupuesto", "media")
                ]
                if gasto < 0.9 * presupuesto:
                    eje.annotate(
                        f"gasta USD {gasto / 1000:.1f} mil:\nno hay más candidatos".replace(
                            ".", ","
                        ),
                        (posicion + desplazamiento, 0),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        color="white",
                        rotation=90,
                    )
        eje.set_xticks(x, [clientes[c]["nombre"] for c in claves], fontsize=11.5)
        eje.set_title(titulo, fontsize=11.5)
        eje.margins(y=0.15)
        eje.grid(axis="x", visible=False)
    ejes[0].legend(loc="upper left", fontsize=10)
    fig.suptitle(
        "Nuestra propuesta vs. elegir por followers (propuesta: media ± desvío sobre "
        "semillas del mapa)",
        fontsize=13,
    )
    fig.tight_layout()
    _guardar(fig, ruta)


def _figura_actualizacion(tabla: pd.DataFrame, ruta: Path) -> None:
    fig, eje = plt.subplots(figsize=(8.5, 5))
    curvas = [
        ("cambia_neurona", "Cambia de neurona", COLOR_ESTRATEGIA["followers"]),
        (
            "cambia_region",
            "Cambia de región (de cliente)",
            COLOR_ESTRATEGIA["propuesta"],
        ),
    ]
    x = tabla["sigma"] * 100
    for columna, etiqueta, color in curvas:
        media = tabla[f"{columna}_media"] * 100
        desvio = tabla[f"{columna}_desvio"] * 100
        eje.plot(x, media, color=color, lw=2.2, marker="o", ms=8, label=etiqueta)
        inferior = (media - desvio).clip(lower=0)
        eje.fill_between(x, inferior, media + desvio, color=color, alpha=0.15)
        eje.annotate(
            f"{media.iloc[-1]:.0f}%",
            (x.iloc[-1], media.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
            color=TINTA,
        )
    techo = (tabla["cambia_neurona_media"] + tabla["cambia_neurona_desvio"]).max()
    eje.set_ylim(0, max(5.0, 130 * techo))
    eje.set_xticks(x, [f"{v:g}%" for v in x])
    eje.set_xlabel("Variación aleatoria de cada variable en un mes (% de su valor)")
    eje.set_ylabel("% de creadores (media ± desvío\nsobre semillas del mapa)")
    eje.legend(loc="upper left")
    eje.set_title("Actualización mensual sin reentrenar el mapa")
    _guardar(fig, ruta)


# --- Análisis ----------------------------------------------------------------


def _resumir(filas: list[dict], claves: list[str]) -> pd.DataFrame:
    tabla = pd.DataFrame(filas)
    metricas = [c for c in tabla.columns if c not in claves + ["semilla"]]
    agrupado = tabla.groupby(claves, sort=False)[metricas]
    resumen = pd.concat(
        {"media": agrupado.mean(), "desvio": agrupado.std(ddof=1).fillna(0.0)},
        axis=1,
    ).swaplevel(axis=1)
    return resumen.sort_index(axis=1)


def _texto_resumen(
    por_id: pd.DataFrame,
    tablas: Mapping[str, pd.DataFrame],
    clientes: Mapping[str, dict],
    resumen: pd.DataFrame,
    selecciones: pd.DataFrame,
    actualizacion: pd.DataFrame,
    contexto: Mapping[str, object],
) -> str:
    top = contexto["top_followers"]
    lineas = [
        "Decisión para las campañas — TP4, ejercicio 1.4",
        "",
        "Supuestos: solo compra el público argentino (alcance útil = alcance",
        "argentino × % de posts del tópico); guardados + visitas = intención.",
        f"Mapa: {contexto['mapa']}",
        f"Top {len(top)} por followers: {', '.join(por_id.loc[top, 'name'])} "
        f"(USD {contexto['presupuesto']:,.0f} = presupuesto de referencia)",
        "",
        "Correlación de followers con:",
    ]
    for variable in [
        "price_usd",
        "argentina_audience_pct",
        "non_follower_reach_pct",
        "content_tutorial_pct",
        "saves_per_1000",
        "comments_per_1000",
        "profile_visits_per_1000",
    ]:
        r = por_id["followers"].corr(por_id[variable])
        lineas.append(f"  {variable:<26} {r:+.2f}")

    lineas += ["", "Por cliente (creadores cuyo tema principal es el del cliente):"]
    for clave, cliente in clientes.items():
        tabla = tablas[clave]
        mediana = tabla["followers"].median()
        chicos = tabla[tabla["followers"] <= mediana]
        grandes = tabla[tabla["followers"] > mediana]
        lineas.append(
            f"  {cliente['nombre']}: USD por mil argentinos (mediana) chicos "
            f"{chicos['usd_por_mil_argentinos'].median():.0f} | grandes "
            f"{grandes['usd_por_mil_argentinos'].median():.0f}; intención chicos "
            f"{chicos['intencion'].median():.0f} | grandes "
            f"{grandes['intencion'].median():.0f}"
        )
        baratos = tabla.nsmallest(3, "usd_por_mil_argentinos")
        lineas.append(
            "    más baratos por argentino: "
            + ", ".join(
                f"{c.name} ({c.usd_por_mil_argentinos:.0f}, "
                f"{c.argentina_audience_pct:.0f}% arg.)"
                for c in baratos.itertuples()
            )
        )

    representativa = contexto["representativa"]
    propuesta = selecciones[
        (selecciones["estrategia"] == "propuesta")
        & (selecciones["semilla"] == representativa)
    ]
    lineas += ["", "Propuesta (semilla representativa) y coincidencia con followers:"]
    for clave, cliente in clientes.items():
        ids = propuesta.loc[propuesta["cliente"] == clave, "creator_id"]
        lineas.append(
            f"  {cliente['nombre']:<9} {', '.join(por_id.loc[ids, 'name'])} "
            f"(coincide {jaccard(ids, top):.0%})"
        )
    columnas = [
        "usd_por_mil_argentinos",
        "intencion_cada_100_usd",
        "argentinos_topico",
        "costo_total",
        "argentinos_mismo_presupuesto",
        "gasto_mismo_presupuesto",
        "afinidad_media",
    ]
    lineas += ["", "Comparación (media sobre semillas):"]
    lineas.append(
        resumen.xs("media", axis=1, level=1)[columnas].to_string(
            float_format=lambda v: f"{v:,.1f}"
        )
    )
    lineas += ["", "Actualización mensual simulada (fracción de creadores):"]
    lineas.append(actualizacion.to_string(index=False, float_format="%.3f"))
    return "\n".join(lineas) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    with resolver_ruta(args.config).open("rb") as archivo:
        config = tomllib.load(archivo)
    entradas = config["entradas"]
    n = int(config["seleccion"]["creadores_por_campana"])
    clientes = config["clientes"]
    actualizacion = config["actualizacion"]
    repeticiones = args.repeticiones or int(actualizacion["repeticiones"])
    salida = resolver_ruta(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    crudos = pd.read_csv(resolver_ruta(entradas["datos"]))
    por_id = crudos.set_index("creator_id")
    dominante = por_id[TOPICOS].idxmax(axis=1)
    corrida = resolver_ruta(entradas["corrida"])
    with (corrida / "config_usada.toml").open("rb") as archivo:
        config_mapa = tomllib.load(archivo)
    k = int(config_mapa["modelo"]["grilla"])
    distancia = config_mapa["modelo"]["distancia"]
    representativa = int(config_mapa["reproducibilidad"]["semilla_representativa"])
    pesos = pd.read_csv(corrida / "pesos_semillas.csv")
    asignaciones = pd.read_csv(corrida / "asignaciones_semillas.csv")
    semillas = sorted(int(s) for s in asignaciones["semilla"].unique())
    pcs = [c for c in pesos.columns if c.startswith("PC")]
    cargas = pd.read_csv(resolver_ruta(entradas["cargas"]), index_col=0)

    # Estrategia actual: los N con más seguidores. Su costo fija el presupuesto.
    por_followers = list(por_id["followers"].sort_values(ascending=False).index)
    top_followers = por_followers[:n]
    presupuesto = float(por_id.loc[top_followers, "price_usd"].sum())
    datos_cliente = {
        clave: agregar_metricas(por_id, cliente["topico"])
        for clave, cliente in clientes.items()
    }

    filas_comparacion, filas_selecciones = [], []
    regiones_por_semilla = {}
    for semilla in semillas:
        asignadas = asignaciones[asignaciones["semilla"] == semilla]
        regiones = topico_dominante(crudos, asignadas, TOPICOS)
        regiones_por_semilla[semilla] = regiones
        region_creador = asignadas.set_index("creator_id")["neurona"].map(regiones)
        for clave, cliente in clientes.items():
            topico = cliente["topico"]
            datos = datos_cliente[clave]
            # Región del mapa ∪ creadores cuyo propio tema principal es el tópico.
            candidatos = sorted(
                set(region_creador[region_creador == topico].index)
                | set(dominante[dominante == topico].index)
            )
            orden = seleccionar(
                puntaje(datos, cliente["metricas"]), candidatos, len(candidatos)
            )
            elecciones = {
                "followers": (top_followers, top_followers),
                "propuesta": (
                    orden[:n],
                    seleccionar_con_presupuesto(orden, datos["price_usd"], presupuesto),
                ),
            }
            for estrategia, (ids, ids_presupuesto) in elecciones.items():
                gastando_igual = metricas_seleccion(datos, ids_presupuesto, topico)
                filas_comparacion.append(
                    {
                        "cliente": clave,
                        "estrategia": estrategia,
                        "semilla": semilla,
                        **metricas_seleccion(datos, ids, topico),
                        "argentinos_mismo_presupuesto": gastando_igual[
                            "argentinos_topico"
                        ],
                        "gasto_mismo_presupuesto": gastando_igual["costo_total"],
                    }
                )
                for posicion, creador in enumerate(ids, start=1):
                    filas_selecciones.append(
                        {
                            "cliente": clave,
                            "estrategia": estrategia,
                            "semilla": semilla,
                            "orden": posicion,
                            "creator_id": creador,
                            "name": por_id.loc[creador, "name"],
                            "usd_por_mil_argentinos": datos.loc[
                                creador, "usd_por_mil_argentinos"
                            ],
                        }
                    )

    resumen = _resumir(filas_comparacion, ["cliente", "estrategia"])
    resumen.to_csv(salida / "comparacion.csv", float_format="%.4f")
    selecciones = pd.DataFrame(filas_selecciones)
    selecciones.to_csv(salida / "selecciones.csv", index=False, float_format="%.4f")

    # Creadores de cada cliente según su propio tema principal (no depende del mapa).
    columnas = [
        "name",
        "followers",
        "argentina_audience_pct",
        "price_usd",
        "argentinos_topico",
        "usd_por_mil_argentinos",
        "non_follower_reach_pct",
        "intencion",
        "content_tutorial_pct",
    ]
    tablas = {
        clave: datos_cliente[clave].loc[dominante == cliente["topico"], columnas]
        for clave, cliente in clientes.items()
    }
    pd.concat(tablas, names=["cliente", "creator_id"]).to_csv(
        salida / "creadores_por_cliente.csv", float_format="%.2f"
    )

    # Actualización mensual: perturbar → proyectar → neurona ganadora.
    proyector = Proyector(crudos, cargas, pcs)
    filas_actualizacion = []
    generador = np.random.default_rng(int(actualizacion["semilla"]))
    for semilla in semillas:
        pesos_semilla = pesos[pesos["semilla"] == semilla].sort_values("neurona")
        matriz_pesos = pesos_semilla[pcs].to_numpy(dtype=float)
        base = ubicar(proyector(crudos), matriz_pesos, distancia)
        guardadas = (
            asignaciones[asignaciones["semilla"] == semilla]
            .set_index("creator_id")
            .loc[crudos["creator_id"], "neurona"]
            .to_numpy()
        )
        if not np.array_equal(base, guardadas):
            raise ValueError(f"Semilla {semilla}: no se reproduce el mapa guardado.")
        etiquetas = (
            regiones_por_semilla[semilla].reindex(pesos_semilla["neurona"]).to_numpy()
        )
        for sigma in actualizacion["sigmas"]:
            for _ in range(repeticiones):
                variables = perturbar(crudos[proyector.variables], sigma, generador)
                nuevas = ubicar(proyector(variables), matriz_pesos, distancia)
                filas_actualizacion.append(
                    {
                        "semilla": semilla,
                        "sigma": sigma,
                        "cambia_neurona": float(np.mean(nuevas != base)),
                        "cambia_region": float(
                            np.mean(etiquetas[nuevas] != etiquetas[base])
                        ),
                    }
                )
    # Primero se promedian las repeticiones por semilla; el desvío es entre semillas.
    por_semilla = pd.DataFrame(filas_actualizacion).groupby(["sigma", "semilla"]).mean()
    agrupado = por_semilla.groupby("sigma")
    tabla_actualizacion = pd.concat(
        [
            agrupado.mean().add_suffix("_media"),
            agrupado.std(ddof=1).add_suffix("_desvio"),
        ],
        axis=1,
    ).reset_index()
    tabla_actualizacion.to_csv(
        salida / "actualizacion.csv", index=False, float_format="%.4f"
    )

    asignadas_rep = asignaciones[asignaciones["semilla"] == representativa]
    _figura_mapa_regiones(crudos, asignadas_rep, k, salida / "fig_mapa_regiones.png")
    _figura_enfoques(tablas, clientes, salida / "fig_enfoques.png")
    # Versiones por cliente para la presentación: el primero solo y el resto.
    primero, *resto = clientes
    _figura_enfoques(
        tablas, {primero: clientes[primero]}, salida / f"fig_enfoques_{primero}.png"
    )
    _figura_enfoques(
        tablas, {c: clientes[c] for c in resto}, salida / "fig_enfoques_resto.png"
    )
    _figura_followers(
        por_id, por_followers[:TOP_FOLLOWERS_FIGURA], salida / "fig_followers.png"
    )
    _figura_comparacion(resumen, clientes, presupuesto, salida / "fig_comparacion.png")
    _figura_actualizacion(tabla_actualizacion, salida / "fig_actualizacion.png")

    texto = _texto_resumen(
        por_id,
        tablas,
        clientes,
        resumen,
        selecciones,
        tabla_actualizacion,
        {
            "top_followers": top_followers,
            "presupuesto": presupuesto,
            "representativa": representativa,
            "mapa": f"{entradas['corrida']} (k={k}, {distancia}, semillas "
            f"{semillas}, representativa {representativa})",
        },
    )
    (salida / "resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Resultados: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
