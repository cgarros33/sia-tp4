#!/usr/bin/env python3
"""Figuras del ejercicio 1 rediseñadas para la presentación (TP4).

Script de análisis: solo lee resultados de 1.1 (correlaciones, cargas, scores
y autovalores del PCA), el dataset y el barrido de Kohonen; no entrena nada.

- ``fig_correlaciones_ordenadas``: matriz de correlación de las 27 variables
  del PCA, ordenadas por su carga en PC1 para que se vean los bloques.
- ``fig_biplot``: biplot PC1-PC2 con texto grande y etiquetas ubicadas para
  no taparse entre sí.
- ``fig_cargas``: las 12 cargas de mayor valor absoluto de PC1, PC2 y PC3, con
  texto grande.
- ``fig_barrido``: QE media ± desvío de cada valor probado en los pasos 4, 5,
  6 y 8 del barrido secuencial, con un eje compartido para comparar pasos.
- ``fig_enfoques``: los tres enfoques de campaña frente al tamaño y a la
  audiencia argentina, con una tendencia por panel (todos los creadores).
- ``fig_enfoques_futbol`` y ``fig_enfoques_resto``: el criterio de fútbol y el
  de refugios, belleza y arte. Para los mejores candidatos de cada cliente,
  cuánto aporta cada familia de métricas al puntaje con que ``campanas.py``
  los ordena (pesos de ``campanas.toml``).
- ``fig_pesos`` y ``fig_pesos_<cliente>``: los pesos del criterio de cada
  campaña (``campanas.toml``) como una barra al 100%, por métrica y familia.
- ``fig_mapa_biplot``: la grilla de Kohonen (pca10, k=4, euclidiana; semilla
  representativa) sobre los creadores en PC1-PC2, con cada neurona del color
  de su tópico dominante.
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
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CORRELACIONES = RAIZ / "resultados" / "ej1" / "eda" / "correlaciones.csv"
RUTA_CARGAS = RAIZ / "resultados" / "ej1" / "pca_0.90" / "cargas.csv"
RUTA_SCORES = RAIZ / "resultados" / "ej1" / "pca_0.90" / "scores.csv"
RUTA_AUTOVALORES = RAIZ / "resultados" / "ej1" / "pca_0.90" / "autovalores.csv"
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"
RUTA_BARRIDO = RAIZ / "resultados" / "ej1" / "ej1.2-pca10" / "kohonen_sweep" / "sweep4"
RUTA_CAMPANAS = RAIZ / "resultados" / "ej1" / "campanas"
RUTA_MAPA = RAIZ / "resultados" / "ej1" / "ej1.2-pca10" / "kohonen_k4_euclidiana"
RUTA_CONFIG_CAMPANAS = RAIZ / "src" / "ej1" / "campanas.toml"
RUTA_SALIDA = RAIZ / "resultados" / "ej1" / "presentacion"
PASOS_FIGURA = (4, 5, 6, 8)

TITULOS_PASO = {
    "topologia": "Topología",
    "distancia": "Distancia",
    "inicializacion": "Inicialización",
    "radio_inicial_relativo": "Radio inicial",
    "decaimiento_radio": "Decaimiento del radio",
    "eta_inicial": "η inicial",
    "decaimiento_eta": "Decaimiento de η",
    "iteraciones": "Iteraciones",
}
# Los colores de tópico de toda la presentación (ver presentacion/guion.md).
TOPICOS = {
    "topic_football_pct": ("Fútbol", "#1baf7a"),
    "topic_dogs_pct": ("Perros", "#eda100"),
    "topic_makeup_pct": ("Maquillaje", "#d95926"),
    "topic_art_pct": ("Arte", "#2a78d6"),
}
DPI = 150
# Distancias (px) a las que se prueba cada etiqueta alrededor de su punto.
RADIOS_PX = (6, 14, 24, 36, 50, 66, 84, 104, 126)
# Más lejos que esto (px) de su punto, la etiqueta lleva línea guía.
GUIA_PX = 14
# Distancia mínima (px) entre una etiqueta y los puntos que no son el suyo.
HOLGURA_PX = 30
# Pasadas de reubicación de etiquetas.
PASADAS = 3
COLOR_ELEGIDO = "#6250d6"
COLOR_RESTO = "#8f8e88"
# Los mismos que la figura de cargas de 1.1.
COLOR_POSITIVA = "#4C72B0"
COLOR_NEGATIVA = "#C44E52"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRILLA = "#e4e3df"
# Título y enfoque de cada cliente en las figuras de criterios.
CLIENTE_FUTBOL = {"futbol": ("Fútbol", "Masividad + afinidad, a buen precio")}
CLIENTES_RESTO = {
    "refugios": ("Refugios", "Descubrimiento + profundidad"),
    "belleza": ("Belleza", "Profundidad + afinidad"),
    "arte": ("Arte", "Masividad + profundidad"),
}
# Familia de cada métrica de los criterios de campanas.toml (ver el comentario
# de ese archivo): los tres enfoques de la consigna (masividad, descubrimiento y
# profundidad), la afinidad con el público del cliente y el costo.
FAMILIAS = {
    "topic_football_pct": "afinidad",
    "topic_dogs_pct": "afinidad",
    "topic_makeup_pct": "afinidad",
    "topic_art_pct": "afinidad",
    "argentina_audience_pct": "afinidad",
    "average_reach": "masividad",
    "non_follower_reach_pct": "descubrimiento",
    "content_tutorial_pct": "profundidad",
    "content_review_pct": "profundidad",
    "saves_per_1000": "profundidad",
    "comments_per_1000": "profundidad",
    "profile_visits_per_1000": "profundidad",
    "price_usd": "costo",
}
# Afinidad es la base de todos los criterios: va en gris neutro.
ESTILO_FAMILIA = {
    "afinidad": ("Afinidad (tópico del cliente, audiencia argentina)", "#9c9a92"),
    "masividad": ("Masividad (alcance)", "#008300"),
    "descubrimiento": ("Descubrimiento (no seguidores)", "#d55181"),
    "profundidad": ("Profundidad (tutoriales, guardados, consultas)", "#8c510a"),
    "costo": ("Costo (precio por post)", "#465a82"),
}
# Candidatos que se muestran por cliente en la figura de criterios.
FILAS_CRITERIO = 8
# Nombre corto de cada métrica para la figura de pesos.
NOMBRES_METRICA = {
    "topic_football_pct": "Posts de fútbol",
    "topic_dogs_pct": "Posts de perros",
    "topic_makeup_pct": "Posts de maquillaje",
    "topic_art_pct": "Posts de arte",
    "argentina_audience_pct": "Audiencia argentina",
    "average_reach": "Alcance por post",
    "non_follower_reach_pct": "Alcance a no seguidores",
    "comments_per_1000": "Comentarios",
    "profile_visits_per_1000": "Visitas",
    "content_tutorial_pct": "Tutoriales",
    "content_review_pct": "Reseñas",
    "saves_per_1000": "Guardados",
    "price_usd": "Precio (menos es mejor)",
}

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
        description="Figuras del ejercicio 1 para la presentación.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--correlaciones", type=Path, default=RUTA_CORRELACIONES)
    parser.add_argument("--cargas", type=Path, default=RUTA_CARGAS)
    parser.add_argument("--scores", type=Path, default=RUTA_SCORES)
    parser.add_argument("--autovalores", type=Path, default=RUTA_AUTOVALORES)
    parser.add_argument("--datos", type=Path, default=RUTA_DATOS)
    parser.add_argument("--barrido", type=Path, default=RUTA_BARRIDO)
    parser.add_argument("--campanas", type=Path, default=RUTA_CAMPANAS)
    parser.add_argument("--mapa", type=Path, default=RUTA_MAPA)
    parser.add_argument("--config-campanas", type=Path, default=RUTA_CONFIG_CAMPANAS)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    return parser.parse_args(argv)


def _coma(valor: float, formato: str = ".2f") -> str:
    return format(valor, formato).replace(".", ",")


def etiqueta(variable: str) -> str:
    """Nombre corto de una variable para los ejes."""
    return (
        variable.replace("_per_1000", " /1000")
        .replace("_pct", " %")
        .replace("average_", "")
        .replace("_", " ")
    )


def orden_por_carga(cargas: pd.DataFrame, componente: str = "PC1") -> list[str]:
    """Variables de mayor a menor carga en la componente."""
    return list(cargas[componente].sort_values(ascending=False).index)


def cargas_principales(
    cargas: pd.DataFrame, componente: str, cantidad: int = 12
) -> pd.Series:
    """Las variables de mayor carga absoluta, ordenadas de menor a mayor carga."""
    serie = cargas[componente]
    elegidas = serie.abs().sort_values(ascending=False).head(cantidad).index
    return serie.loc[elegidas].sort_values()


def formato_valor(parametro: str, valor: object) -> str:
    """Valor de un hiperparámetro tal como se muestra en la figura."""
    if isinstance(valor, str):
        return valor
    if parametro == "radio_inicial_relativo":
        return f"{_coma(float(valor), 'g')}·k"
    if parametro == "iteraciones":
        return f"{int(valor):,}".replace(",", ".")
    return _coma(float(valor), "g")


def mismo_valor(a: object, b: object) -> bool:
    """Compara valores del barrido que pueden estar guardados como texto."""
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _figura_correlaciones(
    correlaciones: pd.DataFrame, cargas: pd.DataFrame, ruta: Path
) -> None:
    orden = orden_por_carga(cargas)
    # Triángulo inferior sin la diagonal: se omiten la primera fila y la última
    # columna, que quedarían vacías.
    filas, columnas = orden[1:], orden[:-1]
    matriz = correlaciones.loc[filas, columnas].to_numpy()
    visible = np.ma.masked_where(np.triu(np.ones_like(matriz, dtype=bool), k=1), matriz)

    fig, eje = plt.subplots(figsize=(11, 10))
    imagen = eje.imshow(visible, cmap="RdBu_r", vmin=-1, vmax=1)
    for i, j in zip(*np.tril_indices(len(filas)), strict=True):
        if abs(matriz[i, j]) >= 0.9:
            eje.text(
                j,
                i,
                _coma(matriz[i, j]).replace("0,", ","),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
            )
    eje.set_xticks(
        range(len(columnas)), [etiqueta(v) for v in columnas], rotation=90, fontsize=9
    )
    eje.set_yticks(range(len(filas)), [etiqueta(v) for v in filas], fontsize=9)
    for borde in eje.spines.values():
        borde.set_visible(False)
    barra = fig.colorbar(imagen, ax=eje, fraction=0.035, pad=0.02)
    barra.set_label("Correlación de Pearson")
    eje.set_title(
        "Correlación entre las 27 variables (ya sin las 4 redundantes exactas)\n"
        "Ordenadas por su carga en PC1: arriba lo masivo, abajo lo profundo. "
        "Números: |r| ≥ 0,9",
        fontsize=12,
    )
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figura_cargas(cargas: pd.DataFrame, autovalores: pd.DataFrame, ruta: Path) -> None:
    ratios = autovalores.set_index("componente")["ratio_varianza"]
    fig, ejes = plt.subplots(1, 3, figsize=(20, 8.5))
    for eje, componente in zip(ejes, ["PC1", "PC2", "PC3"], strict=True):
        serie = cargas_principales(cargas, componente)
        colores = np.where(serie < 0, COLOR_NEGATIVA, COLOR_POSITIVA)
        eje.barh(serie.index, serie, color=colores, height=0.72)
        eje.axvline(0, color=TINTA_SECUNDARIA, lw=1)
        eje.set_title(
            f"{componente} ({_coma(100 * ratios[componente], '.1f')}% de varianza)",
            fontsize=17,
        )
        eje.tick_params(axis="y", labelsize=15, labelcolor=TINTA, length=0)
        eje.tick_params(axis="x", labelsize=13)
        eje.set_xlabel("Carga", fontsize=14)
        eje.spines["left"].set_visible(False)
        eje.grid(axis="x", color=GRILLA, lw=0.8)
        eje.set_axisbelow(True)
        eje.margins(y=0.02)
    fig.suptitle("Cargas de las tres primeras componentes", fontsize=19)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def superposicion(caja: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    """Área de intersección entre una caja [x0, y0, x1, y1] y cada una de otras."""
    ancho = np.minimum(caja[2], cajas[:, 2]) - np.maximum(caja[0], cajas[:, 0])
    alto = np.minimum(caja[3], cajas[:, 3]) - np.maximum(caja[1], cajas[:, 1])
    return np.clip(ancho, 0, None) * np.clip(alto, 0, None)


def distancia_a_caja(caja: np.ndarray, puntos: np.ndarray) -> np.ndarray:
    """Distancia de cada punto al borde más cercano de la caja (0 si está dentro)."""
    dx = np.maximum(np.maximum(caja[0] - puntos[:, 0], puntos[:, 0] - caja[2]), 0)
    dy = np.maximum(np.maximum(caja[1] - puntos[:, 1], puntos[:, 1] - caja[3]), 0)
    return np.hypot(dx, dy)


def _dentro(puntos: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    """Matriz (puntos x cajas) que marca qué punto cae dentro de qué caja."""
    return (
        (puntos[:, None, 0] >= cajas[None, :, 0])
        & (puntos[:, None, 0] <= cajas[None, :, 2])
        & (puntos[:, None, 1] >= cajas[None, :, 1])
        & (puntos[:, None, 1] <= cajas[None, :, 3])
    )


def _fraccion_antes(componente: float) -> float:
    """Parte de la etiqueta que queda antes del punto de anclaje en un eje."""
    if componente > 0.38:
        return 0.0
    if componente < -0.38:
        return 1.0
    return 0.5


def pegados_al_texto(caja: np.ndarray, puntos: np.ndarray) -> np.ndarray:
    """Puntos a la altura del texto y junto a su principio o a su final.

    Quien lee toma a ese punto como el dueño de la etiqueta.
    """
    a_la_altura = (puntos[:, 1] >= caja[1] - 8) & (puntos[:, 1] <= caja[3] + 8)
    antes = (puntos[:, 0] < caja[0]) & (puntos[:, 0] > caja[0] - 44)
    despues = (puntos[:, 0] > caja[2]) & (puntos[:, 0] < caja[2] + 44)
    return a_la_altura & (antes | despues)


def _mejor_caja(
    ancla: np.ndarray,
    tamano: np.ndarray,
    otras_anclas: np.ndarray,
    colocadas: np.ndarray,
    lineas: np.ndarray,
    obstaculos: np.ndarray,
    pesos: np.ndarray,
    limites: np.ndarray,
    direccion: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Caja de menor costo para una etiqueta y los puntos de su línea guía."""
    ancho, alto = tamano
    mejor_costo, mejor_caja, mejor_tramo = np.inf, None, None
    for radio in RADIOS_PX:
        for angulo in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            coseno, seno = np.cos(angulo), np.sin(angulo)
            x0 = ancla[0] + radio * coseno - ancho * _fraccion_antes(coseno)
            y0 = ancla[1] + radio * seno - alto * _fraccion_antes(seno)
            caja = np.array([x0, y0, x0 + ancho, y0 + alto])
            afuera = ancho * alto - superposicion(caja, limites[None])[0]
            propia = distancia_a_caja(caja, ancla[None])[0]
            tramo = np.empty((0, 2))
            if propia > GUIA_PX:
                # Puntos de la línea guía, desde fuera del marcador propio.
                borde = np.clip(ancla, caja[:2], caja[2:])
                pasos = np.linspace(min(10 / propia, 1), 1, 10)[:, None]
                tramo = ancla + pasos * (borde - ancla)
            # Un punto al costado de la etiqueta se lee como su dueño más que uno
            # arriba o abajo: la distancia horizontal cuenta la mitad.
            # Además, la propia tiene que ser claramente la más cercana.
            cercania = np.maximum(
                HOLGURA_PX
                - distancia_a_caja(caja * [0.5, 1, 0.5, 1], otras_anclas * [0.5, 1]),
                propia + 8 - distancia_a_caja(caja, otras_anclas),
            )
            costo = (
                10 * superposicion(caja, colocadas).sum()
                + (pesos * superposicion(caja, obstaculos)).sum()
                + 10 * afuera
                + 30 * np.clip(cercania, 0, None).sum()
                + 3000 * pegados_al_texto(caja, otras_anclas).sum()
                + 300 * _dentro(lineas, caja[None]).sum()
                + 300 * _dentro(tramo, colocadas).sum()
                + 30 * (_dentro(tramo, obstaculos) * pesos).sum()
                + radio
            )
            if direccion is not None:
                costo += 60 * (1 - coseno * direccion[0] - seno * direccion[1])
            if costo < mejor_costo:
                mejor_costo, mejor_caja, mejor_tramo = costo, caja, tramo
    return mejor_caja, mejor_tramo


def ubicar_etiquetas(
    anclas: np.ndarray,
    tamanos: np.ndarray,
    obstaculos: np.ndarray,
    pesos: np.ndarray,
    limites: np.ndarray,
    direcciones: np.ndarray | None = None,
) -> np.ndarray:
    """Caja [x0, y0, x1, y1] de cada etiqueta.

    Prueba 16 direcciones a distancias crecientes del ancla y elige la posición
    de menor costo: lo que tapa de otras etiquetas, de obstáculos (con su peso)
    y de afuera del área; quedar a menos de ``HOLGURA_PX`` de otra ancla, o
    con otra ancla pegada al principio o al final del texto, porque se leería
    como etiqueta ajena; que su línea guía cruce etiquetas u obstáculos, o que
    tape otras líneas guía; la distancia y, si se da, el desvío de la dirección
    preferida.

    La primera pasada ubica en el orden dado contra las ya ubicadas; las
    siguientes reubican cada una contra todas las demás, para que el
    resultado dependa menos del orden.
    """
    cantidad = len(anclas)
    indices = np.arange(cantidad)
    cajas = np.zeros((cantidad, 4))
    tramos = [np.empty((0, 2))] * cantidad
    for pasada in range(PASADAS):
        for indice in indices:
            otras = indices != indice
            if pasada == 0:
                otras &= indices < indice
            cajas[indice], tramos[indice] = _mejor_caja(
                anclas[indice],
                tamanos[indice],
                np.delete(anclas, indice, axis=0),
                cajas[otras],
                np.vstack([np.empty((0, 2))] + [tramos[j] for j in indices[otras]]),
                obstaculos,
                pesos,
                limites,
                None if direcciones is None else direcciones[indice],
            )
    return cajas


def _cajas_alrededor(centros: np.ndarray, medio: float) -> np.ndarray:
    return np.hstack([centros - medio, centros + medio])


def _tamanos_texto(eje, textos, renderer, margen: float, **estilo) -> np.ndarray:
    tamanos = []
    for texto in textos:
        temporal = eje.text(0, 0, texto, **estilo)
        caja = temporal.get_window_extent(renderer)
        temporal.remove()
        tamanos.append((caja.width + 2 * margen, caja.height + 2 * margen))
    return np.array(tamanos)


def _escribir_etiquetas(
    eje, textos, puntos, anclas, cajas, margen: float, **estilo
) -> None:
    """Escribe cada etiqueta en su caja, con una línea guía si quedó lejos.

    Las líneas van debajo de puntos y textos, así no tachan otras etiquetas.
    """
    for texto, punto, ancla, caja in zip(textos, puntos, anclas, cajas, strict=True):
        if distancia_a_caja(caja, ancla[None])[0] > GUIA_PX:
            borde = np.clip(ancla, caja[:2], caja[2:])
            eje.annotate(
                "",
                punto,
                xytext=(borde - ancla) * 72 / DPI,
                textcoords="offset points",
                arrowprops={
                    "arrowstyle": "-",
                    "color": COLOR_RESTO,
                    "lw": 0.8,
                    "shrinkA": 0,
                    "shrinkB": 5,
                },
                zorder=2,
            )
        eje.annotate(
            texto,
            punto,
            xytext=(caja[:2] + margen - ancla) * 72 / DPI,
            textcoords="offset points",
            ha="left",
            va="bottom",
            zorder=5,
            **estilo,
        )


def _figura_biplot(
    scores: pd.DataFrame,
    cargas: pd.DataFrame,
    autovalores: pd.DataFrame,
    datos: pd.DataFrame,
    ruta: Path,
) -> None:
    puntos = scores[["PC1", "PC2"]].to_numpy()
    topicos = datos.set_index("creator_id").loc[scores["creator_id"], list(TOPICOS)]
    dominante = topicos.idxmax(axis=1).to_numpy()
    # Las 8 variables con mayor carga en el plano, escaladas como en 1.1.
    cargas_2d = cargas[["PC1", "PC2"]].to_numpy()
    normas = np.linalg.norm(cargas_2d, axis=1)
    maximas = np.argsort(normas)[-8:]
    puntas = cargas_2d[maximas] * 0.8 * np.abs(puntos).max() / normas[maximas].max()
    variables = list(cargas.index[maximas])

    fig, eje = plt.subplots(figsize=(13, 10), dpi=DPI)
    for punta in puntas:
        eje.annotate(
            "",
            punta,
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLOR_RESTO,
                "lw": 1.4,
                "mutation_scale": 16,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=1,
        )
    for topico, (nombre, color) in TOPICOS.items():
        eje.scatter(
            *puntos[dominante == topico].T,
            s=70,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=nombre,
            zorder=3,
        )
    extremos = np.vstack([puntos, puntas, np.zeros(2)])
    minimo, maximo = extremos.min(axis=0), extremos.max(axis=0)
    margen_datos = 0.1 * (maximo - minimo)
    eje.set_xlim(minimo[0] - margen_datos[0], maximo[0] + margen_datos[0])
    eje.set_ylim(minimo[1] - margen_datos[1], maximo[1] + margen_datos[1])
    eje.axhline(0, color=TINTA_SECUNDARIA, lw=0.6, ls="--", zorder=0)
    eje.axvline(0, color=TINTA_SECUNDARIA, lw=0.6, ls="--", zorder=0)
    ratios = autovalores.set_index("componente")["ratio_varianza"]
    eje.set_xlabel(f"PC1 ({_coma(100 * ratios['PC1'], '.1f')}% de varianza)")
    eje.set_ylabel(f"PC2 ({_coma(100 * ratios['PC2'], '.1f')}% de varianza)")
    eje.set_title("Biplot PC1-PC2: creadores y cargas de variables", fontsize=14)
    leyenda = eje.legend(
        title="Tópico dominante", loc="upper right", fontsize=11, title_fontsize=12
    )
    nota = eje.text(
        0.01,
        0.01,
        "Flechas escaladas para visualización",
        transform=eje.transAxes,
        fontsize=10,
        color=TINTA_SECUNDARIA,
    )
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Obstáculos en píxeles: puntos, tramos de las flechas, leyenda y nota.
    puntos_px = eje.transData.transform(puntos)
    origen_px = eje.transData.transform([(0, 0)])[0]
    puntas_px = eje.transData.transform(puntas)
    tramos = np.vstack(
        [
            origen_px + np.outer(np.linspace(0, 1, 40), punta - origen_px)
            for punta in puntas_px
        ]
    )
    fijos = np.array(
        [
            leyenda.get_window_extent(renderer).extents,
            nota.get_window_extent(renderer).extents,
        ]
    )
    obstaculos = np.vstack(
        [_cajas_alrededor(puntos_px, 9), _cajas_alrededor(tramos, 3), fijos]
    )
    pesos = np.concatenate(
        [np.full(len(puntos_px), 3.0), np.full(len(tramos), 1.0), np.full(2, 10.0)]
    )
    limites = eje.get_window_extent(renderer).extents

    estilo_variables = {
        "fontsize": 13,
        "color": TINTA_SECUNDARIA,
        "bbox": {"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
    }
    margen_variables = 2 * DPI / 72
    tamanos = _tamanos_texto(eje, variables, renderer, margen_variables, fontsize=13)
    direcciones = puntas_px - origen_px
    direcciones /= np.linalg.norm(direcciones, axis=1, keepdims=True)
    cajas_variables = ubicar_etiquetas(
        puntas_px, tamanos, obstaculos, pesos, limites, direcciones
    )
    _escribir_etiquetas(
        eje,
        variables,
        puntas,
        puntas_px,
        cajas_variables,
        margen_variables,
        **estilo_variables,
    )

    # Primero los creadores con más vecinos, que tienen menos lugar libre.
    nombres = scores["name"].astype(str).to_numpy()
    distancias = np.linalg.norm(puntos_px[:, None] - puntos_px[None], axis=2)
    orden = np.argsort(-(distancias < 90).sum(axis=1), kind="stable")
    estilo_creadores = {
        "fontsize": 11,
        "color": TINTA,
        "path_effects": [patheffects.withStroke(linewidth=3, foreground="white")],
    }
    tamanos = _tamanos_texto(eje, nombres[orden], renderer, 1, **estilo_creadores)
    cajas_creadores = ubicar_etiquetas(
        puntos_px[orden],
        tamanos,
        np.vstack([obstaculos, cajas_variables]),
        np.concatenate([pesos, np.full(len(cajas_variables), 10.0)]),
        limites,
    )
    _escribir_etiquetas(
        eje,
        nombres[orden],
        puntos[orden],
        puntos_px[orden],
        cajas_creadores,
        1,
        **estilo_creadores,
    )
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def leer_barrido(
    directorio: Path, pasos_mostrados: Sequence[int]
) -> list[tuple[int, str, pd.DataFrame, object]]:
    """Pasos pedidos del barrido: número, parámetro, resumen y valor elegido."""
    pasos = pd.read_csv(directorio / "resumen_pasos.csv").sort_values("paso")
    pasos = pasos[pasos["paso"].isin(pasos_mostrados)]
    resultado = []
    for fila in pasos.itertuples():
        archivo = directorio / f"paso_{fila.paso:02d}_{fila.parametro}_resumen.csv"
        resultado.append(
            (fila.paso, fila.parametro, pd.read_csv(archivo), fila.valor_elegido)
        )
    return resultado


def _figura_barrido(directorio: Path, ruta: Path) -> None:
    pasos = leer_barrido(directorio, PASOS_FIGURA)
    acoplamientos = pd.read_csv(
        directorio / "acoplamientos" / "resumen_acoplamientos.csv"
    )
    adoptados = acoplamientos[acoplamientos["adoptado"]]
    # QE del último paso, o la de un acoplamiento si se adoptó y la mejora.
    ultimo = pd.read_csv(directorio / "resumen_pasos.csv")["qe_media"].iloc[-1]
    qe_final = min([ultimo, *adoptados["qe_media"]])
    todos = pd.concat([p[2] for p in pasos])
    techo = (todos["qe_media"] + todos["qe_desvio"]).max()
    piso = (todos["qe_media"] - todos["qe_desvio"]).min()
    margen = 0.08 * (techo - piso)

    columnas = 2
    filas = int(np.ceil(len(pasos) / columnas))
    fig, ejes = plt.subplots(
        filas, columnas, figsize=(12, 4.3 * filas), sharey=True, squeeze=False
    )
    for eje, (paso, parametro, resumen, elegido) in zip(ejes.flat, pasos, strict=False):
        x = np.arange(len(resumen))
        es_elegido = resumen["valor"].map(lambda v, e=elegido: mismo_valor(v, e))
        colores = np.where(es_elegido, COLOR_ELEGIDO, COLOR_RESTO)
        for posicion, fila, color in zip(x, resumen.itertuples(), colores, strict=True):
            eje.errorbar(
                posicion,
                fila.qe_media,
                yerr=fila.qe_desvio,
                fmt="o",
                ms=9,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
            )
        eje.axhline(qe_final, color=TINTA_SECUNDARIA, lw=1, ls=":", zorder=0)
        etiquetas = [formato_valor(parametro, v) for v in resumen["valor"]]
        eje.set_xticks(
            x,
            etiquetas,
            rotation=35 if len(x) > 4 else 0,
            ha="right" if len(x) > 4 else "center",
            fontsize=9.5,
        )
        for etiqueta_x, elegida in zip(eje.get_xticklabels(), es_elegido, strict=True):
            if elegida:
                etiqueta_x.set_color(COLOR_ELEGIDO)
                etiqueta_x.set_fontweight("bold")
        eje.set_xlim(-0.6, len(x) - 0.4)
        eje.set_title(f"{paso}. {TITULOS_PASO.get(parametro, parametro)}", fontsize=12)
        eje.grid(axis="y", color=GRILLA, lw=0.6)
        if len(x) == 1:
            nota = "fija (barrido de control)"
        elif resumen["qe_media"].nunique() == 1:
            nota = "todas dan lo mismo"
        else:
            nota = ""
        if nota:
            eje.text(
                0.5,
                0.92,
                nota,
                transform=eje.transAxes,
                ha="center",
                fontsize=10,
                color=TINTA_SECUNDARIA,
            )
    for eje in ejes.flat[len(pasos) :]:
        eje.set_visible(False)
    for eje in ejes[:, 0]:
        eje.set_ylabel("QE (media ± desvío, 3 semillas)")
    ejes[0, 0].set_ylim(piso - margen, techo + margen)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def elegidos_por_mayoria(selecciones: pd.DataFrame, cliente: str) -> set[str]:
    """Creadores que la propuesta elige para el cliente en la mayoría de semillas."""
    propuesta = selecciones[
        (selecciones["estrategia"] == "propuesta") & (selecciones["cliente"] == cliente)
    ]
    veces = propuesta.groupby("name")["semilla"].nunique()
    return set(veces[veces >= propuesta["semilla"].nunique() / 2].index)


def tendencia(
    x: np.ndarray, y: np.ndarray, log_x: bool = False, log_y: bool = False
) -> tuple[float, np.ndarray, np.ndarray]:
    """Correlación y recta de mínimos cuadrados, con los ejes pedidos en log.

    Devuelve r y la recta evaluada entre el mínimo y el máximo de ``x``, en las
    unidades originales (con ``log_y`` la recta es una curva exponencial).
    """
    tx = np.log(x) if log_x else np.asarray(x, dtype=float)
    ty = np.log(y) if log_y else np.asarray(y, dtype=float)
    pendiente, ordenada = np.polyfit(tx, ty, 1)
    malla = np.linspace(tx.min(), tx.max(), 100)
    recta = ordenada + pendiente * malla
    return (
        float(np.corrcoef(tx, ty)[0, 1]),
        np.exp(malla) if log_x else malla,
        np.exp(recta) if log_y else recta,
    )


def _signo(valor: float) -> str:
    return _coma(valor, "+.2f").replace("-", "−")


def _eje_followers(eje) -> None:
    eje.set_xscale("log")
    eje.set_xlim(8e3, 1.05e6)
    eje.set_xticks([1e4, 1e5, 1e6], ["10 mil", "100 mil", "1 M"])
    eje.xaxis.set_minor_locator(NullLocator())
    eje.set_xlabel("Followers", fontsize=14)


def _eje_costo(eje) -> None:
    """Costo por argentino en escala log: separa los baratos, que se amontonan."""
    eje.set_yscale("log")
    eje.set_yticks([20, 50, 100, 200], ["20", "50", "100", "200"])
    eje.yaxis.set_minor_locator(NullLocator())


def _titulo(eje, titulo: str, subtitulo: str) -> None:
    """Título en negrita con un subtítulo gris debajo, alineados a la izquierda."""
    eje.annotate(
        subtitulo,
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        va="bottom",
        fontsize=14,
        color=TINTA_SECUNDARIA,
    )
    alto = 14 * 1.2 * (subtitulo.count("\n") + 1)
    eje.annotate(
        titulo,
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 16 + alto),
        textcoords="offset points",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color=TINTA,
    )


# Por panel: columnas x e y, si se ajusta en log(x) y en log(y), título,
# subtítulo y esquina libre para el r.
ENFOQUES = (
    (
        "followers",
        "non_follower_reach_pct",
        True,
        False,
        "El descubrimiento\ncrece con el tamaño",
        "% del alcance que\nno sigue al creador",
        "izquierda",
    ),
    (
        "followers",
        "intencion",
        True,
        False,
        "La intención\ncae con el tamaño",
        "Guardados + visitas al perfil\ncada mil alcanzados",
        "derecha",
    ),
    (
        "argentina_audience_pct",
        "usd_por_mil_argentinos",
        False,
        True,
        "El costo depende de la\naudiencia argentina",
        "USD por mil argentinos\nalcanzados en el tópico",
        "derecha",
    ),
)


def _figura_enfoques(creadores: pd.DataFrame, ruta: Path) -> None:
    fig, ejes = plt.subplots(1, len(ENFOQUES), figsize=(11, 4.6))
    for eje, (columna_x, columna_y, log_x, log_y, titulo, subtitulo, esquina) in zip(
        ejes, ENFOQUES, strict=True
    ):
        x, y = creadores[columna_x], creadores[columna_y]
        eje.scatter(
            x,
            y,
            s=55,
            color=COLOR_POSITIVA,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        r, xs, ys = tendencia(x, y, log_x, log_y)
        eje.plot(xs, ys, color=TINTA, lw=2.5, zorder=4)
        nota = f"r = {_signo(r)}"
        if columna_x != "followers":
            con_tamano, _, _ = tendencia(creadores["followers"], y, True, log_y)
            nota += f"\ncon el tamaño: r = {_signo(con_tamano)}"
        izquierda = esquina == "izquierda"
        eje.text(
            0.03 if izquierda else 0.97,
            0.97,
            nota,
            transform=eje.transAxes,
            ha="left" if izquierda else "right",
            va="top",
            fontsize=14,
            color=TINTA,
        )
        if log_x:
            _eje_followers(eje)
        else:
            eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
            eje.set_xlabel("Audiencia en Argentina", fontsize=14)
        if columna_y.endswith("_pct"):
            eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        # Lugar libre arriba para el texto del r.
        if log_y:
            _eje_costo(eje)
            eje.set_ylim(y.min() / 1.3, y.max() * 2.6)
        else:
            eje.set_ylim(0, y.max() * 1.25)
        eje.tick_params(labelsize=13)
        eje.grid(color=GRILLA, lw=0.8)
        eje.set_axisbelow(True)
        _titulo(eje, titulo, subtitulo)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def topico_por_neurona(asignaciones: pd.DataFrame, datos: pd.DataFrame) -> pd.Series:
    """Tópico de mayor promedio entre los creadores de cada neurona."""
    unidas = asignaciones[["creator_id", "neurona"]].merge(datos, on="creator_id")
    return unidas.groupby("neurona")[list(TOPICOS)].mean().idxmax(axis=1)


def aristas_grilla(celdas: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pares de índices de celdas (fila, columna) vecinas en la grilla."""
    indice = {celda: i for i, celda in enumerate(celdas)}
    pares = []
    for i, (fila, columna) in enumerate(celdas):
        for vecina in ((fila, columna + 1), (fila + 1, columna)):
            if vecina in indice:
                pares.append((i, indice[vecina]))
    return pares


def _semilla_representativa(directorio: Path) -> int:
    with (directorio / "config_usada.toml").open("rb") as archivo:
        return int(tomllib.load(archivo)["reproducibilidad"]["semilla_representativa"])


def _figura_mapa_biplot(
    scores: pd.DataFrame, datos: pd.DataFrame, directorio: Path, ruta: Path
) -> None:
    semilla = _semilla_representativa(directorio)
    pesos = pd.read_csv(directorio / "pesos_semillas.csv")
    pesos = pesos[pesos["semilla"] == semilla].sort_values("neurona")
    asignaciones = pd.read_csv(directorio / "asignaciones_semillas.csv")
    dominante = topico_por_neurona(
        asignaciones[asignaciones["semilla"] == semilla], datos
    )
    neuronas = pesos[["PC1", "PC2"]].to_numpy()
    celdas = list(zip(pesos["fila"], pesos["columna"], strict=True))
    puntos = scores[["PC1", "PC2"]].to_numpy()
    topico_creador = (
        datos.set_index("creator_id")
        .loc[scores["creator_id"], list(TOPICOS)]
        .idxmax(axis=1)
        .to_numpy()
    )

    fig, eje = plt.subplots(figsize=(8, 6.2), dpi=DPI)
    for topico, (nombre, color) in TOPICOS.items():
        eje.scatter(
            *puntos[topico_creador == topico].T,
            s=50,
            color=color,
            alpha=0.5,
            linewidth=0,
            zorder=2,
            label=nombre,
        )
    for i, j in aristas_grilla(celdas):
        eje.plot(*neuronas[[i, j]].T, color=TINTA_SECUNDARIA, lw=1.6, zorder=3)
    colores = [
        TOPICOS[dominante[n]][1] if n in dominante.index else "white"
        for n in pesos["neurona"]
    ]
    eje.scatter(
        *neuronas.T,
        s=230,
        marker="s",
        c=colores,
        edgecolor=TINTA,
        linewidth=1.3,
        zorder=4,
    )
    for relleno, nombre in [
        ("0.75", "Neurona (color de su tópico)"),
        ("white", "Neurona vacía"),
    ]:
        eje.scatter(
            [],
            [],
            s=150,
            marker="s",
            color=relleno,
            edgecolor=TINTA,
            linewidth=1.3,
            label=nombre,
        )
    eje.axhline(0, color=GRILLA, lw=1, zorder=0)
    eje.axvline(0, color=GRILLA, lw=1, zorder=0)
    eje.set_xlabel("PC1:   ← profundo   ·   masivo →", fontsize=14)
    eje.set_ylabel("PC2:   ← perros, audiencia arg.   ·   arte →", fontsize=14)
    eje.tick_params(labelsize=13)
    manijas, nombres = eje.get_legend_handles_labels()
    fig.legend(manijas, nombres, loc="lower center", ncol=3, fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    neuronas_px = eje.transData.transform(neuronas)
    tramos = np.vstack(
        [
            neuronas_px[i]
            + np.outer(np.linspace(0, 1, 30), neuronas_px[j] - neuronas_px[i])
            for i, j in aristas_grilla(celdas)
        ]
    )
    obstaculos = np.vstack(
        [
            _cajas_alrededor(neuronas_px, 12),
            _cajas_alrededor(eje.transData.transform(puntos), 6),
            _cajas_alrededor(tramos, 3),
        ]
    )
    pesos_obstaculos = np.concatenate(
        [
            np.full(len(neuronas_px), 6.0),
            np.full(len(puntos), 1.0),
            np.full(len(tramos), 2.0),
        ]
    )
    textos = [f"({fila},{columna})" for fila, columna in celdas]
    estilo = {
        "fontsize": 13.5,
        "color": TINTA,
        "path_effects": [patheffects.withStroke(linewidth=3, foreground="white")],
    }
    tamanos = _tamanos_texto(eje, textos, renderer, 1, **estilo)
    cajas = ubicar_etiquetas(
        neuronas_px,
        tamanos,
        obstaculos,
        pesos_obstaculos,
        eje.get_window_extent(renderer).extents,
    )
    _escribir_etiquetas(eje, textos, neuronas, neuronas_px, cajas, 1, **estilo)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def aportes_por_familia(
    datos: pd.DataFrame, pesos: Mapping[str, float]
) -> pd.DataFrame:
    """Aporte de cada familia de métricas al puntaje de ``campanas.py``.

    El puntaje es el promedio de los z-scores ponderado por ``|peso|``; cada
    métrica suma ``peso · z / Σ|peso|`` a su familia, así que cada fila suma
    el puntaje del creador.
    """
    total = sum(abs(peso) for peso in pesos.values())
    aportes = pd.DataFrame(index=datos.index)
    for metrica, peso in pesos.items():
        serie = datos[metrica]
        z = (serie - serie.mean()) / serie.std(ddof=1)
        familia = FAMILIAS[metrica]
        aportes[familia] = aportes.get(familia, 0.0) + peso * z / total
    return aportes


def candidatos(
    datos: pd.DataFrame, asignaciones: pd.DataFrame, topico: str
) -> list[str]:
    """Creadores de la región del mapa del tópico más los que lo tienen de tema."""
    region = asignaciones.set_index("creator_id")["neurona"].map(
        topico_por_neurona(asignaciones, datos)
    )
    propio = datos.set_index("creator_id")[list(TOPICOS)].idxmax(axis=1)
    return sorted(
        set(region[region == topico].index) | set(propio[propio == topico].index)
    )


def _figura_criterios(
    datos: pd.DataFrame,
    selecciones: pd.DataFrame,
    clientes: Mapping[str, dict],
    titulos: Mapping[str, tuple[str, str]],
    directorio_mapa: Path,
    ruta: Path,
    tamano: tuple[float, float],
) -> None:
    """Por qué se elige a cada creador: aporte de cada familia a su puntaje.

    Un panel por cliente de ``titulos`` (título y enfoque).
    """
    semilla = _semilla_representativa(directorio_mapa)
    asignaciones = pd.read_csv(directorio_mapa / "asignaciones_semillas.csv")
    asignaciones = asignaciones[asignaciones["semilla"] == semilla]
    por_id = datos.set_index("creator_id")

    fig, ejes = plt.subplots(1, len(titulos), figsize=tamano, dpi=DPI, squeeze=False)
    familias_usadas = set()
    for eje, (clave, (titulo, subtitulo)) in zip(ejes[0], titulos.items(), strict=True):
        cliente = clientes[clave]
        aportes = aportes_por_familia(por_id, cliente["metricas"])
        aportes = aportes.loc[candidatos(datos, asignaciones, cliente["topico"])]
        aportes["puntaje"] = aportes.sum(axis=1)
        aportes = aportes.sort_values("puntaje", ascending=False).head(FILAS_CRITERIO)
        nombres = por_id.loc[aportes.index, "name"].to_numpy()
        elegido = np.isin(nombres, list(elegidos_por_mayoria(selecciones, clave)))
        filas = np.arange(len(aportes))[::-1]
        derecha, izquierda = np.zeros(len(aportes)), np.zeros(len(aportes))
        for familia, (_, color) in ESTILO_FAMILIA.items():
            if familia not in aportes:
                continue
            familias_usadas.add(familia)
            valores = aportes[familia].to_numpy()
            colores = [to_rgba(color, 1.0 if e else 0.35) for e in elegido]
            positivos, negativos = valores.clip(min=0), valores.clip(max=0)
            eje.barh(filas, positivos, left=derecha, color=colores, height=0.7)
            eje.barh(filas, negativos, left=izquierda, color=colores, height=0.7)
            derecha += positivos
            izquierda += negativos
        eje.scatter(
            aportes["puntaje"],
            filas,
            marker="|",
            s=320,
            linewidths=2.5,
            color=TINTA,
            zorder=5,
        )
        eje.axvline(0, color=TINTA_SECUNDARIA, lw=1)
        eje.set_yticks(filas, nombres, fontsize=12.5)
        for texto, es_elegido in zip(eje.get_yticklabels(), elegido, strict=True):
            texto.set_color(TINTA if es_elegido else TINTA_SECUNDARIA)
            texto.set_fontweight("bold" if es_elegido else "normal")
        cantidad = int(elegido.sum())
        if elegido[:cantidad].all():
            eje.axhline(filas[cantidad - 1] - 0.5, color=TINTA_SECUNDARIA, lw=1)
        eje.tick_params(axis="y", length=0)
        eje.tick_params(axis="x", labelsize=12)
        eje.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: _coma(v, "g").replace("-", "−"))
        )
        # Las barras fijan los bordes del eje: se deja aire a mano.
        bajo, alto = min(izquierda.min(), 0.0), max(derecha.max(), 0.0)
        eje.set_xlim(bajo - 0.08 * (alto - bajo), alto + 0.08 * (alto - bajo))
        eje.spines["left"].set_visible(False)
        eje.grid(axis="x", color=GRILLA, lw=0.8)
        eje.set_axisbelow(True)
        eje.set_xlabel("Aporte al puntaje", fontsize=13)
        _titulo(eje, titulo, subtitulo)
    manijas = [
        Patch(color=color, label=nombre)
        for familia, (nombre, color) in ESTILO_FAMILIA.items()
        if familia in familias_usadas
    ]
    manijas.append(
        Line2D(
            [],
            [],
            marker="|",
            ls="none",
            markersize=16,
            markeredgewidth=2.5,
            color=TINTA,
            label="Puntaje total",
        )
    )
    columnas = 3 if len(titulos) > 1 else 2
    fig.legend(
        handles=manijas,
        loc="lower center",
        ncol=columnas,
        fontsize=12.5,
        frameon=False,
    )
    # Cada renglón de la leyenda ocupa unas 0,32 pulgadas.
    filas_leyenda = -(-len(manijas) // columnas)
    abajo = (0.32 * filas_leyenda + 0.1) / tamano[1]
    fig.tight_layout(rect=(0, abajo, 1, 1), w_pad=2)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pesos_relativos(pesos: Mapping[str, float]) -> pd.Series:
    """Parte de cada métrica en el total, ordenadas por familia y de mayor a menor.

    Cuenta el valor absoluto: un peso negativo penaliza la métrica, pero pesa
    igual en el puntaje.
    """
    total = sum(abs(peso) for peso in pesos.values())
    familias = list(ESTILO_FAMILIA)
    orden = sorted(pesos, key=lambda m: (familias.index(FAMILIAS[m]), -abs(pesos[m])))
    return pd.Series({m: abs(pesos[m]) / total for m in orden})


def formato_porcentaje(fraccion: float) -> str:
    """Porcentaje con coma decimal y a lo sumo un decimal."""
    return f"{_coma(round(100 * fraccion, 1), 'g')}%"


def _figura_pesos(clientes: Mapping[str, dict], ruta: Path) -> None:
    """Una barra al 100% por campaña: cuánto pesa cada métrica y cada familia."""
    nombres = {
        clave: titulo
        for clave, (titulo, _) in (CLIENTE_FUTBOL | CLIENTES_RESTO).items()
    }
    fig, eje = plt.subplots(figsize=(13, 0.9 + 1.15 * len(clientes)), dpi=DPI)
    textos = []
    for fila, (clave, cliente) in enumerate(clientes.items()):
        y = len(clientes) - 1 - fila
        inicio, tramos = 0.0, {}
        for metrica, peso in pesos_relativos(cliente["metricas"]).items():
            familia = FAMILIAS[metrica]
            color = ESTILO_FAMILIA[familia][1]
            eje.barh(
                y,
                peso,
                left=inicio,
                height=0.52,
                color=color,
                edgecolor="white",
                linewidth=2,
            )
            # Sobre el gris de afinidad el blanco no se lee.
            textos.append(
                (
                    eje.text(
                        inicio + peso / 2,
                        y,
                        f"{NOMBRES_METRICA[metrica]}\n{formato_porcentaje(peso)}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=TINTA if familia == "afinidad" else "white",
                    ),
                    peso,
                )
            )
            desde, _ = tramos.get(familia, (inicio, inicio))
            tramos[familia] = (desde, inicio + peso)
            inicio += peso
        # Llave arriba de cada familia con su peso total.
        for familia, (desde, hasta) in tramos.items():
            nombre, color = ESTILO_FAMILIA[familia]
            eje.plot(
                [desde + 0.004, hasta - 0.004],
                [y + 0.36, y + 0.36],
                color=color,
                lw=3,
                solid_capstyle="butt",
            )
            eje.text(
                (desde + hasta) / 2,
                y + 0.41,
                f"{nombre.split(' (')[0]} {formato_porcentaje(hasta - desde)}",
                ha="center",
                va="bottom",
                fontsize=12.5,
                color=TINTA_SECUNDARIA,
            )
    eje.set_yticks(
        range(len(clientes)),
        [nombres[clave] for clave in reversed(list(clientes))],
        fontsize=15,
        fontweight="bold",
        color=TINTA,
    )
    eje.tick_params(axis="y", length=0)
    eje.set_xlim(0, 1)
    eje.set_ylim(-0.4, len(clientes) - 0.2)
    eje.set_xticks([])
    for borde in eje.spines.values():
        borde.set_visible(False)
    fig.tight_layout()
    # Si el nombre no entra en su segmento, se achica la letra.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ancho_eje = eje.get_window_extent(renderer).width
    for texto, peso in textos:
        disponible = peso * ancho_eje - 8
        ancho = texto.get_window_extent(renderer).width
        if ancho > disponible:
            texto.set_fontsize(max(9.0, 12 * disponible / ancho))
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    correlaciones = pd.read_csv(args.correlaciones, index_col=0)
    cargas = pd.read_csv(args.cargas, index_col=0)
    _figura_correlaciones(
        correlaciones, cargas, salida / "fig_correlaciones_ordenadas.png"
    )
    autovalores = pd.read_csv(args.autovalores)
    _figura_cargas(cargas, autovalores, salida / "fig_cargas.png")
    _figura_biplot(
        pd.read_csv(args.scores),
        cargas,
        autovalores,
        pd.read_csv(args.datos),
        salida / "fig_biplot.png",
    )
    _figura_barrido(args.barrido, salida / "fig_barrido.png")
    creadores = pd.read_csv(args.campanas / "creadores_por_cliente.csv")
    selecciones = pd.read_csv(args.campanas / "selecciones.csv")
    _figura_enfoques(creadores, salida / "fig_enfoques.png")
    with args.config_campanas.open("rb") as archivo:
        clientes = tomllib.load(archivo)["clientes"]
    for titulos, nombre, tamano in [
        (CLIENTE_FUTBOL, "fig_enfoques_futbol.png", (7.5, 6.2)),
        (CLIENTES_RESTO, "fig_enfoques_resto.png", (13.5, 6.4)),
    ]:
        _figura_criterios(
            pd.read_csv(args.datos),
            selecciones,
            clientes,
            titulos,
            args.mapa,
            salida / nombre,
            tamano,
        )
    _figura_pesos(clientes, salida / "fig_pesos.png")
    for clave, cliente in clientes.items():
        _figura_pesos({clave: cliente}, salida / f"fig_pesos_{clave}.png")
    _figura_mapa_biplot(
        pd.read_csv(args.scores),
        pd.read_csv(args.datos),
        args.mapa,
        salida / "fig_mapa_biplot.png",
    )
    print(f"Figuras: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
