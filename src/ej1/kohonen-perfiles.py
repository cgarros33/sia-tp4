#!/usr/bin/env python3
"""Perfiles de las neuronas de Kohonen y su ubicación en el plano PC1-PC2.

Script de análisis: solo lee corridas finales de ``kohonen.py`` y resultados de
1.1; no entrena nada. Para cada corrida genera:

- ``neuronas_pcs.csv`` y ``fig_mapa_biplot.png``: los pesos de cada neurona
  proyectados sobre PC1-PC2, con las aristas de la grilla, los creadores y las
  cargas principales, para comparar las regiones del mapa con el biplot.
- ``perfil_neuronas.csv``: promedio de cada variable original (en sus unidades)
  de los creadores asignados a cada neurona, con sus nombres.
- ``perfil_neuronas_z.csv`` y ``fig_perfil_neuronas.png``: el mismo perfil en
  desvíos estándar respecto de los 50 creadores.

Además ``alineacion_pcs.csv`` mide, para cada semilla, cuánto de la variación
de cada PC entre neuronas se explica linealmente por la posición en la grilla
(R² de ``PC ~ fila + columna``, invariante a rotar o espejar la grilla).
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "datos" / "creadores.csv"
RUTA_CARGAS = RAIZ / "resultados" / "ej1" / "pca_0.90" / "cargas.csv"
RUTA_SCORES = RAIZ / "resultados" / "ej1" / "pca_0.90" / "scores.csv"
RUTA_SALIDA = RAIZ / "resultados" / "ej1" / "ej1.2-perfiles"

CORRIDAS_POR_DEFECTO = [
    "completa_k3=resultados/ej1/ej1.2-full/kohonen_k3",
    "completa_k4=resultados/ej1/ej1.2-full/kohonen_k4",
    "completa_k5=resultados/ej1/ej1.2-full/kohonen_k5",
    "pca10_k3=resultados/ej1/ej1.2-pca10/kohonen_k3",
    "pca10_k4=resultados/ej1/ej1.2-pca10/kohonen_k4",
    "pca10_k4_euclidiana=resultados/ej1/ej1.2-pca10/kohonen_k4_euclidiana",
    "pca10_k5=resultados/ej1/ej1.2-pca10/kohonen_k5",
]

COLUMNAS_ID = ["creator_id", "name"]
COLUMNAS_GRILLA = {"semilla", "neurona", "fila", "columna"}
PCS_ALINEACION = ["PC1", "PC2", "PC3", "PC4"]
CANTIDAD_FLECHAS = 8


def resolver_ruta(ruta: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio."""
    camino = Path(ruta)
    if camino.is_absolute():
        return camino
    return RAIZ / camino


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Perfiles de neuronas de Kohonen y mapa sobre el biplot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corrida",
        action="append",
        default=None,
        help="Corrida a analizar como etiqueta=directorio. Se puede repetir.",
    )
    parser.add_argument("--datos", type=Path, default=RUTA_DATOS)
    parser.add_argument("--cargas", type=Path, default=RUTA_CARGAS)
    parser.add_argument("--scores", type=Path, default=RUTA_SCORES)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA)
    return parser.parse_args(argv)


def r2_grilla(valores: np.ndarray, filas: np.ndarray, columnas: np.ndarray) -> float:
    """R² de la regresión lineal de ``valores`` sobre la posición en la grilla."""
    valores = np.asarray(valores, dtype=float)
    diseno = np.column_stack([np.ones_like(valores), filas, columnas]).astype(float)
    coeficientes, *_ = np.linalg.lstsq(diseno, valores, rcond=None)
    residuo = valores - diseno @ coeficientes
    total = float(np.sum((valores - valores.mean()) ** 2))
    if total == 0.0:
        return 0.0
    return 1.0 - float(np.sum(residuo**2)) / total


def proyectar_a_pcs(
    pesos: pd.DataFrame, cargas: pd.DataFrame, pcs: Sequence[str]
) -> np.ndarray:
    """Coordenadas de los pesos sobre las PCs pedidas.

    Si los pesos ya están en PCs se devuelven esas columnas; si están en las
    variables estandarizadas se proyectan con las cargas (``s = w E``).
    """
    if set(pcs).issubset(pesos.columns):
        return pesos[list(pcs)].to_numpy(dtype=float)
    return (
        pesos[list(cargas.index)].to_numpy(dtype=float) @ cargas[list(pcs)].to_numpy()
    )


def perfil_neuronas(
    datos: pd.DataFrame, asignaciones: pd.DataFrame, variables: Sequence[str]
) -> pd.DataFrame:
    """Promedio de cada variable por neurona, con cantidad y nombres."""
    unidas = asignaciones[["creator_id", "neurona", "fila", "columna"]].merge(
        datos, on="creator_id"
    )
    agrupadas = unidas.groupby(["neurona", "fila", "columna"], sort=True)
    perfil = agrupadas[list(variables)].mean()
    perfil.insert(0, "creadores", agrupadas.size())
    perfil["nombres"] = agrupadas["name"].agg(lambda nombres: ", ".join(nombres))
    return perfil.reset_index()


def _color_neurona(fila: int, columna: int, k: int) -> tuple[float, float, float]:
    """Color que codifica la posición en la grilla: fila → rojo, columna → azul."""
    escala = max(k - 1, 1)
    return (0.15 + 0.75 * fila / escala, 0.45, 0.15 + 0.75 * columna / escala)


def _figura_mapa_biplot(
    etiqueta: str,
    neuronas: pd.DataFrame,
    creadores: pd.DataFrame,
    cargas: pd.DataFrame,
    k: int,
    ruta: Path,
) -> None:
    fig, eje = plt.subplots(figsize=(10, 8))

    posicion = {
        (int(n.fila), int(n.columna)): (n.PC1, n.PC2) for n in neuronas.itertuples()
    }
    for (fila, columna), (x, y) in posicion.items():
        for vecina in ((fila, columna + 1), (fila + 1, columna)):
            if vecina in posicion:
                x2, y2 = posicion[vecina]
                eje.plot([x, x2], [y, y2], color="0.55", lw=1.2, zorder=1)

    colores_creadores = [
        _color_neurona(int(c.fila), int(c.columna), k) for c in creadores.itertuples()
    ]
    eje.scatter(
        creadores["PC1"],
        creadores["PC2"],
        c=colores_creadores,
        s=28,
        alpha=0.75,
        zorder=2,
        label="Creadores (color de su neurona)",
    )

    for neurona in neuronas.itertuples():
        color = _color_neurona(int(neurona.fila), int(neurona.columna), k)
        viva = neurona.creadores > 0
        eje.scatter(
            neurona.PC1,
            neurona.PC2,
            s=90 + 40 * neurona.creadores,
            facecolors=color if viva else "white",
            edgecolors="black",
            linewidths=1.2,
            marker="s",
            zorder=3,
        )
        eje.annotate(
            f"({neurona.fila},{neurona.columna})",
            (neurona.PC1, neurona.PC2),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            zorder=4,
        )

    largo = np.hypot(cargas["PC1"], cargas["PC2"])
    principales = largo.sort_values(ascending=False).index[:CANTIDAD_FLECHAS]
    escala = 0.8 * float(np.abs(creadores[["PC1", "PC2"]]).max().min()) / largo.max()
    for variable in principales:
        x, y = (
            cargas.loc[variable, "PC1"] * escala,
            cargas.loc[variable, "PC2"] * escala,
        )
        eje.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "#B22222", "lw": 1.1},
            zorder=0,
        )
        eje.text(x * 1.07, y * 1.07, variable, color="#B22222", fontsize=7.5, zorder=0)

    eje.scatter(
        [],
        [],
        marker="s",
        facecolors="0.6",
        edgecolors="black",
        s=90,
        label="Neurona (tamaño ∝ creadores)",
    )
    eje.scatter(
        [],
        [],
        marker="s",
        facecolors="white",
        edgecolors="black",
        s=90,
        label="Neurona muerta",
    )
    eje.axhline(0, color="0.85", lw=0.8, zorder=0)
    eje.axvline(0, color="0.85", lw=0.8, zorder=0)
    eje.set_xlabel("PC1")
    eje.set_ylabel("PC2")
    eje.set_title(
        f"{etiqueta}: grilla {k}×{k} proyectada sobre PC1-PC2\n"
        "Líneas grises: vecinas en la grilla. Color: fila → rojo, columna → azul. "
        f"Flechas: {CANTIDAD_FLECHAS} cargas más largas (escaladas)",
        fontsize=10,
    )
    eje.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _figura_perfil(etiqueta: str, perfil_z: pd.DataFrame, variables, ruta) -> None:
    valores = perfil_z[list(variables)].to_numpy()
    etiquetas = [
        f"({p.fila},{p.columna})  n={p.creadores}" for p in perfil_z.itertuples()
    ]
    fig, eje = plt.subplots(figsize=(17, 0.42 * len(perfil_z) + 3.2))
    imagen = eje.imshow(valores, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    for i, j in np.ndindex(valores.shape):
        eje.text(
            j,
            i,
            f"{valores[i, j]:.1f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if abs(valores[i, j]) > 1.6 else "black",
        )
    eje.set_xticks(
        range(len(variables)), variables, rotation=60, ha="right", fontsize=8
    )
    eje.set_yticks(range(len(etiquetas)), etiquetas, fontsize=8)
    eje.set_ylabel("Neurona (fila, columna) y creadores asignados")
    eje.set_title(
        f"{etiqueta}: perfil medio de los creadores de cada neurona "
        "(desvíos estándar respecto de los 50 creadores; neuronas muertas omitidas)",
        fontsize=10,
    )
    fig.colorbar(imagen, ax=eje, fraction=0.02, pad=0.01, label="z medio")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def analizar_corrida(
    etiqueta: str,
    directorio: Path,
    datos: pd.DataFrame,
    z: pd.DataFrame,
    scores: pd.DataFrame,
    cargas: pd.DataFrame,
    salida: Path,
) -> pd.DataFrame:
    """Genera tablas y figuras de una corrida y devuelve su alineación con PCs."""
    with (directorio / "config_usada.toml").open("rb") as archivo:
        config = tomllib.load(archivo)
    if config["modelo"]["topologia"] != "rectangular":
        raise ValueError(f"{etiqueta}: solo se grafica la grilla rectangular.")
    k = int(config["modelo"]["grilla"])
    representativa = int(config["reproducibilidad"]["semilla_representativa"])

    pesos = pd.read_csv(directorio / "pesos_semillas.csv")
    asignaciones = pd.read_csv(directorio / "asignaciones_semillas.csv")

    alineacion = []
    for semilla, grupo in pesos.groupby("semilla"):
        grupo = grupo.sort_values("neurona")
        coordenadas = proyectar_a_pcs(grupo, cargas, PCS_ALINEACION)
        for indice, pc in enumerate(PCS_ALINEACION):
            alineacion.append(
                {
                    "corrida": etiqueta,
                    "semilla": int(semilla),
                    "pc": pc,
                    "r2": r2_grilla(
                        coordenadas[:, indice],
                        grupo["fila"].to_numpy(),
                        grupo["columna"].to_numpy(),
                    ),
                }
            )

    pesos_rep = pesos[pesos["semilla"] == representativa].sort_values("neurona")
    asignadas = asignaciones[asignaciones["semilla"] == representativa]
    conteos = asignadas["neurona"].value_counts()
    neuronas = pesos_rep[["neurona", "fila", "columna"]].reset_index(drop=True)
    neuronas["creadores"] = neuronas["neurona"].map(conteos).fillna(0).astype(int)
    coordenadas = proyectar_a_pcs(pesos_rep, cargas, PCS_ALINEACION)
    for indice, pc in enumerate(PCS_ALINEACION):
        neuronas[pc] = coordenadas[:, indice]

    creadores = asignadas[["creator_id", "fila", "columna"]].merge(
        scores[["PC1", "PC2"]], left_on="creator_id", right_index=True
    )

    variables_originales = [
        c
        for c in datos.select_dtypes(include=np.number).columns
        if c not in COLUMNAS_ID
    ]
    perfil = perfil_neuronas(datos, asignadas, variables_originales)
    perfil_z = perfil_neuronas(
        z.reset_index().merge(datos[COLUMNAS_ID], on="creator_id"),
        asignadas,
        list(z.columns),
    )

    carpeta = salida / etiqueta
    carpeta.mkdir(parents=True, exist_ok=True)
    neuronas.to_csv(carpeta / "neuronas_pcs.csv", index=False, float_format="%.6f")
    perfil.to_csv(carpeta / "perfil_neuronas.csv", index=False, float_format="%.4f")
    perfil_z.to_csv(carpeta / "perfil_neuronas_z.csv", index=False, float_format="%.4f")
    _figura_mapa_biplot(
        etiqueta, neuronas, creadores, cargas, k, carpeta / "fig_mapa_biplot.png"
    )
    _figura_perfil(
        etiqueta, perfil_z, list(z.columns), carpeta / "fig_perfil_neuronas.png"
    )
    return pd.DataFrame(alineacion)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = parse_args(argv)
    cargas = pd.read_csv(resolver_ruta(args.cargas), index_col=0)
    cargas = cargas[[c for c in cargas.columns if c.startswith("PC")]]
    datos = pd.read_csv(resolver_ruta(args.datos))
    scores = pd.read_csv(resolver_ruta(args.scores)).set_index("creator_id")

    matriz = datos[list(cargas.index)].to_numpy(dtype=float)
    z = pd.DataFrame(
        (matriz - matriz.mean(axis=0)) / matriz.std(axis=0, ddof=1),
        index=datos["creator_id"],
        columns=list(cargas.index),
    )

    salida = resolver_ruta(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    alineaciones = []
    for especificacion in args.corrida or CORRIDAS_POR_DEFECTO:
        etiqueta, _, directorio = especificacion.partition("=")
        if not directorio:
            raise ValueError(f"Corrida inválida, se espera etiqueta=dir: {etiqueta}")
        alineaciones.append(
            analizar_corrida(
                etiqueta,
                resolver_ruta(directorio),
                datos,
                z,
                scores,
                cargas,
                salida,
            )
        )

    por_semilla = pd.concat(alineaciones, ignore_index=True)
    resumen = (
        por_semilla.groupby(["corrida", "pc"], sort=False)["r2"]
        .agg(
            r2_media="mean",
            r2_desvio=lambda serie: (
                float(np.std(serie, ddof=1)) if len(serie) > 1 else 0.0
            ),
            semillas="size",
        )
        .reset_index()
    )
    por_semilla.to_csv(
        salida / "alineacion_pcs_por_semilla.csv", index=False, float_format="%.6f"
    )
    resumen.to_csv(salida / "alineacion_pcs.csv", index=False, float_format="%.6f")

    tabla = resumen.pivot(index="corrida", columns="pc", values="r2_media")
    tabla = tabla.loc[resumen["corrida"].unique(), PCS_ALINEACION]
    lineas = [
        "Perfiles de neuronas y alineación con PCs — TP4, ejercicio 1.2",
        "",
        "R² de PC ~ fila + columna sobre los pesos de las neuronas (media sobre",
        "semillas): cuánto de la variación de cada PC entre neuronas sigue un",
        "gradiente lineal a lo largo de la grilla.",
        "",
        tabla.to_string(float_format=lambda valor: f"{valor:.3f}"),
        "",
        "Por corrida, en su carpeta: neuronas_pcs.csv, perfil_neuronas.csv,",
        "perfil_neuronas_z.csv, fig_mapa_biplot.png y fig_perfil_neuronas.png",
        "(semilla representativa de cada corrida).",
    ]
    texto = "\n".join(lineas) + "\n"
    (salida / "resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Resultados: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
