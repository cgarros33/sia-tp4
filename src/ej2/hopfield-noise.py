#!/usr/bin/env python3
"""Barrido de ruido sobre redes de Hopfield entrenadas (TP4, ejercicio 2.1).

Carga las redes guardadas por ``hopfield.py`` (nunca entrena), genera cada
consulta ruidosa una sola vez y la recupera con todas las redes, para que la
comparación entre reglas sea justa. Cada invocación ejecuta un modelo de ruido,
la escalera de combinaciones (o uno de sus pasos) o la consolidación de los
resultados ya guardados.

No guarda estados completos de todas las consultas: solo la consulta de
ejemplo de cada nivel, en ``ejemplos.npz``. La energía por barrido se guarda
para todas.

Uso:
    python src/ej2/hopfield-noise.py --modelo aleatorio
    python src/ej2/hopfield-noise.py --combinaciones --paso 3
    python src/ej2/hopfield-noise.py --consolidar
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ / "src" / "ej2" / "hopfield-noise.toml"
MODELOS = ("aleatorio", "esquina", "finder", "rayones", "manchas", "bloques")
CLAVES_NIVEL = ["modelo", "orden_nivel", "nivel"]
# Métricas que se calculan por semilla y se resumen con media y desvío.
METRICAS_SEMILLA = (
    "exactitud",
    "tasa_incorrecto",
    "tasa_espurio",
    "tasa_ciclo",
    "tasa_no_converge",
    "barridos",
    "actualizaciones",
    "cambios_totales",
    "delta_energia_por_neurona",
    "hamming_final_normalizado",
    "proporcion_monotona",
    "fraccion_afectada",
    "fraccion_nominal",
    "hamming_inicial",
)
METRICAS_COMPARADAS = ("exactitud", "hamming_final_normalizado", "barridos")
TABLAS_CONSOLIDADAS = (
    "consultas.csv",
    "metricas_por_semilla.csv",
    "resumen_niveles.csv",
    "comparacion_reglas.csv",
    "resumen_por_patron.csv",
    "resumen_energia.csv",
)
# Intervalos de 5% de fracción afectada real, de 0 a 50%, para el mapa de calor
# global. El nivel 0,50 del ruido aleatorio invierte round(684,5) = 684 bits
# (49,96%) y cae en el último.
BORDES_INTERVALOS = np.round(np.arange(0.0, 0.501, 0.05), 2)


def _cargar_modulo(archivo: str, nombre: str) -> ModuleType:
    """Carga un script hermano de la misma carpeta como módulo."""
    ruta = Path(__file__).resolve().parent / archivo
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el script: {ruta}")
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


runner = _cargar_modulo("hopfield-runner.py", "hopfield_runner_noise")
motor = runner.motor
ruido = _cargar_modulo("ruido.py", "hopfield_ruido_noise")


@dataclass(frozen=True)
class Nivel:
    """Un nivel del barrido: los componentes activos con su valor."""

    orden: int
    etiqueta: str
    componentes: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class Resultados:
    """Tablas de una corrida del barrido."""

    consultas: pd.DataFrame
    energias: pd.DataFrame
    ejemplos: dict[str, np.ndarray]


def etiqueta_nivel(modelo: str, valor: Any) -> str:
    """Texto del nivel nominal de un modelo, para tablas y ejes."""
    if modelo == "manchas":
        cantidad, radio = valor
        return f"{cantidad}xr{radio}"
    return f"{valor:g}"


def niveles_modelo(config: dict, modelo: str) -> list[Nivel]:
    """Niveles de un modelo individual, en el orden del TOML."""
    return [
        Nivel(orden, etiqueta_nivel(modelo, valor), ((modelo, valor),))
        for orden, valor in enumerate(config["niveles"][modelo])
    ]


def nivel_combinacion(config: dict, paso: int) -> Nivel:
    """El paso ``paso`` de la escalera activa los primeros ``paso`` componentes."""
    combinaciones = config["combinaciones"]
    activos = combinaciones["componentes"][:paso]
    return Nivel(
        paso - 1,
        f"paso_{paso}",
        tuple((componente, combinaciones[componente]) for componente in activos),
    )


def crear_flujos(
    componentes: Sequence[tuple[str, Any]], semilla: int
) -> dict[str, np.random.Generator]:
    """Un flujo aleatorio por componente: agregar uno no cambia los demás.

    ``aleatorio`` usa ``default_rng(semilla)``, igual que el runner, así que
    reproduce sus consultas.
    """
    flujos = {}
    for componente, _ in componentes:
        semillas = (
            semilla
            if componente == "aleatorio"
            else [semilla, ruido.CODIGOS[componente]]
        )
        flujos[componente] = np.random.default_rng(semillas)
    return flujos


def generar_consulta(
    patron: np.ndarray,
    componentes: Sequence[tuple[str, Any]],
    flujos: dict[str, np.random.Generator],
    filas: int,
    columnas: int,
) -> tuple[np.ndarray, dict, int]:
    """Aplica los componentes en orden y devuelve la consulta.

    El ruido aleatorio se aplica primero y los borrados estructurales después,
    así que un borrado tiene prioridad sobre un bit invertido en la misma
    posición. Devuelve el estado, los parámetros efectivos de cada componente y
    la cantidad de píxeles dañados (la unión de las zonas de todos los
    componentes, cambien o no de valor).
    """
    estado = np.asarray(patron).astype(np.int8, copy=True)
    zona = np.zeros(estado.size, dtype=bool)
    parametros: dict[str, dict] = {}
    for componente, valor in componentes:
        rng = flujos[componente]
        if componente == "aleatorio":
            nuevo = runner.aplicar_ruido(estado, float(valor), rng)
            invertidos = nuevo != estado
            parametros[componente] = {"bits": int(invertidos.sum())}
            zona |= invertidos
            estado = nuevo
        else:
            mascara, efectivos = ruido.GENERADORES[componente](
                valor, filas, columnas, rng
            )
            parametros[componente] = efectivos
            zona |= mascara.reshape(-1)
            estado = ruido.borrar(estado, mascara)
    return estado, parametros, int(zona.sum())


def margen_inicial(estado: np.ndarray, indice_origen: int, red: Any) -> dict:
    """QR almacenado más cercano a la consulta y margen frente al origen.

    El margen es la distancia de Hamming al otro QR más cercano menos la
    distancia al origen: si es negativo, el daño dejó la consulta más cerca de
    otro QR que del suyo.
    """
    distancias = motor.calcular_distancias_hamming(estado, red.patrones)
    otras = np.delete(distancias, indice_origen)
    return {
        "nombre_inicial_mas_cercano": red.nombres[int(np.argmin(distancias))],
        "margen_inicial": int(otras.min() - distancias[indice_origen]),
    }


def ejecutar(
    redes: dict[str, Any],
    dinamicas: dict[str, dict],
    niveles: Sequence[Nivel],
    modelo: str,
    semillas: Sequence[int],
    repeticiones: int,
    patron_ejemplo: str,
    lector: Callable[[np.ndarray, int, int], str] | None = None,
) -> Resultados:
    """Genera las consultas de cada nivel y las recupera con todas las redes.

    Con ``lector``, cada consulta guarda también el texto que un lector de QR
    obtiene de la consulta dañada (``texto_inicial``) y del estado final
    (``texto_final``).
    """
    primera = next(iter(redes.values()))
    filas, columnas = primera.filas, primera.columnas
    n_neuronas = primera.n_neuronas
    registros: list[dict[str, object]] = []
    energias: list[dict[str, object]] = []
    iniciales, finales, trayectorias = [], {regla: [] for regla in redes}, {}

    for nivel in niveles:
        for semilla in semillas:
            flujos = crear_flujos(nivel.componentes, semilla)
            for indice_patron, (nombre, patron) in enumerate(
                zip(primera.nombres, primera.patrones, strict=True)
            ):
                for repeticion in range(repeticiones):
                    estado, parametros, area = generar_consulta(
                        patron, nivel.componentes, flujos, filas, columnas
                    )
                    consulta = f"{nombre}_rep{repeticion}"
                    ejemplo = (
                        nombre == patron_ejemplo
                        and semilla == semillas[0]
                        and repeticion == 0
                    )
                    if ejemplo:
                        iniciales.append(estado)
                    lectura = {}
                    if lector is not None:
                        lectura["texto_inicial"] = lector(estado, filas, columnas)
                    base = {
                        "modelo": modelo,
                        "orden_nivel": nivel.orden,
                        "nivel": nivel.etiqueta,
                        "semilla": semilla,
                        "repeticion": repeticion,
                        "consulta": consulta,
                        "nombre_origen": nombre,
                        "indice_patron": indice_patron,
                        "parametros": json.dumps(parametros),
                        "fraccion_nominal": area / n_neuronas,
                        "hamming_inicial": int(np.count_nonzero(estado != patron)),
                        **margen_inicial(estado, indice_patron, primera),
                    }
                    for regla, red in redes.items():
                        dinamica = dinamicas[regla]
                        resultado = motor.recuperar(
                            red.pesos,
                            estado,
                            modo=dinamica["modo"],
                            orden=dinamica["orden"],
                            max_barridos=dinamica["max_barridos"],
                            semilla=dinamica["semilla"],
                            registrar_actualizaciones=ejemplo,
                        )
                        clasificacion = runner.clasificar_consulta(
                            resultado, red, nombre
                        )
                        registros.append(
                            {
                                **base,
                                "regla": regla,
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
                                **runner.metricas_consulta(resultado, patron),
                                **lectura,
                            }
                        )
                        if lector is not None:
                            registros[-1]["texto_final"] = lector(
                                resultado.estado_final, filas, columnas
                            )
                        energias.extend(
                            {
                                "modelo": modelo,
                                "orden_nivel": nivel.orden,
                                "nivel": nivel.etiqueta,
                                "regla": regla,
                                "semilla": semilla,
                                "consulta": consulta,
                                "barrido": paso.barrido,
                                "energia": paso.energia,
                                "cambios": paso.cambios,
                            }
                            for paso in resultado.historial
                        )
                        if ejemplo:
                            finales[regla].append(resultado.estado_final)
                            trayectorias[f"energia_{regla}_{nivel.orden}"] = np.asarray(
                                resultado.energia_actualizaciones, dtype=np.float32
                            )

    indice_ejemplo = primera.nombres.index(patron_ejemplo)
    ejemplos = {
        "niveles": np.asarray([nivel.etiqueta for nivel in niveles], dtype=str),
        "ordenes": np.asarray([nivel.orden for nivel in niveles]),
        "patron": np.asarray(patron_ejemplo, dtype=str),
        "reglas": np.asarray(list(redes), dtype=str),
        "filas": np.asarray(filas),
        "columnas": np.asarray(columnas),
        "original": primera.patrones[indice_ejemplo].astype(np.int8),
        "iniciales": np.asarray(iniciales, dtype=np.int8),
        **{
            f"finales_{regla}": np.asarray(estados, dtype=np.int8)
            for regla, estados in finales.items()
        },
        **trayectorias,
    }
    return Resultados(pd.DataFrame(registros), pd.DataFrame(energias), ejemplos)


def metricas_por_semilla(consultas: pd.DataFrame) -> pd.DataFrame:
    """Tasas y medias de cada nivel, regla y semilla."""
    filas = []
    claves = [*CLAVES_NIVEL, "regla", "semilla"]
    for valores, grupo in consultas.groupby(claves, sort=True):
        resumen = runner.resumir_metricas(grupo)
        fila: dict[str, object] = dict(zip(claves, valores, strict=True))
        fila["consultas"] = resumen["consultas"]
        fila.update({tasa: resumen[tasa] for tasa in runner.TASAS})
        fila.update(
            {
                metrica: resumen[f"{metrica}_media"]
                for metrica in runner.METRICAS_RESUMIDAS
            }
        )
        fila["proporcion_monotona"] = resumen["proporcion_monotona"]
        for metrica in ("fraccion_afectada", "fraccion_nominal", "hamming_inicial"):
            fila[metrica] = float(grupo[metrica].mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def resumir_niveles(por_semilla: pd.DataFrame) -> pd.DataFrame:
    """Media y desvío muestral sobre semillas, en formato largo."""
    agrupado = por_semilla.groupby([*CLAVES_NIVEL, "regla"], sort=True)
    resumen = agrupado[list(METRICAS_SEMILLA)].agg(
        ["mean", lambda serie: serie.std(ddof=1)]
    )
    resumen.columns = [
        f"{metrica}_{'media' if estadistico == 'mean' else 'desvio'}"
        for metrica, estadistico in resumen.columns
    ]
    resumen["semillas"] = agrupado.size()
    resumen["consultas"] = agrupado["consultas"].sum()
    return resumen.reset_index()


def comparar_reglas(resumen: pd.DataFrame) -> pd.DataFrame:
    """Hebb y pseudo-inversa lado a lado, con la diferencia de medias."""
    columnas = [
        f"{metrica}_{estadistico}"
        for metrica in METRICAS_COMPARADAS
        for estadistico in ("media", "desvio")
    ]
    ancho = resumen.pivot(index=CLAVES_NIVEL, columns="regla", values=columnas)
    ancho.columns = [f"{columna}_{regla}" for columna, regla in ancho.columns]
    reglas = set(resumen["regla"])
    if {"hebb", "pseudoinversa"} <= reglas:
        for metrica in METRICAS_COMPARADAS:
            ancho[f"{metrica}_diferencia"] = (
                ancho[f"{metrica}_media_pseudoinversa"] - ancho[f"{metrica}_media_hebb"]
            )
    return ancho.reset_index()


def resumir_por_patron(consultas: pd.DataFrame) -> pd.DataFrame:
    """Exactitud por QR de origen: media y desvío sobre semillas."""
    correcto = consultas["tipo_resultado"] == "almacenado_correcto"
    claves = [*CLAVES_NIVEL, "regla", "nombre_origen"]
    por_semilla = correcto.groupby(
        [consultas[clave] for clave in [*claves, "semilla"]]
    ).mean()
    resumen = por_semilla.groupby(level=claves).agg(
        exactitud_media="mean", exactitud_desvio=lambda serie: serie.std(ddof=1)
    )
    return resumen.reset_index()


def resumir_energia(energias: pd.DataFrame, n_neuronas: int) -> pd.DataFrame:
    """Energía media por barrido de cada nivel y regla.

    Las consultas que ya se estabilizaron conservan su energía final en los
    barridos siguientes, para que la media sea comparable entre barridos.
    """
    filas = []
    claves = [*CLAVES_NIVEL, "regla"]
    for valores, grupo in energias.groupby(claves, sort=True):
        tabla = grupo.pivot(
            index=["semilla", "consulta"], columns="barrido", values="energia"
        ).ffill(axis=1)
        activas = grupo["barrido"].value_counts()
        for barrido in tabla.columns:
            serie = tabla[barrido]
            filas.append(
                {
                    **dict(zip(claves, valores, strict=True)),
                    "barrido": int(barrido),
                    "energia_media": float(serie.mean()),
                    "energia_desvio": float(serie.std(ddof=1)),
                    "energia_por_neurona_media": float(serie.mean()) / n_neuronas,
                    "consultas_activas": int(activas[barrido]),
                }
            )
    return pd.DataFrame(filas)


def exactitud_por_intervalo(consultas: pd.DataFrame) -> pd.DataFrame:
    """Exactitud por modelo, regla e intervalo de 5% de fracción afectada real."""
    etiquetas = [
        f"{inicio:.0%}–{fin:.0%}"
        for inicio, fin in zip(
            BORDES_INTERVALOS[:-1], BORDES_INTERVALOS[1:], strict=True
        )
    ]
    datos = consultas.assign(
        intervalo=pd.cut(
            consultas["fraccion_afectada"],
            bins=BORDES_INTERVALOS,
            labels=etiquetas,
            right=False,
        ),
        correcto=consultas["tipo_resultado"] == "almacenado_correcto",
    )
    claves = ["modelo", "regla", "intervalo"]
    por_semilla = datos.groupby([*claves, "semilla"], observed=True)["correcto"].mean()
    resumen = por_semilla.groupby(level=claves, observed=True).agg(
        exactitud_media="mean", exactitud_desvio=lambda serie: serie.std(ddof=1)
    )
    resumen["consultas"] = datos.groupby(claves, observed=True).size()
    resumen = resumen.reset_index()
    resumen["orden_intervalo"] = resumen["intervalo"].map(etiquetas.index).astype(int)
    return resumen.sort_values(["modelo", "regla", "orden_intervalo"])


def _valor_toml(valor: Any) -> str:
    """Formatea un valor simple o una lista para un archivo TOML."""
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, (int, float)):
        return repr(valor)
    if isinstance(valor, str):
        return json.dumps(valor, ensure_ascii=False)
    if isinstance(valor, (list, tuple)):
        return "[" + ", ".join(_valor_toml(elemento) for elemento in valor) + "]"
    raise TypeError(f"No se puede escribir en TOML: {valor!r}")


def guardar_toml(config: dict, ruta: Path) -> None:
    """Guarda una configuración de secciones con valores simples o listas."""
    lineas: list[str] = []
    for seccion, valores in config.items():
        lineas.append(f"[{seccion}]")
        lineas += [
            f"{clave} = {_valor_toml(valor)}" for clave, valor in valores.items()
        ]
        lineas.append("")
    ruta.write_text("\n".join(lineas).rstrip() + "\n", encoding="utf-8")


def construir_resumen(titulo: str, resumen: pd.DataFrame) -> str:
    """Resumen textual: exactitud y errores por nivel y regla."""
    lineas = [titulo, "Media ± desvío muestral sobre semillas.", ""]
    for fila in resumen.itertuples():
        lineas.append(
            f"  {fila.nivel:<8} {fila.regla:<14} "
            f"exactitud {fila.exactitud_media:.3f} ± {fila.exactitud_desvio:.3f} | "
            f"incorrecto {fila.tasa_incorrecto_media:.3f} | "
            f"espurio {fila.tasa_espurio_media:.3f} | "
            f"barridos {fila.barridos_media:.2f} | "
            f"H/N {fila.hamming_final_normalizado_media:.4f} | "
            f"afectada {fila.fraccion_afectada_media:.4f}"
        )
    return "\n".join(lineas) + "\n"


def guardar_resultados(
    directorio: Path,
    config_usada: dict,
    resultados: Resultados,
    n_neuronas: int,
    titulo: str,
) -> pd.DataFrame:
    """Guarda las tablas de una corrida y devuelve el resumen por nivel."""
    directorio.mkdir(parents=True, exist_ok=True)
    guardar_toml(config_usada, directorio / "config_usada.toml")
    resultados.consultas.to_csv(directorio / "consultas.csv", index=False)
    resultados.energias.to_csv(directorio / "energia.csv", index=False)
    por_semilla = metricas_por_semilla(resultados.consultas)
    por_semilla.to_csv(directorio / "metricas_por_semilla.csv", index=False)
    resumen = resumir_niveles(por_semilla)
    resumen.to_csv(directorio / "resumen_niveles.csv", index=False)
    comparar_reglas(resumen).to_csv(directorio / "comparacion_reglas.csv", index=False)
    resumir_por_patron(resultados.consultas).to_csv(
        directorio / "resumen_por_patron.csv", index=False
    )
    resumir_energia(resultados.energias, n_neuronas).to_csv(
        directorio / "resumen_energia.csv", index=False
    )
    np.savez_compressed(directorio / "ejemplos.npz", **resultados.ejemplos)
    (directorio / "resumen.txt").write_text(
        construir_resumen(titulo, resumen), encoding="utf-8"
    )
    return resumen


COLUMNAS_LEGIBILIDAD = (
    "legible_inicial",
    "lee_origen_inicial",
    "legible_final",
    "lee_origen_final",
)


def lecturas_patrones(
    red: Any, lector: Callable[[np.ndarray, int, int], str]
) -> pd.DataFrame:
    """Texto que se lee en cada QR almacenado y en el estado mezcla."""
    estados = [*zip(red.nombres, red.patrones, strict=True)]
    estados.append(("mezcla", motor.estado_mezcla(red.patrones)))
    filas = []
    for nombre, estado in estados:
        texto = lector(estado, red.filas, red.columnas)
        filas.append({"estado": nombre, "texto": texto, "legible": texto != ""})
    return pd.DataFrame(filas)


def tabla_legibilidad(consultas: pd.DataFrame, patrones: pd.DataFrame) -> pd.DataFrame:
    """Qué QR almacenado se lee en la consulta dañada y en el estado final.

    La lectura es el nombre del QR cuyo texto coincide, ``""`` si no se lee y
    ``"otro"`` si se lee un texto que no es de ningún QR almacenado.
    """
    por_texto = {
        fila.texto: fila.estado for fila in patrones.itertuples() if fila.texto
    }

    def nombre(texto: str) -> str:
        return por_texto.get(texto, "otro") if texto else ""

    claves = [*CLAVES_NIVEL, "semilla", "repeticion", "consulta", "nombre_origen"]
    tabla = consultas[[*claves, "regla", "tipo_resultado"]].copy()
    tabla["lectura_inicial"] = consultas["texto_inicial"].map(nombre)
    tabla["lectura_final"] = consultas["texto_final"].map(nombre)
    tabla["legible_inicial"] = tabla["lectura_inicial"] != ""
    tabla["lee_origen_inicial"] = tabla["lectura_inicial"] == tabla["nombre_origen"]
    tabla["legible_final"] = tabla["lectura_final"] != ""
    tabla["lee_origen_final"] = tabla["lectura_final"] == tabla["nombre_origen"]
    return tabla


def resumir_legibilidad(tabla: pd.DataFrame) -> pd.DataFrame:
    """Proporción de QR legibles por nivel y regla: media y desvío sobre semillas."""
    por_semilla = tabla.groupby([*CLAVES_NIVEL, "regla", "semilla"], sort=True)[
        list(COLUMNAS_LEGIBILIDAD)
    ].mean()
    agrupado = por_semilla.groupby(level=[*CLAVES_NIVEL, "regla"])
    resumen = agrupado.agg(["mean", lambda serie: serie.std(ddof=1)])
    resumen.columns = [
        f"{metrica}_{'media' if estadistico == 'mean' else 'desvio'}"
        for metrica, estadistico in resumen.columns
    ]
    resumen["semillas"] = agrupado.size()
    resumen["consultas"] = tabla.groupby([*CLAVES_NIVEL, "regla"]).size()
    return resumen.reset_index()


def verificar_consistencia(guardadas: pd.DataFrame, nuevas: pd.DataFrame) -> None:
    """Las consultas regeneradas deben ser las mismas que las del barrido guardado."""
    claves = ["orden_nivel", "semilla", "consulta", "regla"]
    columnas = ["hamming_inicial", "tipo_resultado", "barridos"]
    viejas = guardadas.set_index(claves)[columnas].sort_index().astype(str)
    recientes = nuevas.set_index(claves)[columnas].sort_index().astype(str)
    if not (viejas.index.equals(recientes.index) and viejas.equals(recientes)):
        raise ValueError(
            "Las consultas regeneradas no coinciden con consultas.csv: "
            "la configuración no es la del barrido guardado."
        )


def consolidar(salida: Path) -> None:
    """Une las tablas de los modelos y de los pasos de la escalera."""
    grupos = (
        (
            [salida / modelo for modelo in MODELOS],
            salida / "consolidado",
        ),
        (
            sorted(
                (salida / "combinaciones").glob("paso_*"),
                key=lambda carpeta: int(carpeta.name.split("_")[1]),
            ),
            salida / "combinaciones" / "consolidado",
        ),
    )
    for carpetas, destino in grupos:
        carpetas = [c for c in carpetas if (c / "consultas.csv").exists()]
        if not carpetas:
            continue
        destino.mkdir(parents=True, exist_ok=True)
        archivos = (*TABLAS_CONSOLIDADAS, "legibilidad.csv", "legibilidad_resumen.csv")
        for archivo in archivos:
            presentes = [c for c in carpetas if (c / archivo).exists()]
            if not presentes:
                continue
            tabla = pd.concat(
                [
                    pd.read_csv(carpeta / archivo, dtype={"nivel": str})
                    for carpeta in presentes
                ],
                ignore_index=True,
            ).drop_duplicates()
            tabla.to_csv(destino / archivo, index=False)
        consultas = pd.read_csv(destino / "consultas.csv", dtype={"nivel": str})
        exactitud_por_intervalo(consultas).to_csv(
            destino / "exactitud_por_intervalo.csv", index=False
        )
        print(f"Consolidado en {destino}", file=sys.stderr)


def cargar_config(ruta: Path) -> dict:
    """Carga y valida la configuración del barrido."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la configuración: {ruta}")
    with ruta.open("rb") as archivo:
        config = tomllib.load(archivo)
    # [combinaciones] es opcional: un barrido exploratorio puede no tener escalera.
    for seccion in ("redes", "barrido", "niveles"):
        if not isinstance(config.get(seccion), dict):
            raise ValueError(f"Falta la sección obligatoria: {seccion}.")
    if set(config["niveles"]) - set(MODELOS):
        raise ValueError(f"Los niveles deben ser de los modelos {MODELOS}.")
    barrido = config["barrido"]
    semillas = barrido.get("semillas")
    if (
        not isinstance(semillas, list)
        or not semillas
        or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in semillas
        )
    ):
        raise ValueError("barrido.semillas debe ser una lista de enteros >= 0.")
    repeticiones = barrido.get("repeticiones")
    if not isinstance(repeticiones, int) or repeticiones < 1:
        raise ValueError("barrido.repeticiones debe ser un entero >= 1.")
    combinaciones = config.get("combinaciones", {"componentes": []})
    if set(combinaciones["componentes"]) - set(MODELOS):
        raise ValueError(f"Los componentes deben ser de {MODELOS}.")
    return config


def cargar_redes(config: dict) -> tuple[dict[str, Any], dict[str, dict], dict]:
    """Carga las redes y su dinámica; todas deben guardar los mismos patrones."""
    redes, dinamicas, rutas = {}, {}, {}
    for regla, directorio in config["redes"].items():
        ruta = motor.resolver_ruta(directorio)
        redes[regla] = motor.cargar_red(ruta)
        dinamicas[regla] = runner.cargar_config_red(ruta)["modelo"]
        rutas[regla] = str(ruta)
    primera = next(iter(redes.values()))
    for regla, red in redes.items():
        if red.nombres != primera.nombres or not np.array_equal(
            red.patrones, primera.patrones
        ):
            raise ValueError(f"La red {regla} no guarda los mismos patrones.")
    if config["barrido"].get("patron_ejemplo") not in primera.nombres:
        raise ValueError("barrido.patron_ejemplo debe ser uno de los patrones.")
    return redes, dinamicas, rutas


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta un modelo, la escalera de combinaciones o la consolidación."""
    parser = argparse.ArgumentParser(
        description="Barrido de ruido sobre redes de Hopfield entrenadas."
    )
    parser.add_argument("--config", type=Path, default=RUTA_CONFIG)
    accion = parser.add_mutually_exclusive_group(required=True)
    accion.add_argument("--modelo", choices=MODELOS, help="Modelo de ruido a correr.")
    accion.add_argument(
        "--combinaciones",
        action="store_true",
        help="Escalera de combinaciones; todos los pasos o el de --paso.",
    )
    accion.add_argument(
        "--consolidar", action="store_true", help="Une los resultados guardados."
    )
    parser.add_argument("--paso", type=int, default=None, help="Paso de la escalera.")
    parser.add_argument(
        "--legibilidad",
        action="store_true",
        help=(
            "Con --modelo o --combinaciones: regenera y recupera las mismas consultas, "
            "las lee con un lector de QR y guarda solo las tablas de legibilidad."
        ),
    )
    parser.add_argument(
        "--salida", type=Path, default=None, help="Pisa barrido.salida del TOML."
    )
    argumentos = parser.parse_args(argv)

    config = cargar_config(motor.resolver_ruta(argumentos.config))
    cantidad_pasos = len(config.get("combinaciones", {}).get("componentes", []))
    if argumentos.combinaciones and cantidad_pasos == 0:
        parser.error("La configuración no tiene escalera de combinaciones.")
    if argumentos.modelo is not None and argumentos.modelo not in config["niveles"]:
        parser.error(f"La configuración no tiene niveles para {argumentos.modelo}.")
    if argumentos.paso is not None:
        if not argumentos.combinaciones:
            parser.error("--paso requiere --combinaciones.")
        if not 1 <= argumentos.paso <= cantidad_pasos:
            parser.error(f"--paso debe estar entre 1 y {cantidad_pasos}.")
    if argumentos.legibilidad and argumentos.consolidar:
        parser.error("--legibilidad requiere --modelo o --combinaciones.")
    salida = motor.resolver_ruta(argumentos.salida or config["barrido"]["salida"])

    if argumentos.consolidar:
        consolidar(salida)
        return 0

    redes, dinamicas, rutas = cargar_redes(config)
    barrido = config["barrido"]
    n_neuronas = next(iter(redes.values())).n_neuronas
    if argumentos.modelo is not None:
        corridas = [
            (
                argumentos.modelo,
                niveles_modelo(config, argumentos.modelo),
                salida / argumentos.modelo,
            )
        ]
    else:
        pasos = [argumentos.paso] if argumentos.paso else range(1, cantidad_pasos + 1)
        corridas = [
            (
                "combinaciones",
                [nivel_combinacion(config, paso)],
                salida / "combinaciones" / f"paso_{paso}",
            )
            for paso in pasos
        ]

    lector = None
    if argumentos.legibilidad:
        lector = _cargar_modulo("lector_qr.py", "hopfield_lector_qr").leer
        patrones = lecturas_patrones(next(iter(redes.values())), lector)
        salida.mkdir(parents=True, exist_ok=True)
        patrones.to_csv(salida / "legibilidad_patrones.csv", index=False)

    for modelo, niveles, directorio in corridas:
        resultados = ejecutar(
            redes,
            dinamicas,
            niveles,
            modelo,
            barrido["semillas"],
            barrido["repeticiones"],
            barrido["patron_ejemplo"],
            lector,
        )
        if lector is not None:
            guardadas = directorio / "consultas.csv"
            if not guardadas.exists():
                raise FileNotFoundError(
                    f"Primero hay que correr el barrido: {guardadas}"
                )
            verificar_consistencia(
                pd.read_csv(guardadas, dtype={"nivel": str}), resultados.consultas
            )
            tabla = tabla_legibilidad(resultados.consultas, patrones)
            tabla.to_csv(directorio / "legibilidad.csv", index=False)
            resumir_legibilidad(tabla).to_csv(
                directorio / "legibilidad_resumen.csv", index=False
            )
            print(f"Legibilidad guardada en {directorio}", file=sys.stderr)
            continue
        config_usada = {
            "redes": rutas,
            "barrido": {**barrido, "salida": str(salida)},
            "corrida": {
                "modelo": modelo,
                "etiquetas": [nivel.etiqueta for nivel in niveles],
                "componentes": [
                    componente for componente, _ in niveles[-1].componentes
                ],
            },
            "niveles": {
                nivel.etiqueta: [valor for _, valor in nivel.componentes]
                for nivel in niveles
            },
        }
        guardar_resultados(
            directorio,
            config_usada,
            resultados,
            n_neuronas,
            f"Ruido: {directorio.name}",
        )
        print(f"Resultados guardados en {directorio}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
