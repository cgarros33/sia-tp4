"""Tests del script de figuras de presentación del ejercicio 1."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "figuras-presentacion.py"


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script como módulo."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    especificacion = importlib.util.spec_from_file_location(
        "figuras_presentacion", RUTA_SCRIPT
    )
    assert especificacion is not None and especificacion.loader is not None
    cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = cargado
    especificacion.loader.exec_module(cargado)
    return cargado


def test_etiqueta_legible(modulo: ModuleType) -> None:
    """Los nombres de variables se acortan para el gráfico."""
    assert modulo.etiqueta("average_reach") == "reach"
    assert modulo.etiqueta("saves_per_1000") == "saves /1000"
    assert modulo.etiqueta("topic_art_pct") == "topic art %"


def test_orden_por_carga(modulo: ModuleType) -> None:
    """Las variables se ordenan de mayor a menor carga en la componente."""
    cargas = pd.DataFrame({"PC1": [0.1, -0.5, 0.7]}, index=["a", "b", "c"])
    assert modulo.orden_por_carga(cargas) == ["c", "a", "b"]


def test_cargas_principales(modulo: ModuleType) -> None:
    """Las de mayor carga absoluta, de menor a mayor para el gráfico de barras."""
    cargas = pd.DataFrame({"PC1": [0.1, -0.5, 0.7, 0.2]}, index=["a", "b", "c", "d"])
    principales = modulo.cargas_principales(cargas, "PC1", 3)
    assert list(principales.index) == ["b", "d", "c"]


def test_formato_valor(modulo: ModuleType) -> None:
    """Los valores del barrido se muestran como en la presentación."""
    assert modulo.formato_valor("radio_inicial_relativo", 0.3) == "0,3·k"
    assert modulo.formato_valor("iteraciones", 10000.0) == "10.000"
    assert modulo.formato_valor("eta_inicial", 0.05) == "0,05"
    assert modulo.formato_valor("topologia", "rectangular") == "rectangular"


def test_mismo_valor(modulo: ModuleType) -> None:
    """Compara valores del barrido guardados como texto o número."""
    assert modulo.mismo_valor("0.3", 0.30000000)
    assert modulo.mismo_valor("lineal", "lineal")
    assert not modulo.mismo_valor("0.3", 0.4)


def test_leer_barrido_filtra_pasos(modulo: ModuleType) -> None:
    """Solo se leen los pasos pedidos, con su número original."""
    pasos = modulo.leer_barrido(modulo.RUTA_BARRIDO, (4, 5, 6, 8))
    assert [(paso[0], paso[1]) for paso in pasos] == [
        (4, "radio_inicial_relativo"),
        (5, "decaimiento_radio"),
        (6, "eta_inicial"),
        (8, "iteraciones"),
    ]


def test_superposicion(modulo: ModuleType) -> None:
    """Área de intersección de una caja con cada una de otras."""
    caja = np.array([0.0, 0.0, 2.0, 2.0])
    otras = np.array([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0]])
    assert modulo.superposicion(caja, otras) == pytest.approx([1.0, 0.0])


def test_ubicar_etiquetas_sin_superponer(modulo: ModuleType) -> None:
    """Etiquetas de puntos encimados quedan en cajas disjuntas y dentro del área."""
    anclas = np.full((4, 2), 200.0)
    tamanos = np.tile([60.0, 12.0], (4, 1))
    cajas = modulo.ubicar_etiquetas(
        anclas,
        tamanos,
        obstaculos=np.empty((0, 4)),
        pesos=np.empty(0),
        limites=np.array([0.0, 0.0, 400.0, 400.0]),
    )
    for i in range(len(cajas)):
        assert modulo.superposicion(cajas[i], np.delete(cajas, i, axis=0)).sum() == 0
    assert cajas[:, :2].min() >= 0 and cajas[:, 2:].max() <= 400


def test_ubicar_etiquetas_mas_cerca_de_su_ancla(modulo: ModuleType) -> None:
    """Cada etiqueta queda más cerca de su punto que del de otra."""
    anclas = np.array([[200.0, 200.0], [230.0, 200.0]])
    cajas = modulo.ubicar_etiquetas(
        anclas,
        np.tile([60.0, 12.0], (2, 1)),
        obstaculos=np.empty((0, 4)),
        pesos=np.empty(0),
        limites=np.array([0.0, 0.0, 400.0, 400.0]),
    )
    for caja, propia, otra in zip(cajas, anclas, anclas[::-1], strict=True):
        distancias = modulo.distancia_a_caja(caja, np.array([propia, otra]))
        assert distancias[0] < distancias[1]


def test_pegados_al_texto(modulo: ModuleType) -> None:
    """Puntos a la altura del texto y junto a su principio o a su final."""
    caja = np.array([100.0, 100.0, 160.0, 112.0])
    puntos = np.array([[90.0, 106.0], [170.0, 106.0], [130.0, 140.0], [40.0, 106.0]])
    assert list(modulo.pegados_al_texto(caja, puntos)) == [True, True, False, False]


def test_agrupar_cercanos(modulo: ModuleType) -> None:
    """Los puntos encadenados a menos de la distancia forman un mismo grupo."""
    puntos = np.array([[0.0, 0.0], [100.0, 0.0], [10.0, 0.0], [300.0, 0.0], [18.0, 0]])
    assert modulo.agrupar_cercanos(puntos, 12) == [[0, 2, 4], [1], [3]]


def test_ubicar_etiquetas_lejos_de_vecinos(modulo: ModuleType) -> None:
    """Un punto sin etiqueta también cuenta: la etiqueta no se le arrima."""
    ancla = np.array([[200.0, 200.0]])
    vecino = np.array([[200.0, 190.0]])
    cajas = modulo.ubicar_etiquetas(
        ancla,
        np.array([[60.0, 12.0]]),
        obstaculos=np.empty((0, 4)),
        pesos=np.empty(0),
        limites=np.array([0.0, 0.0, 400.0, 400.0]),
        vecinos=vecino,
    )
    distancias = modulo.distancia_a_caja(cajas[0], np.vstack([ancla, vecino]))
    assert distancias[0] < distancias[1]


def test_ubicar_etiquetas_sigue_direccion(modulo: ModuleType) -> None:
    """Con una dirección preferida, la etiqueta queda de ese lado del ancla."""
    cajas = modulo.ubicar_etiquetas(
        np.array([[200.0, 200.0]]),
        np.array([[60.0, 12.0]]),
        obstaculos=np.empty((0, 4)),
        pesos=np.empty(0),
        limites=np.array([0.0, 0.0, 400.0, 400.0]),
        direcciones=np.array([[0.0, -1.0]]),
    )
    assert cajas[0, 3] <= 200.0


def test_elegidos_por_mayoria(modulo: ModuleType) -> None:
    """Cuenta solo la propuesta del cliente y exige la mayoría de las semillas."""
    selecciones = pd.DataFrame(
        {
            "cliente": ["a", "a", "a", "a", "a", "b"],
            "estrategia": ["propuesta"] * 4 + ["followers", "propuesta"],
            "semilla": [0, 1, 2, 0, 1, 0],
            "name": ["@x", "@x", "@x", "@y", "@y", "@z"],
        }
    )
    assert modulo.elegidos_por_mayoria(selecciones, "a") == {"@x"}
    assert modulo.elegidos_por_mayoria(selecciones, "b") == {"@z"}


def test_tendencia(modulo: ModuleType) -> None:
    """Recta en log(x) y curva exponencial (recta en log(y)), con su r."""
    x = np.array([1e3, 1e4, 1e5])
    r, xs, ys = modulo.tendencia(x, np.array([10.0, 20.0, 30.0]), log_x=True)
    assert r == pytest.approx(1.0)
    assert ys[0] == pytest.approx(10.0) and ys[-1] == pytest.approx(30.0)
    assert xs[0] == pytest.approx(1e3) and xs[-1] == pytest.approx(1e5)
    r, _, ys = modulo.tendencia(
        np.array([1.0, 2.0, 3.0]), np.array([100.0, 10.0, 1.0]), log_y=True
    )
    assert r == pytest.approx(-1.0)
    assert ys[0] == pytest.approx(100.0) and ys[-1] == pytest.approx(1.0)


def test_topico_por_neurona(modulo: ModuleType) -> None:
    """Tópico de mayor promedio entre los creadores de cada neurona."""
    asignaciones = pd.DataFrame({"creator_id": ["a", "b", "c"], "neurona": [0, 0, 5]})
    datos = pd.DataFrame(
        {
            "creator_id": ["a", "b", "c"],
            "topic_football_pct": [60.0, 10.0, 0.0],
            "topic_dogs_pct": [0.0, 40.0, 0.0],
            "topic_makeup_pct": [40.0, 50.0, 10.0],
            "topic_art_pct": [0.0, 0.0, 90.0],
        }
    )
    dominante = modulo.topico_por_neurona(asignaciones, datos)
    assert dominante.to_dict() == {0: "topic_makeup_pct", 5: "topic_art_pct"}


def test_aristas_grilla(modulo: ModuleType) -> None:
    """Une cada celda con la de su derecha y la de abajo, si existen."""
    celdas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert modulo.aristas_grilla(celdas) == [(0, 1), (0, 2), (1, 3), (2, 3)]


def test_candidatos(modulo: ModuleType) -> None:
    """Región del mapa del tópico más los creadores con ese tema principal."""
    datos = pd.DataFrame(
        {
            "creator_id": ["a", "b", "c", "d"],
            "topic_football_pct": [90.0, 10.0, 60.0, 0.0],
            "topic_dogs_pct": [10.0, 50.0, 0.0, 80.0],
            "topic_makeup_pct": [0.0, 40.0, 40.0, 20.0],
            "topic_art_pct": [0.0, 0.0, 0.0, 0.0],
        }
    )
    # Neurona 0 (a y b) es de fútbol en promedio; neurona 1 (c y d), de perros.
    asignaciones = pd.DataFrame(
        {"creator_id": ["a", "b", "c", "d"], "neurona": [0, 0, 1, 1]}
    )
    assert modulo.candidatos(datos, asignaciones, "topic_football_pct") == [
        "a",
        "b",
        "c",
    ]
    assert modulo.candidatos(datos, asignaciones, "topic_dogs_pct") == ["b", "c", "d"]


def test_aportes_suman_el_puntaje_de_campanas(modulo: ModuleType) -> None:
    """Los aportes por familia suman el puntaje con que campanas.py ordena."""
    spec = importlib.util.spec_from_file_location(
        "campanas", RAIZ / "src" / "ej1" / "campanas.py"
    )
    assert spec is not None and spec.loader is not None
    campanas = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = campanas
    spec.loader.exec_module(campanas)
    with (RAIZ / "src" / "ej1" / "campanas.toml").open("rb") as archivo:
        clientes = tomllib.load(archivo)["clientes"]
    por_id = pd.read_csv(RAIZ / "datos" / "creadores.csv").set_index("creator_id")
    for cliente in clientes.values():
        aportes = modulo.aportes_por_familia(por_id, cliente["metricas"])
        assert set(aportes.columns) <= set(modulo.ESTILO_FAMILIA)
        puntaje = campanas.puntaje(
            campanas.agregar_metricas(por_id, cliente["topico"]), cliente["metricas"]
        )
        assert np.allclose(aportes.sum(axis=1), puntaje)


def test_pesos_relativos(modulo: ModuleType) -> None:
    """Parte de cada métrica en el total, agrupadas por familia y de mayor a menor."""
    pesos = {
        "comments_per_1000": 0.5,
        "argentina_audience_pct": 1.0,
        "non_follower_reach_pct": 1.0,
        "profile_visits_per_1000": 0.5,
    }
    relativos = modulo.pesos_relativos(pesos)
    assert list(relativos.index) == [
        "argentina_audience_pct",
        "non_follower_reach_pct",
        "comments_per_1000",
        "profile_visits_per_1000",
    ]
    assert relativos.to_numpy() == pytest.approx([1 / 3, 1 / 3, 1 / 6, 1 / 6])
    # Un peso negativo (penalizar el costo) cuenta por su valor absoluto.
    assert modulo.pesos_relativos({"price_usd": -1.0}).iloc[0] == 1.0


def test_formato_porcentaje(modulo: ModuleType) -> None:
    """Porcentajes con coma decimal y a lo sumo un decimal."""
    assert modulo.formato_porcentaje(0.125) == "12,5%"
    assert modulo.formato_porcentaje(1 / 11) == "9,1%"
    assert modulo.formato_porcentaje(0.25) == "25%"
    assert modulo.formato_porcentaje(2.5 / 4.5) == "55,6%"
    assert modulo.formato_porcentaje(1.0) == "100%"


def test_corrida_completa(modulo: ModuleType, tmp_path: Path) -> None:
    """Con los resultados guardados genera todas las figuras."""
    assert modulo.main(["--salida", str(tmp_path)]) == 0
    assert (tmp_path / "fig_correlaciones_ordenadas.png").exists()
    assert (tmp_path / "fig_cargas.png").exists()
    assert (tmp_path / "fig_barrido.png").exists()
    assert (tmp_path / "fig_biplot.png").exists()
    assert (tmp_path / "fig_enfoques.png").exists()
    assert (tmp_path / "fig_enfoques_futbol.png").exists()
    assert (tmp_path / "fig_enfoques_resto.png").exists()
    assert (tmp_path / "fig_mapa_biplot.png").exists()
    assert (tmp_path / "fig_pesos.png").exists()
    for cliente in ("futbol", "refugios", "belleza", "arte"):
        assert (tmp_path / f"fig_pesos_{cliente}.png").exists()
