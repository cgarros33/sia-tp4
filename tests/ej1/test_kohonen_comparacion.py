"""Tests del análisis comparativo de Kohonen: 27 variables contra PC1..PC10."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "kohonen-comparacion.py"
RUTA_KOHONEN = RAIZ / "src" / "ej1" / "kohonen.py"
RUTA_CONFIG = RAIZ / "src" / "ej1" / "kohonen.toml"
RUTA_CONFIG_PCA10 = RAIZ / "src" / "ej1" / "kohonen-pca10.toml"


def _cargar(ruta: Path, nombre: str) -> ModuleType:
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el script de comparación como módulo."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    return _cargar(RUTA_SCRIPT, "kohonen_comparacion")


def test_ari_particiones_identicas(modulo: ModuleType) -> None:
    """Dos particiones iguales concuerdan perfectamente."""
    etiquetas = np.array([0, 0, 1, 1, 2, 2])
    assert modulo.indice_rand_ajustado(etiquetas, etiquetas) == pytest.approx(1.0)


def test_ari_ignora_nombres_de_grupo(modulo: ModuleType) -> None:
    """Renombrar los grupos no cambia la concordancia."""
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 9, 9, 7, 7])
    assert modulo.indice_rand_ajustado(a, b) == pytest.approx(1.0)


def test_ari_valores_conocidos(modulo: ModuleType) -> None:
    """Casos calculados a mano con la fórmula de Hubert y Arabie."""
    a = np.array([0, 0, 1, 1])
    assert modulo.indice_rand_ajustado(a, np.array([0, 0, 1, 2])) == pytest.approx(
        4 / 7
    )
    assert modulo.indice_rand_ajustado(a, np.array([0, 1, 0, 1])) == pytest.approx(-0.5)


def test_ari_un_solo_grupo_en_ambas(modulo: ModuleType) -> None:
    """Si las dos particiones tienen un único grupo, concuerdan."""
    unos = np.zeros(5, dtype=int)
    assert modulo.indice_rand_ajustado(unos, unos) == pytest.approx(1.0)


def test_retroproyectar_recupera_puntos_del_subespacio(modulo: ModuleType) -> None:
    """Un punto que vive en el subespacio de las PCs vuelve sin error."""
    generador = np.random.default_rng(0)
    cargas, _ = np.linalg.qr(generador.normal(size=(6, 6)))
    cargas_q = cargas[:, :3]
    punto = generador.normal(size=3) @ cargas_q.T
    scores = punto @ cargas_q

    reconstruido = modulo.retroproyectar(scores[None, :], cargas_q)
    np.testing.assert_allclose(reconstruido[0], punto, atol=1e-12)


def test_qe_asignada(modulo: ModuleType) -> None:
    """La QE usa la neurona asignada por el modelo, no la más cercana."""
    datos = np.array([[0.0, 0.0], [3.0, 4.0]])
    pesos = np.array([[0.0, 0.0], [3.0, 0.0]])
    bmu = np.array([1, 0])
    assert modulo.qe_asignada(datos, pesos, bmu) == pytest.approx((3.0 + 5.0) / 2)


def test_desglose_topografico(modulo: ModuleType) -> None:
    """Clasifica la segunda BMU como vecina, diagonal o lejana."""
    # Grilla 3 x 3 con pesos sobre una recta: la neurona n tiene peso n.
    filas, columnas = np.divmod(np.arange(9), 3)
    pesos = np.arange(9, dtype=float)[:, None]
    # 0.4 -> (0, 1) vecinas; 2.6 -> (3, 2): (1,0)-(0,2) lejana;
    # 3.6 -> (4, 3): (1,1)-(1,0) vecina; 2.4 -> (2, 3): (0,2)-(1,0) lejana.
    datos = np.array([[0.4], [2.6], [3.6], [2.4]])
    desglose = modulo.desglose_topografico(datos, pesos, filas, columnas)
    assert desglose == pytest.approx({"vecina": 0.5, "diagonal": 0.0, "lejana": 0.5})

    # Con la neurona 4 en el peso 1.1 la segunda BMU de 0.4 pasa a ser diagonal.
    pesos_diagonal = pesos.copy()
    pesos_diagonal[1] = 50.0
    pesos_diagonal[4] = 1.1
    desglose = modulo.desglose_topografico(datos[:1], pesos_diagonal, filas, columnas)
    assert desglose == pytest.approx({"vecina": 0.0, "diagonal": 1.0, "lejana": 0.0})


def test_concordancia_misma_corrida_excluye_misma_semilla(
    modulo: ModuleType,
) -> None:
    """La estabilidad entre semillas no compara una semilla consigo misma."""
    asignaciones = pd.DataFrame(
        {
            "semilla": [0, 0, 0, 0, 1, 1, 1, 1],
            "creator_id": ["a", "b", "c", "d"] * 2,
            "neurona": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    valores = modulo.concordancia(asignaciones, asignaciones, misma_corrida=True)
    assert valores == pytest.approx([-0.5])

    cruzados = modulo.concordancia(asignaciones, asignaciones, misma_corrida=False)
    assert len(cruzados) == 4


def test_integracion_smoke(modulo: ModuleType, tmp_path: Path) -> None:
    """Compara una corrida chica completa contra una de PCs y escribe todo."""
    kohonen = _cargar(RUTA_KOHONEN, "kohonen_para_comparacion")
    comunes = ["--grilla", "3", "--iteraciones", "200", "--semillas", "0,1"]
    completa = tmp_path / "completa"
    pcs = tmp_path / "pcs"
    assert (
        kohonen.main(
            ["--config", str(RUTA_CONFIG), "--salida", str(completa)] + comunes
        )
        == 0
    )
    assert (
        kohonen.main(
            ["--config", str(RUTA_CONFIG_PCA10), "--salida", str(pcs)] + comunes
        )
        == 0
    )

    salida = tmp_path / "comparacion"
    codigo = modulo.main(
        [
            "--corrida",
            f"completa_k3={completa}",
            "--corrida",
            f"pca10_k3={pcs}",
            "--salida",
            str(salida),
        ]
    )

    assert codigo == 0
    por_semilla = pd.read_csv(salida / "metricas_por_semilla.csv")
    resumen = pd.read_csv(salida / "metricas_resumen.csv")
    concordancias = pd.read_csv(salida / "concordancia.csv")

    assert len(por_semilla) == 4
    assert set(resumen["corrida"]) == {"completa_k3", "pca10_k3"}
    # En la corrida completa la QE común coincide con la QE propia del modelo.
    fila = por_semilla[por_semilla["corrida"] == "completa_k3"]
    np.testing.assert_allclose(fila["qe_comun"], fila["qe_propia"], atol=1e-6)
    # El desglose reproduce el TE del motor: diagonal + lejana = TE.
    np.testing.assert_allclose(
        por_semilla["te_diagonal"] + por_semilla["te_lejana"],
        por_semilla["te"],
        atol=1e-12,
    )
    # Tres pares: estabilidad de cada corrida y el cruce entre ambas.
    assert len(concordancias) == 3
    assert (salida / "fig_metricas.png").exists()
    assert (salida / "fig_concordancia.png").exists()
    assert (salida / "fig_topografia.png").exists()
    assert (salida / "resumen.txt").exists()
