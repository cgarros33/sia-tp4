"""Tests del análisis de perfiles de Kohonen: mapa sobre el biplot y perfiles."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej1" / "kohonen-perfiles.py"
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
    """Carga el script de perfiles como módulo."""
    assert RUTA_SCRIPT.exists(), f"No existe el script: {RUTA_SCRIPT}"
    return _cargar(RUTA_SCRIPT, "kohonen_perfiles")


def test_r2_alineacion_lineal_perfecta(modulo: ModuleType) -> None:
    """Si una PC varía linealmente con la grilla, el R² es 1."""
    filas, columnas = np.divmod(np.arange(16), 4)
    valores = 2.0 * filas - 0.5 * columnas + 3.0
    assert modulo.r2_grilla(valores, filas, columnas) == pytest.approx(1.0)


def test_r2_alineacion_invariante_a_rotar_la_grilla(modulo: ModuleType) -> None:
    """Rotar o espejar la grilla no cambia el R²."""
    generador = np.random.default_rng(0)
    filas, columnas = np.divmod(np.arange(16), 4)
    valores = filas + generador.normal(scale=0.5, size=16)
    original = modulo.r2_grilla(valores, filas, columnas)
    rotado = modulo.r2_grilla(valores, columnas, 3 - filas)
    assert rotado == pytest.approx(original)
    assert 0.0 < original < 1.0


def test_proyectar_a_pcs_desde_variables(modulo: ModuleType) -> None:
    """Pesos en variables estandarizadas vuelven a sus coordenadas de PCs."""
    generador = np.random.default_rng(1)
    cargas, _ = np.linalg.qr(generador.normal(size=(5, 5)))
    nombres = [f"v{i}" for i in range(5)]
    tabla_cargas = pd.DataFrame(
        cargas, index=nombres, columns=[f"PC{i}" for i in range(1, 6)]
    )
    coordenadas = generador.normal(size=(3, 5))
    pesos = pd.DataFrame(coordenadas @ cargas.T, columns=nombres)

    proyectados = modulo.proyectar_a_pcs(pesos, tabla_cargas, ["PC1", "PC2"])
    np.testing.assert_allclose(proyectados, coordenadas[:, :2], atol=1e-12)


def test_proyectar_a_pcs_desde_pcs(modulo: ModuleType) -> None:
    """Pesos que ya están en PCs se devuelven tal cual."""
    tabla_cargas = pd.DataFrame(
        np.eye(3), index=["a", "b", "c"], columns=["PC1", "PC2", "PC3"]
    )
    pesos = pd.DataFrame({"PC1": [1.0, 2.0], "PC2": [3.0, 4.0]})
    proyectados = modulo.proyectar_a_pcs(pesos, tabla_cargas, ["PC1", "PC2"])
    np.testing.assert_allclose(proyectados, [[1.0, 3.0], [2.0, 4.0]])


def test_perfil_neuronas(modulo: ModuleType) -> None:
    """El perfil promedia en unidades originales y cuenta creadores."""
    datos = pd.DataFrame(
        {
            "creator_id": ["a", "b", "c"],
            "name": ["@a", "@b", "@c"],
            "followers": [100.0, 300.0, 1000.0],
        }
    )
    asignaciones = pd.DataFrame(
        {
            "creator_id": ["a", "b", "c"],
            "neurona": [0, 0, 2],
            "fila": [0, 0, 1],
            "columna": [0, 0, 0],
        }
    )
    perfil = modulo.perfil_neuronas(datos, asignaciones, ["followers"])

    assert perfil["neurona"].tolist() == [0, 2]
    assert perfil["creadores"].tolist() == [2, 1]
    assert perfil["followers"].tolist() == pytest.approx([200.0, 1000.0])
    assert perfil.loc[perfil["neurona"] == 0, "nombres"].item() == "@a, @b"


def test_integracion_smoke(modulo: ModuleType, tmp_path: Path) -> None:
    """Genera tablas y figuras para una corrida completa y una sobre PCs."""
    kohonen = _cargar(RUTA_KOHONEN, "kohonen_para_perfiles")
    comunes = ["--grilla", "3", "--iteraciones", "200", "--semillas", "0,1"]
    completa = tmp_path / "completa"
    pcs = tmp_path / "pcs"
    for config, salida in ((RUTA_CONFIG, completa), (RUTA_CONFIG_PCA10, pcs)):
        assert (
            kohonen.main(["--config", str(config), "--salida", str(salida)] + comunes)
            == 0
        )

    salida = tmp_path / "perfiles"
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
    alineacion = pd.read_csv(salida / "alineacion_pcs.csv")
    assert set(alineacion["corrida"]) == {"completa_k3", "pca10_k3"}
    assert alineacion["r2_media"].between(0.0, 1.0).all()
    for etiqueta in ("completa_k3", "pca10_k3"):
        carpeta = salida / etiqueta
        perfil = pd.read_csv(carpeta / "perfil_neuronas.csv")
        assert perfil["creadores"].sum() == 50
        assert (carpeta / "perfil_neuronas_z.csv").exists()
        assert (carpeta / "neuronas_pcs.csv").exists()
        assert (carpeta / "fig_mapa_biplot.png").exists()
        assert (carpeta / "fig_perfil_neuronas.png").exists()
    assert (salida / "resumen.txt").exists()
