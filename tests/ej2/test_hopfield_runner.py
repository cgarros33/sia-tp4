"""Tests del ejercicio 2.2: runner de una red de Hopfield entrenada."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_RUNNER = RAIZ / "src" / "ej2" / "hopfield-runner.py"
RUTA_MOTOR = RAIZ / "src" / "ej2" / "hopfield.py"


def _cargar_modulo(ruta: Path, nombre: str) -> ModuleType:
    """Carga un script como módulo para poder probar sus funciones."""
    assert ruta.exists(), f"No existe el script: {ruta}"
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    assert especificacion is not None
    assert especificacion.loader is not None
    modulo_cargado = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo_cargado
    especificacion.loader.exec_module(modulo_cargado)
    return modulo_cargado


@pytest.fixture(scope="module")
def motor() -> ModuleType:
    """Carga el motor de Hopfield."""
    return _cargar_modulo(RUTA_MOTOR, "hopfield_motor_test")


@pytest.fixture(scope="module")
def modulo() -> ModuleType:
    """Carga el runner de Hopfield."""
    return _cargar_modulo(RUTA_RUNNER, "hopfield_runner")


def _crear_red(
    tmp_path: Path,
    motor: ModuleType,
) -> tuple[Path, Path]:
    """Crea una red sintética de un patrón, todos unos, en 2x2."""
    dataset = motor.Dataset(
        nombres=("A",),
        patrones=np.ones((1, 4), dtype=np.int8),
        filas=2,
        columnas=2,
    )
    ruta_datos = tmp_path / "letras.csv"
    motor.escribir_dataset(dataset, ruta_datos)

    red_dir = tmp_path / "red"
    pesos = motor.calcular_pesos(dataset.patrones)
    motor.guardar_red(red_dir, dataset, pesos)

    config = deepcopy(motor.CONFIG_POR_DEFECTO)
    config["datos"]["ruta"] = str(ruta_datos)
    config["salida"]["directorio"] = str(red_dir)
    motor.guardar_config(config, red_dir / "config_usada.toml")

    return red_dir, ruta_datos


@pytest.fixture
def red_sintetica(
    tmp_path: Path,
    motor: ModuleType,
) -> tuple[Path, Path]:
    """Red sintética de un patrón para las pruebas del runner."""
    return _crear_red(tmp_path, motor)


def _resultado(
    motor: ModuleType,
    estado_final: list[int],
    *,
    convergio: bool = True,
    ciclo: bool = False,
) -> object:
    """Construye un resultado de recuperación para las pruebas de clasificación."""
    estado = np.asarray(estado_final, dtype=int)
    return motor.ResultadoRecuperacion(
        estado_inicial=estado.copy(),
        estado_final=estado,
        barridos=1,
        actualizaciones=len(estado),
        energia_inicial=0.0,
        energia_final=0.0,
        historial=[],
        convergio=convergio,
        ciclo=ciclo,
        motivo="estable" if convergio else ("ciclo" if ciclo else "max_barridos"),
    )


def test_renderizar_matriz(modulo: ModuleType) -> None:
    """El modo gráfico debe usar * para 1 y espacio para -1."""
    texto = modulo.renderizar_matriz(np.array([1, 1, 1, -1]), filas=2, columnas=2)

    assert texto == "**\n* "


def test_aplicar_ruido_cantidad(modulo: ModuleType) -> None:
    """El ruido debe invertir exactamente round(f * N) bits distintos."""
    base = np.ones(10, dtype=int)
    rng = np.random.default_rng(0)

    ruidoso = modulo.aplicar_ruido(base, 0.4, rng)

    assert np.count_nonzero(ruidoso == -1) == 4
    assert np.array_equal(base, np.ones(10, dtype=int))


def test_aplicar_ruido_extremos(modulo: ModuleType) -> None:
    """Ruido 0 no cambia nada y ruido 1 invierte todo."""
    base = np.ones(10, dtype=int)

    sin_ruido = modulo.aplicar_ruido(base, 0.0, np.random.default_rng(0))
    todo_ruido = modulo.aplicar_ruido(base, 1.0, np.random.default_rng(0))

    assert np.array_equal(sin_ruido, base)
    assert np.all(todo_ruido == -1)


def test_aplicar_ruido_reproducible(modulo: ModuleType) -> None:
    """La misma semilla debe producir el mismo patrón ruidoso."""
    base = np.ones(20, dtype=int)

    primero = modulo.aplicar_ruido(base, 0.5, np.random.default_rng(7))
    segundo = modulo.aplicar_ruido(base, 0.5, np.random.default_rng(7))

    assert np.array_equal(primero, segundo)


@pytest.mark.parametrize("ruido", [-0.1, 1.1])
def test_aplicar_ruido_invalido(modulo: ModuleType, ruido: float) -> None:
    """El nivel de ruido debe estar entre 0 y 1."""
    with pytest.raises(ValueError, match="ruido"):
        modulo.aplicar_ruido(np.ones(4, dtype=int), ruido, np.random.default_rng(0))


def test_clasificar_consulta(modulo: ModuleType, motor: ModuleType) -> None:
    """La clasificación debe cubrir almacenado, espurio, ciclo y no convergencia."""
    red = motor.RedHopfield(
        pesos=np.zeros((2, 2)),
        patrones=np.array([[1, 1], [1, -1]], dtype=int),
        nombres=("a", "b"),
        filas=1,
        columnas=2,
    )

    correcto = modulo.clasificar_consulta(
        _resultado(motor, [1, 1]),
        red,
        "a",
    )
    incorrecto = modulo.clasificar_consulta(
        _resultado(motor, [1, -1]),
        red,
        "a",
    )
    espurio = modulo.clasificar_consulta(
        _resultado(motor, [-1, -1]),
        red,
        "a",
    )
    ciclo = modulo.clasificar_consulta(
        _resultado(motor, [1, 1], convergio=False, ciclo=True),
        red,
        "a",
    )
    no_converge = modulo.clasificar_consulta(
        _resultado(motor, [1, 1], convergio=False),
        red,
        "a",
    )

    assert correcto.tipo == "almacenado_correcto"
    assert correcto.nombre == "a"
    assert incorrecto.tipo == "almacenado_incorrecto"
    assert incorrecto.nombre == "b"
    assert espurio.tipo == "espurio"
    assert espurio.distancia_minima == 1
    assert ciclo.tipo == "ciclo"
    assert no_converge.tipo == "no_converge"


def test_cargar_config_red(
    modulo: ModuleType,
    red_sintetica: tuple[Path, Path],
) -> None:
    """La configuración de la red debe cargarse desde config_usada.toml."""
    red_dir, _ = red_sintetica

    config = modulo.cargar_config_red(red_dir)

    assert config["modelo"]["modo"] == "asincronica"
    assert config["modelo"]["orden"] == "secuencial"


def test_main_consulta_ruidosa(
    modulo: ModuleType,
    red_sintetica: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El runner debe generar las tablas y el gráfico de las consultas."""
    red_dir, ruta_datos = red_sintetica
    salida = tmp_path / "salida"

    codigo = modulo.main(
        [
            "--red",
            str(red_dir),
            "--datos",
            str(ruta_datos),
            "--ruido",
            "0.25",
            "--semilla",
            "0",
            "--repeticiones",
            "2",
            "--salida",
            str(salida),
            "--graphic",
        ]
    )
    stdout = capsys.readouterr().out

    assert codigo == 0
    for nombre in (
        "consultas.csv",
        "energia.csv",
        "estados_iniciales.csv",
        "estados_finales.csv",
        "resumen.txt",
    ):
        assert (salida / nombre).exists(), nombre

    consultas = pd.read_csv(salida / "consultas.csv")
    assert len(consultas) == 2
    assert set(consultas["tipo_resultado"]) == {"almacenado_correcto"}
    assert {
        "consulta",
        "nombre_origen",
        "repeticion",
        "ruido",
        "semilla",
        "tipo_resultado",
        "nombre_final",
        "distancia_hamming",
        "barridos",
        "energia_inicial",
        "energia_final",
    }.issubset(consultas.columns)

    estados_iniciales = pd.read_csv(salida / "estados_iniciales.csv")
    estados_finales = pd.read_csv(salida / "estados_finales.csv")
    energia = pd.read_csv(salida / "energia.csv")

    assert len(estados_iniciales) == 2
    assert len(estados_finales) == 2
    assert len(energia) >= 2
    columnas_pixeles = [
        columna for columna in estados_finales if columna.startswith("p")
    ]
    assert len(columnas_pixeles) == 4
    assert np.all(estados_finales[columnas_pixeles].to_numpy() == 1)
    assert (salida / "resumen.txt").read_text(encoding="utf-8").strip()
    assert "consulta:" in stdout
    assert "**" in stdout


def test_main_sin_datos_usa_patrones(
    modulo: ModuleType,
    red_sintetica: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Si no se pasa --datos, se consulta el patrones.csv de la red."""
    red_dir, _ = red_sintetica
    salida = tmp_path / "salida"

    codigo = modulo.main(
        ["--red", str(red_dir), "--salida", str(salida), "--ruido", "0"]
    )

    assert codigo == 0
    consultas = pd.read_csv(salida / "consultas.csv")
    assert len(consultas) == 1
    assert consultas.loc[0, "nombre_origen"] == "A"


def test_main_salida_por_defecto(
    modulo: ModuleType,
    red_sintetica: tuple[Path, Path],
) -> None:
    """Sin --salida, las consultas se guardan en <red>/consultas."""
    red_dir, _ = red_sintetica

    codigo = modulo.main(["--red", str(red_dir), "--ruido", "0"])

    assert codigo == 0
    assert (red_dir / "consultas" / "consultas.csv").exists()


def test_main_dimensiones_incompatibles(
    modulo: ModuleType,
    motor: ModuleType,
    red_sintetica: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Una consulta con otra forma debe rechazarse."""
    red_dir, _ = red_sintetica
    consulta = motor.Dataset(
        nombres=("B",),
        patrones=np.ones((1, 9), dtype=np.int8),
        filas=3,
        columnas=3,
    )
    ruta_consulta = tmp_path / "consulta.csv"
    motor.escribir_dataset(consulta, ruta_consulta)

    with pytest.raises(ValueError, match="dimensiones"):
        modulo.main(
            [
                "--red",
                str(red_dir),
                "--datos",
                str(ruta_consulta),
                "--salida",
                str(tmp_path / "salida"),
            ]
        )


@pytest.mark.parametrize(
    ("argumento", "valor"),
    [
        ("--ruido", "1.5"),
        ("--repeticiones", "0"),
        ("--semilla", "-1"),
    ],
)
def test_main_argumentos_invalidos(
    modulo: ModuleType,
    red_sintetica: tuple[Path, Path],
    argumento: str,
    valor: str,
) -> None:
    """Los argumentos fuera de rango deben rechazarse."""
    red_dir, _ = red_sintetica

    with pytest.raises(SystemExit):
        modulo.main(["--red", str(red_dir), argumento, valor])


def test_main_red_inexistente(modulo: ModuleType, tmp_path: Path) -> None:
    """Una red inexistente debe producir un error claro."""
    with pytest.raises(FileNotFoundError):
        modulo.main(["--red", str(tmp_path / "no_existe")])


def test_aplicar_esquina(modulo: ModuleType) -> None:
    """Borra en blanco un bloque cuadrado pegado a una esquina."""
    patron = np.ones(100, dtype=np.int8)
    rng = np.random.default_rng(0)
    resultado = modulo.aplicar_esquina(patron, 0.25, 10, 10, rng).reshape(10, 10)

    blancos = np.argwhere(resultado == -1)
    assert len(blancos) == 25
    filas, columnas = blancos[:, 0], blancos[:, 1]
    assert filas.max() - filas.min() == 4
    assert columnas.max() - columnas.min() == 4
    assert filas.min() in (0, 5) and columnas.min() in (0, 5)


def test_aplicar_esquina_nivel_cero(modulo: ModuleType) -> None:
    """Sin nivel no se modifica nada."""
    patron = np.ones(16, dtype=np.int8)
    rng = np.random.default_rng(0)
    assert np.array_equal(modulo.aplicar_esquina(patron, 0.0, 4, 4, rng), patron)


def test_aplicar_manchas(modulo: ModuleType) -> None:
    """Invierte exactamente round(nivel·N) píxeles agrupados en bloques."""
    patron = np.ones(37 * 37, dtype=np.int8)
    rng = np.random.default_rng(1)
    resultado = modulo.aplicar_manchas(patron, 0.1, 37, 37, rng)

    invertidos = resultado == -1
    assert invertidos.sum() == round(0.1 * 37 * 37)
    # Las manchas son contiguas: casi todo píxel invertido tiene un vecino
    # invertido, cosa que con ruido aleatorio al 10% casi nunca pasa.
    matriz = invertidos.reshape(37, 37)
    relleno = np.pad(matriz, 1)
    vecinos = (
        relleno[:-2, 1:-1] | relleno[2:, 1:-1] | relleno[1:-1, :-2] | relleno[1:-1, 2:]
    )
    assert (vecinos & matriz).sum() / matriz.sum() > 0.9


def test_aplicar_manchas_reproducible(modulo: ModuleType) -> None:
    """La misma semilla da las mismas manchas."""
    patron = np.ones(100, dtype=np.int8)
    a = modulo.aplicar_manchas(patron, 0.3, 10, 10, np.random.default_rng(5))
    b = modulo.aplicar_manchas(patron, 0.3, 10, 10, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_main_deterioro_y_energia_detallada(
    modulo: ModuleType,
    red_sintetica: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """El runner registra el deterioro, la distancia inicial y la energía fina."""
    red_dir, _ = red_sintetica
    salida = tmp_path / "salida"

    codigo = modulo.main(
        [
            "--red",
            str(red_dir),
            "--deterioro",
            "esquina",
            "--nivel",
            "0.25",
            "--energia-detallada",
            "--salida",
            str(salida),
        ]
    )

    assert codigo == 0
    consultas = pd.read_csv(salida / "consultas.csv")
    assert consultas["deterioro"].tolist() == ["esquina"]
    assert consultas["ruido"].tolist() == [0.25]
    assert consultas["hamming_inicial"].tolist() == [1]
    detalle = pd.read_csv(salida / "energia_actualizaciones.csv")
    assert list(detalle.columns) == ["consulta", "actualizacion", "energia"]
    assert len(detalle) == 1 + consultas["actualizaciones"].item()


COLUMNAS_VIEJAS = [
    "consulta",
    "nombre_origen",
    "indice_patron",
    "repeticion",
    "deterioro",
    "ruido",
    "semilla",
    "hamming_inicial",
    "tipo_resultado",
    "nombre_final",
    "distancia_hamming",
    "nombre_mas_cercano",
    "barridos",
    "actualizaciones",
    "energia_inicial",
    "energia_final",
    "convergio",
    "ciclo",
    "motivo",
]
COLUMNAS_NUEVAS = [
    "hamming_final_origen",
    "hamming_final_normalizado",
    "fraccion_afectada",
    "cambios_totales",
    "delta_energia",
    "delta_energia_por_neurona",
    "monotona",
]


def _resultado_con_historial(
    motor: ModuleType, energias: list[float], cambios: list[int]
) -> object:
    """Resultado sintético de 4 neuronas con la energía y los cambios dados."""
    historial = [
        motor.PasoEnergia(barrido=barrido, energia=energia, cambios=cambio)
        for barrido, (energia, cambio) in enumerate(zip(energias, cambios, strict=True))
    ]
    return motor.ResultadoRecuperacion(
        estado_inicial=np.array([1, -1, -1, 1]),
        estado_final=np.array([1, 1, -1, -1]),
        barridos=len(energias) - 1,
        actualizaciones=4 * (len(energias) - 1),
        energia_inicial=energias[0],
        energia_final=energias[-1],
        historial=historial,
        convergio=True,
        ciclo=False,
        motivo="estable",
    )


def test_metricas_consulta(modulo: ModuleType, motor: ModuleType) -> None:
    """Hamming respecto del origen, cambios y energía de una consulta."""
    resultado = _resultado_con_historial(motor, [0.0, -2.0, -3.0, -3.0], [0, 2, 1, 0])
    patron = np.array([1, 1, 1, 1])

    metricas = modulo.metricas_consulta(resultado, patron)

    assert list(metricas) == COLUMNAS_NUEVAS
    assert metricas["hamming_final_origen"] == 2
    assert metricas["hamming_final_normalizado"] == pytest.approx(0.5)
    assert metricas["fraccion_afectada"] == pytest.approx(0.5)
    assert metricas["cambios_totales"] == 3
    assert metricas["delta_energia"] == pytest.approx(-3.0)
    assert metricas["delta_energia_por_neurona"] == pytest.approx(-0.75)
    assert metricas["monotona"] is True


def test_metricas_consulta_no_monotona(modulo: ModuleType, motor: ModuleType) -> None:
    """Una energía que sube entre barridos no es monotónica."""
    resultado = _resultado_con_historial(motor, [0.0, -1.0, 0.5], [0, 1, 1])
    assert modulo.metricas_consulta(resultado, np.ones(4))["monotona"] is False


def test_resumir_metricas(modulo: ModuleType) -> None:
    """Tasas por tipo de resultado y medias y desvíos de las métricas."""
    consultas = pd.DataFrame(
        {
            "tipo_resultado": [
                "almacenado_correcto",
                "almacenado_correcto",
                "espurio",
                "almacenado_incorrecto",
            ],
            "barridos": [1, 2, 3, 2],
            "actualizaciones": [4, 8, 12, 8],
            "cambios_totales": [0, 1, 5, 2],
            "delta_energia_por_neurona": [0.0, -0.5, -1.0, -0.5],
            "hamming_final_normalizado": [0.0, 0.0, 0.5, 0.25],
            "monotona": [True, True, True, False],
        }
    )

    metricas = modulo.resumir_metricas(consultas)

    assert metricas["consultas"] == 4
    assert metricas["exactitud"] == pytest.approx(0.5)
    assert metricas["tasa_incorrecto"] == pytest.approx(0.25)
    assert metricas["tasa_espurio"] == pytest.approx(0.25)
    assert metricas["tasa_ciclo"] == 0.0
    assert metricas["tasa_no_converge"] == 0.0
    assert metricas["barridos_media"] == pytest.approx(2.0)
    assert metricas["barridos_desvio"] == pytest.approx(np.std([1, 2, 3, 2], ddof=1))
    assert metricas["cambios_totales_media"] == pytest.approx(2.0)
    assert metricas["hamming_final_normalizado_media"] == pytest.approx(0.1875)
    assert metricas["proporcion_monotona"] == pytest.approx(0.75)


def test_main_metricas_letras(modulo: ModuleType, tmp_path: Path) -> None:
    """Las columnas nuevas se agregan al final y son coherentes con energia.csv."""
    salida = tmp_path / "salida"
    argumentos = ["--red", str(RAIZ / "resultados" / "ej2" / "hopfield" / "letras_5x5")]
    argumentos += ["--ruido", "0.3", "--semilla", "0", "--repeticiones", "5"]
    assert modulo.main(argumentos + ["--salida", str(salida)]) == 0

    consultas = pd.read_csv(salida / "consultas.csv", keep_default_na=False)
    assert list(consultas.columns) == COLUMNAS_VIEJAS + COLUMNAS_NUEVAS
    assert consultas["monotona"].all()
    assert (consultas["fraccion_afectada"] == consultas["hamming_inicial"] / 25).all()

    energia = pd.read_csv(salida / "energia.csv").groupby("consulta", sort=False)
    por_consulta = consultas.set_index("consulta")
    np.testing.assert_array_equal(
        energia["cambios"].sum()[por_consulta.index], por_consulta["cambios_totales"]
    )
    delta = energia["energia"].last() - energia["energia"].first()
    np.testing.assert_allclose(delta[por_consulta.index], por_consulta["delta_energia"])
    correctos = por_consulta["tipo_resultado"] == "almacenado_correcto"
    assert (por_consulta.loc[correctos, "hamming_final_origen"] == 0).all()

    metricas = pd.read_csv(salida / "metricas.csv")
    assert len(metricas) == 1
    assert metricas.loc[0, "consultas"] == 20
    assert metricas.loc[0, "exactitud"] == pytest.approx(correctos.mean())
    assert "Métricas agregadas:" in (salida / "resumen.txt").read_text(encoding="utf-8")


RUTA_LETRAS_5X5 = RAIZ / "resultados" / "ej2" / "hopfield" / "letras_5x5"
CORRIDAS_2_2 = [("limpias", 0.0, 0, 1)] + [
    (f"ruido_{nivel}_semilla_{semilla}", nivel, semilla, 5)
    for nivel in (0.1, 0.2, 0.3, 0.4, 0.5)
    for semilla in (0, 1, 2)
]


@pytest.mark.parametrize(("carpeta", "ruido", "semilla", "repeticiones"), CORRIDAS_2_2)
def test_regresion_letras_5x5(
    modulo: ModuleType,
    tmp_path: Path,
    carpeta: str,
    ruido: float,
    semilla: int,
    repeticiones: int,
) -> None:
    """El runner reproduce las consultas guardadas de 2.2 sin escribir en resultados/."""
    guardada = RUTA_LETRAS_5X5 / "consultas" / carpeta
    salida = tmp_path / carpeta
    argumentos = ["--red", str(RUTA_LETRAS_5X5), "--salida", str(salida)]
    argumentos += ["--ruido", str(ruido), "--semilla", str(semilla)]
    argumentos += ["--repeticiones", str(repeticiones)]
    assert modulo.main(argumentos) == 0

    esperada = pd.read_csv(guardada / "consultas.csv", keep_default_na=False)
    obtenida = pd.read_csv(salida / "consultas.csv", keep_default_na=False)
    for columna in esperada.columns:
        if esperada[columna].dtype.kind == "f":
            np.testing.assert_allclose(obtenida[columna], esperada[columna], atol=1e-9)
        else:
            assert obtenida[columna].tolist() == esperada[columna].tolist(), columna
    pd.testing.assert_frame_equal(
        pd.read_csv(salida / "estados_finales.csv"),
        pd.read_csv(guardada / "estados_finales.csv"),
    )
