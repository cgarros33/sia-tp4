"""Tests del ejercicio 2.1: barrido de ruido sobre redes de Hopfield entrenadas."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]
RUTA_SCRIPT = RAIZ / "src" / "ej2" / "hopfield-noise.py"
RUTA_MOTOR = RAIZ / "src" / "ej2" / "hopfield.py"
RUTA_RUNNER = RAIZ / "src" / "ej2" / "hopfield-runner.py"
REGLAS = ("hebb", "pseudoinversa")


def _cargar(ruta: Path, nombre: str) -> ModuleType:
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
def modulo() -> ModuleType:
    """Carga el orquestador del barrido de ruido."""
    return _cargar(RUTA_SCRIPT, "hopfield_noise")


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    """Carga el runner, para comparar contra sus consultas."""
    return _cargar(RUTA_RUNNER, "hopfield_runner_test_noise")


def _escribir_config(ruta: Path, redes: dict[str, Path], salida: Path, **extra) -> Path:
    """Escribe un TOML de barrido chico para las pruebas."""
    lineas = ["[redes]"]
    lineas += [f"{regla} = {json.dumps(str(red))}" for regla, red in redes.items()]
    lineas += [
        "",
        "[barrido]",
        f"semillas = {extra.get('semillas', [0, 1])}",
        f"repeticiones = {extra.get('repeticiones', 2)}",
        'patron_ejemplo = "b"',
        f"salida = {json.dumps(str(salida))}",
        "",
        "[niveles]",
        "aleatorio = [0.1, 0.3]",
        "esquina = [2, 3]",
        "finder = [1, 3]",
        "rayones = [1, 2]",
        "manchas = [[1, 1], [2, 1]]",
        "bloques = [0.25, 0.5]",
        "",
        "[combinaciones]",
        'componentes = ["aleatorio", "esquina", "finder", "rayones", "manchas",'
        ' "bloques"]',
        "aleatorio = 0.1",
        "esquina = 2",
        "finder = 1",
        "rayones = 1",
        "manchas = [1, 1]",
        "bloques = 0.25",
    ]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


@pytest.fixture(scope="module")
def entorno(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Dos redes chicas (Hebb y pseudo-inversa) de tres patrones correlacionados."""
    motor = _cargar(RUTA_MOTOR, "hopfield_motor_test_noise")
    base = tmp_path_factory.mktemp("noise")
    generador = np.random.default_rng(0)
    patron = generador.choice([-1, 1], size=144)
    patrones = np.tile(patron, (3, 1)).astype(np.int8)
    for indice in range(3):
        patrones[indice, indice * 12 : indice * 12 + 12] *= -1
    dataset = motor.Dataset(
        nombres=("a", "b", "c"), patrones=patrones, filas=12, columnas=12
    )
    ruta_datos = base / "datos.csv"
    motor.escribir_dataset(dataset, ruta_datos)
    redes = {}
    for regla in REGLAS:
        redes[regla] = base / regla
        argumentos = ["--datos", str(ruta_datos), "--regla", regla]
        assert motor.main(argumentos + ["--salida", str(redes[regla])]) == 0
    salida = base / "ruido"
    config = _escribir_config(base / "noise.toml", redes, salida)
    return {"config": config, "salida": salida, "base": base, **redes}


@pytest.fixture(scope="module")
def corrida_aleatorio(modulo: ModuleType, entorno: dict[str, Path]) -> Path:
    """Corre el modelo aleatorio una vez para varias pruebas."""
    assert (
        modulo.main(["--config", str(entorno["config"]), "--modelo", "aleatorio"]) == 0
    )
    return entorno["salida"] / "aleatorio"


def test_salidas_del_modelo(corrida_aleatorio: Path) -> None:
    """La corrida de un modelo genera todas las tablas esperadas."""
    for nombre in (
        "config_usada.toml",
        "consultas.csv",
        "energia.csv",
        "metricas_por_semilla.csv",
        "resumen_niveles.csv",
        "comparacion_reglas.csv",
        "resumen_por_patron.csv",
        "resumen_energia.csv",
        "ejemplos.npz",
        "resumen.txt",
    ):
        assert (corrida_aleatorio / nombre).exists(), nombre
    # Ninguna salida guarda estados completos de todas las consultas.
    assert not list(corrida_aleatorio.glob("estados_*.csv"))


def test_consultas_por_regla(corrida_aleatorio: Path) -> None:
    """Cada consulta se genera una vez y se recupera con las dos reglas."""
    consultas = pd.read_csv(corrida_aleatorio / "consultas.csv")
    # 2 niveles × 2 semillas × 3 patrones × 2 repeticiones × 2 reglas.
    assert len(consultas) == 48
    assert set(consultas["regla"]) == set(REGLAS)
    assert set(consultas["modelo"]) == {"aleatorio"}
    claves = ["nivel", "semilla", "consulta"]
    por_regla = consultas.pivot(index=claves, columns="regla", values="hamming_inicial")
    assert (por_regla["hebb"] == por_regla["pseudoinversa"]).all()
    assert (consultas["fraccion_afectada"] == consultas["fraccion_nominal"]).all()
    esperado = {0.1: round(0.1 * 144), 0.3: round(0.3 * 144)}
    assert (consultas["hamming_inicial"] == consultas["nivel"].map(esperado)).all()


def test_aleatorio_reproduce_el_runner(
    runner: ModuleType, entorno: dict[str, Path], corrida_aleatorio: Path
) -> None:
    """El modelo aleatorio genera exactamente las consultas del runner."""
    consultas = pd.read_csv(corrida_aleatorio / "consultas.csv")
    salida = entorno["base"] / "runner_0.3_1"
    argumentos = ["--red", str(entorno["hebb"]), "--ruido", "0.3", "--semilla", "1"]
    assert (
        runner.main(argumentos + ["--repeticiones", "2", "--salida", str(salida)]) == 0
    )
    esperadas = pd.read_csv(salida / "consultas.csv").set_index("consulta")
    obtenidas = consultas[
        (consultas["nivel"] == 0.3)
        & (consultas["semilla"] == 1)
        & (consultas["regla"] == "hebb")
    ].set_index("consulta")
    for columna in ("hamming_inicial", "tipo_resultado", "barridos", "cambios_totales"):
        assert (
            obtenidas[columna].tolist()
            == esperadas.loc[obtenidas.index, columna].tolist()
        )


def test_reproducible(
    modulo: ModuleType, entorno: dict[str, Path], corrida_aleatorio: Path
) -> None:
    """La misma configuración produce los mismos CSV."""
    otra = entorno["base"] / "ruido_repetido"
    argumentos = ["--config", str(entorno["config"]), "--modelo", "aleatorio"]
    assert modulo.main(argumentos + ["--salida", str(otra)]) == 0
    for nombre in ("consultas.csv", "energia.csv", "resumen_niveles.csv"):
        assert (otra / "aleatorio" / nombre).read_bytes() == (
            corrida_aleatorio / nombre
        ).read_bytes()


def test_resumen_por_semillas(corrida_aleatorio: Path) -> None:
    """Las tasas se calculan por semilla y se resumen con media y desvío."""
    consultas = pd.read_csv(corrida_aleatorio / "consultas.csv")
    por_semilla = pd.read_csv(corrida_aleatorio / "metricas_por_semilla.csv")
    resumen = pd.read_csv(corrida_aleatorio / "resumen_niveles.csv")
    assert len(por_semilla) == 2 * 2 * 2
    assert len(resumen) == 2 * 2

    consultas["correcto"] = consultas["tipo_resultado"] == "almacenado_correcto"
    exactitud = consultas.groupby(["nivel", "regla", "semilla"])["correcto"].mean()
    esperado = exactitud.groupby(["nivel", "regla"]).agg(["mean", "std"])
    obtenido = resumen.set_index(["nivel", "regla"])
    np.testing.assert_allclose(
        obtenido.loc[esperado.index, "exactitud_media"], esperado["mean"]
    )
    np.testing.assert_allclose(
        obtenido.loc[esperado.index, "exactitud_desvio"], esperado["std"]
    )
    assert (obtenido["semillas"] == 2).all()

    comparacion = pd.read_csv(corrida_aleatorio / "comparacion_reglas.csv")
    diferencia = (
        comparacion["exactitud_media_pseudoinversa"]
        - comparacion["exactitud_media_hebb"]
    )
    np.testing.assert_allclose(comparacion["exactitud_diferencia"], diferencia)

    por_patron = pd.read_csv(corrida_aleatorio / "resumen_por_patron.csv")
    assert len(por_patron) == 2 * 2 * 3


def test_resumen_energia_mantiene_la_energia_final() -> None:
    """Una consulta estable conserva su energía en los barridos siguientes."""
    modulo = _cargar(RUTA_SCRIPT, "hopfield_noise_energia")
    energias = pd.DataFrame(
        {
            "modelo": "m",
            "orden_nivel": 0,
            "nivel": "1",
            "regla": "hebb",
            "semilla": 0,
            "consulta": ["a", "a", "b", "b", "b"],
            "barrido": [0, 1, 0, 1, 2],
            "energia": [-1.0, -3.0, -2.0, -4.0, -6.0],
            "cambios": [0, 1, 0, 2, 1],
        }
    )
    resumen = modulo.resumir_energia(energias, n_neuronas=2)
    assert resumen["barrido"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(resumen["energia_media"], [-1.5, -3.5, -4.5])
    np.testing.assert_allclose(
        resumen["energia_por_neurona_media"], [-0.75, -1.75, -2.25]
    )
    assert resumen["consultas_activas"].tolist() == [2, 2, 1]


def test_ejemplos(corrida_aleatorio: Path) -> None:
    """Los ejemplos guardan la consulta de ejemplo de cada nivel."""
    consultas = pd.read_csv(corrida_aleatorio / "consultas.csv")
    with np.load(corrida_aleatorio / "ejemplos.npz", allow_pickle=False) as ejemplos:
        assert ejemplos["niveles"].tolist() == ["0.1", "0.3"]
        assert str(ejemplos["patron"]) == "b"
        assert ejemplos["iniciales"].shape == (2, 144)
        for regla in REGLAS:
            assert ejemplos[f"finales_{regla}"].shape == (2, 144)
            for orden in range(2):
                energia = ejemplos[f"energia_{regla}_{orden}"]
                fila = consultas[
                    (consultas["orden_nivel"] == orden)
                    & (consultas["semilla"] == 0)
                    & (consultas["consulta"] == "b_rep0")
                    & (consultas["regla"] == regla)
                ].iloc[0]
                assert len(energia) == 1 + fila["actualizaciones"]
                assert energia[0] == pytest.approx(fila["energia_inicial"], rel=1e-6)
                assert energia[-1] == pytest.approx(fila["energia_final"], rel=1e-6)
        diferencias = (ejemplos["iniciales"] != ejemplos["original"]).sum(axis=1)
    assert diferencias.tolist() == [round(0.1 * 144), round(0.3 * 144)]


def test_consolidar(modulo: ModuleType, entorno: dict[str, Path]) -> None:
    """La consolidación une los modelos sin duplicar filas."""
    config = entorno["config"]
    salida = entorno["base"] / "ruido_consolidado"
    for modelo in ("aleatorio", "aleatorio"):
        argumentos = ["--config", str(config), "--modelo", modelo]
        assert modulo.main(argumentos + ["--salida", str(salida)]) == 0
    assert (
        modulo.main(["--config", str(config), "--consolidar", "--salida", str(salida)])
        == 0
    )

    consolidado = salida / "consolidado"
    resumen = pd.read_csv(consolidado / "resumen_niveles.csv")
    individual = pd.read_csv(salida / "aleatorio" / "resumen_niveles.csv")
    assert len(resumen) == len(individual)
    consultas = pd.read_csv(consolidado / "consultas.csv")
    assert len(consultas) == 48
    intervalos = pd.read_csv(consolidado / "exactitud_por_intervalo.csv")
    assert set(intervalos["regla"]) == set(REGLAS)
    assert intervalos["consultas"].sum() == 48


def test_no_escribe_en_resultados(corrida_aleatorio: Path) -> None:
    """Las pruebas escriben solo en el directorio temporal."""
    assert RAIZ / "resultados" not in corrida_aleatorio.parents


def test_argumentos_invalidos(modulo: ModuleType, entorno: dict[str, Path]) -> None:
    """Un paso fuera de la escalera o --paso sin --combinaciones fallan."""
    config = str(entorno["config"])
    with pytest.raises(SystemExit):
        modulo.main(["--config", config, "--modelo", "aleatorio", "--paso", "2"])
    with pytest.raises(SystemExit):
        modulo.main(["--config", config, "--combinaciones", "--paso", "7"])
    with pytest.raises(SystemExit):
        modulo.main(["--config", config])


def test_nivel_combinacion(modulo: ModuleType, entorno: dict[str, Path]) -> None:
    """El paso k activa los primeros k componentes con su nivel fijo."""
    config = modulo.cargar_config(entorno["config"])
    paso = modulo.nivel_combinacion(config, 3)
    assert paso.etiqueta == "paso_3"
    assert paso.orden == 2
    assert paso.componentes == (("aleatorio", 0.1), ("esquina", 2), ("finder", 1))


def test_consulta_combinada_es_la_composicion(modulo: ModuleType) -> None:
    """Primero se invierten bits y después se borra; la zona es la unión."""
    patron = np.random.default_rng(4).choice([-1, 1], size=144).astype(np.int8)
    componentes = (("aleatorio", 0.2), ("esquina", 5), ("manchas", [1, 2]))
    flujos = modulo.crear_flujos(componentes, semilla=3)
    estado, parametros, area = modulo.generar_consulta(
        patron, componentes, flujos, 12, 12
    )

    assert list(parametros) == ["aleatorio", "esquina", "manchas"]
    invertido = modulo.runner.aplicar_ruido(patron, 0.2, np.random.default_rng(3))
    esquina, datos_esquina = modulo.ruido.mascara_esquina(
        5, 12, 12, np.random.default_rng([3, modulo.ruido.CODIGOS["esquina"]])
    )
    manchas, _ = modulo.ruido.mascara_manchas(
        [1, 2], 12, 12, np.random.default_rng([3, modulo.ruido.CODIGOS["manchas"]])
    )
    esperado = modulo.ruido.borrar(modulo.ruido.borrar(invertido, esquina), manchas)
    assert np.array_equal(estado, esperado)
    assert parametros["esquina"] == datos_esquina
    # El borrado tiene prioridad sobre el bit invertido.
    borrados = (esquina | manchas).reshape(-1)
    assert (estado[borrados] == -1).all()
    zona = (invertido != patron) | borrados
    assert area == int(zona.sum())


def test_escalera(modulo: ModuleType, entorno: dict[str, Path]) -> None:
    """Cada paso guarda su carpeta y sus componentes coinciden con los modelos."""
    config = str(entorno["config"])
    salida = entorno["base"] / "ruido_escalera"
    argumentos = ["--config", config, "--salida", str(salida)]
    assert modulo.main([*argumentos, "--combinaciones"]) == 0
    assert modulo.main([*argumentos, "--modelo", "esquina"]) == 0
    for paso in range(1, 7):
        consultas = pd.read_csv(
            salida / "combinaciones" / f"paso_{paso}" / "consultas.csv"
        )
        assert set(consultas["nivel"]) == {f"paso_{paso}"}
        assert set(consultas["orden_nivel"]) == {paso - 1}
        claves = {tuple(json.loads(p)) for p in consultas["parametros"]}
        esperado = ("aleatorio", "esquina", "finder", "rayones", "manchas", "bloques")
        assert claves == {esperado[:paso]}

    # Agregar componentes no cambia los anteriores: la esquina del paso 6 es la
    # misma que la del modelo individual en su nivel.
    escalera = pd.read_csv(salida / "combinaciones" / "paso_6" / "consultas.csv")
    individual = pd.read_csv(salida / "esquina" / "consultas.csv")
    individual = individual[individual["nivel"] == 2]
    claves = ["semilla", "consulta", "regla"]
    escalera = escalera.set_index(claves)["parametros"].map(
        lambda p: json.loads(p)["esquina"]
    )
    individual = individual.set_index(claves)["parametros"].map(
        lambda p: json.loads(p)["esquina"]
    )
    assert escalera.loc[individual.index].tolist() == individual.tolist()

    assert modulo.main([*argumentos, "--consolidar"]) == 0
    consolidado = pd.read_csv(
        salida / "combinaciones" / "consolidado" / "resumen_niveles.csv"
    )
    assert consolidado["nivel"].tolist() == [
        f"paso_{paso}" for paso in range(1, 7) for _ in REGLAS
    ]
    # La fracción dañada crece con cada componente.
    zona = consolidado.groupby("orden_nivel")["fraccion_nominal_media"].first()
    assert zona.is_monotonic_increasing


def test_margen_inicial(corrida_aleatorio: Path) -> None:
    """El margen inicial compara la distancia al origen con la del otro QR más cercano."""
    consultas = pd.read_csv(corrida_aleatorio / "consultas.csv")
    assert {"nombre_inicial_mas_cercano", "margen_inicial"} <= set(consultas.columns)
    # Con poco ruido la consulta sigue más cerca de su origen.
    bajas = consultas[consultas["nivel"] == 0.1]
    assert (bajas["margen_inicial"] > 0).all()
    assert (bajas["nombre_inicial_mas_cercano"] == bajas["nombre_origen"]).all()
    # El margen es una propiedad de la consulta: es el mismo para las dos reglas.
    por_regla = consultas.pivot(
        index=["nivel", "semilla", "consulta"], columns="regla", values="margen_inicial"
    )
    assert (por_regla["hebb"] == por_regla["pseudoinversa"]).all()


def test_config_sin_combinaciones(modulo: ModuleType, entorno: dict[str, Path]) -> None:
    """Un barrido sin [combinaciones] corre sus modelos y rechaza los que no tiene."""
    texto = entorno["config"].read_text(encoding="utf-8")
    sin_combinaciones = texto.split("[combinaciones]")[0]
    sin_combinaciones = sin_combinaciones.replace("aleatorio = [0.1, 0.3]\n", "")
    config = entorno["base"] / "noise_exploratorio.toml"
    config.write_text(sin_combinaciones, encoding="utf-8")
    salida = entorno["base"] / "ruido_exploratorio"
    argumentos = ["--config", str(config), "--salida", str(salida)]

    assert modulo.main([*argumentos, "--modelo", "esquina"]) == 0
    assert (salida / "esquina" / "consultas.csv").exists()
    with pytest.raises(SystemExit):
        modulo.main([*argumentos, "--modelo", "aleatorio"])
    with pytest.raises(SystemExit):
        modulo.main([*argumentos, "--combinaciones"])


def test_legibilidad_no_modifica_el_barrido(
    modulo: ModuleType, entorno: dict[str, Path], corrida_aleatorio: Path
) -> None:
    """--legibilidad solo agrega sus tablas y no toca las del barrido."""
    antes = {p.name: p.read_bytes() for p in corrida_aleatorio.iterdir() if p.is_file()}
    argumentos = ["--config", str(entorno["config"]), "--modelo", "aleatorio"]
    assert modulo.main([*argumentos, "--legibilidad"]) == 0
    for nombre, contenido in antes.items():
        assert (corrida_aleatorio / nombre).read_bytes() == contenido, nombre

    tabla = pd.read_csv(corrida_aleatorio / "legibilidad.csv", keep_default_na=False)
    consultas = pd.read_csv(corrida_aleatorio / "consultas.csv")
    assert len(tabla) == len(consultas)
    assert {
        "lectura_inicial",
        "lectura_final",
        "legible_inicial",
        "legible_final",
        "lee_origen_final",
    } <= set(tabla.columns)
    # Un patrón al azar de 12 × 12 no es un QR: nada se lee.
    assert not tabla["legible_inicial"].any()
    assert not tabla["legible_final"].any()
    resumen = pd.read_csv(corrida_aleatorio / "legibilidad_resumen.csv")
    assert len(resumen) == 2 * 2
    assert (entorno["salida"] / "legibilidad_patrones.csv").exists()


def test_legibilidad_inconsistente(
    modulo: ModuleType, entorno: dict[str, Path], corrida_aleatorio: Path
) -> None:
    """Si las consultas regeneradas no coinciden con las guardadas, falla."""
    texto = entorno["config"].read_text(encoding="utf-8")
    otra = entorno["base"] / "noise_otra.toml"
    otra.write_text(
        texto.replace("repeticiones = 2", "repeticiones = 1"), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        modulo.main(["--config", str(otra), "--modelo", "aleatorio", "--legibilidad"])


def test_legibilidad_qr_reales(modulo: ModuleType, tmp_path: Path) -> None:
    """Con los QR reales: los originales se leen y la mezcla de Hebb no."""
    motor = _cargar(RUTA_MOTOR, "hopfield_motor_test_legibilidad")
    datos = RAIZ / "datos" / "hopfield" / "qr_37x37.csv"
    redes = {}
    for regla in REGLAS:
        redes[regla] = tmp_path / regla
        argumentos = ["--datos", str(datos), "--regla", regla]
        assert motor.main([*argumentos, "--salida", str(redes[regla])]) == 0
    config = _escribir_config(
        tmp_path / "noise.toml", redes, tmp_path / "ruido", semillas=[0], repeticiones=1
    )
    texto = config.read_text(encoding="utf-8")
    config.write_text(
        texto.replace(
            'patron_ejemplo = "b"', 'patron_ejemplo = "food_menu_qr"'
        ).replace("aleatorio = [0.1, 0.3]", "aleatorio = [0.05]"),
        encoding="utf-8",
    )
    argumentos = ["--config", str(config), "--modelo", "aleatorio"]
    assert modulo.main(argumentos) == 0
    assert modulo.main([*argumentos, "--legibilidad"]) == 0

    patrones = pd.read_csv(
        tmp_path / "ruido" / "legibilidad_patrones.csv", keep_default_na=False
    )
    originales = patrones[patrones["estado"] != "mezcla"]
    assert originales["legible"].all()
    assert originales["texto"].str.startswith("https://").all()
    assert not patrones.loc[patrones["estado"] == "mezcla", "legible"].item()

    tabla = pd.read_csv(
        tmp_path / "ruido" / "aleatorio" / "legibilidad.csv", keep_default_na=False
    )
    pseudo = tabla[tabla["regla"] == "pseudoinversa"]
    hebb = tabla[tabla["regla"] == "hebb"]
    assert pseudo["lee_origen_final"].all()
    assert not hebb["legible_final"].any()
