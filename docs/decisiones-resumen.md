# Decisiones del TP4 — resumen

> Versión corta de `docs/decisiones.md` (≈1400 líneas). Acá está qué se
> decidió, por qué y los números para defenderlo. Para el detalle, cada sección
> indica dónde mirar.

## Reglas generales (valen para todo el TP)

- **Sin librerías de ML.** Todo está hecho con NumPy, pandas y matplotlib.
  scikit-learn solo se usa en `tests/`, como referencia para verificar.
- **3 semillas `[0, 1, 2]`.** Todo resultado se reporta como media ± desvío.
- **Sin split de test.** Los dos ejercicios son no supervisados o de memoria
  asociativa: se usan los 50 creadores y los 5 QR completos. Es una excepción
  documentada a la regla de `CLAUDE.md`.
- **Los scripts de entrenamiento escriben y los de análisis solo leen.** Cada
  corrida guarda su `config_usada.toml`, así que todo se puede regenerar.

---

## 1.1 EDA y PCA

**Variables: de 31 numéricas a 27.** Se sacan 4 que son redundantes exactas:

| Variable descartada | Motivo |
|---|---|
| `price_ars` | `price_usd × 1500` |
| `estimated_argentina_reach` | `reach × argentina_pct / 100` |
| `total_engagement_per_1000` | suma de likes + comments + saves + shares |
| `organic_posts_pct` | `100 − sponsored_posts_pct` |

- Se conservan los 4 tópicos y los 4 tipos de contenido, aunque cada grupo suma
  100%, porque se entienden mejor así. Por eso la matriz de correlación tiene
  **rango 25**: hay 2 autovalores nulos, que se detectan y se ignoran.
- Se estandariza (z-score) y se usa la **matriz de correlación**, porque las
  escalas son muy distintas (followers vs. porcentajes).
- No se sacan outliers.
- Los autovectores se calculan con `numpy.linalg.eigh`. El signo de PC1 se fija
  para que `followers` tenga carga positiva.

**Cuántas componentes:**

| Umbral | Componentes | Varianza acumulada |
|---|---:|---:|
| 90% | 10 | 91,5% |
| 95% | 13 | 95,6% |
| 99% | 18 | 99,1% |

En la práctica se usa el **90% (10 PCs)** como entrada de Kohonen. Oja lee el
PCA del 95%, pero solo usa PC1, que es la misma con cualquier umbral.
`decisiones.md` todavía lo lista como pendiente; conviene dejarlo cerrado.

**Interpretación de las componentes:**

| PC | Varianza | Qué representa |
|---|---:|---|
| PC1 | 49,7% | Tamaño, alcance y profesionalización (correlación ≈ 0,87 con followers y con price) |
| PC2 | 9,2% | Arte/showcase (+) vs. audiencia argentina y perros (−) |
| PC3 | 7,0% | Publicaciones patrocinadas + fútbol |
| PC4 | 5,8% | Maquillaje (+) vs. fútbol (−) |

**Idea clave:** PC1 es casi "followers". Los nichos temáticos, que es lo que le
importa a cada cliente, aparecen recién en PC2–PC4.

Más detalle: `resultados/ej1/pca_0.90/resumen.txt`.

---

## 1.2 Kohonen

**Entrada:** PC1..PC10 de 1.1, **sin reestandarizar**, para que PC1 conserve
su peso relativo. Antes hubo una corrida con las 27 variables, que quedó
archivada en `ej1.2-full/` y sirve para comparar.

**Modelo:**

- SOM online: una muestra por iteración y épocas mezcladas.
- La neurona ganadora (BMU) se elige con distancia euclidiana o con
  "exponencial" (vectores normalizados, como en la teoría, págs. 121–124).
- Regla de actualización: `w ← w + η(t)(x − w)` para la BMU y sus vecinas
  dentro del radio `R(t)`.
- El radio decrece de `R0` a 1 y la tasa η decrece de `η0` a 0,001.
- Las métricas siempre se calculan con distancia euclidiana:
  - QE: distancia media de cada creador a su neurona.
  - TE: fracción de creadores cuyas dos neuronas más cercanas no son vecinas en
    la grilla.
  - Neuronas muertas y ocupación.
- Matriz U: distancia media de cada neurona a sus vecinas (4 en grilla
  rectangular, 6 en hexagonal).

**Cómo se eligieron los hiperparámetros:**

1. **El tamaño de la grilla no se barre.** Se fija k = 3, 4 y 5, y se hace un
   barrido completo para cada uno. Los primeros barridos (sweep1/2) dejaban
   variar k y siempre elegían el más grande: con más neuronas que creadores, el
   mapa tiende a "una neurona por creador" y la QE colapsa sin significar nada
   (`docs/sweep.md`).
2. **Barrido secuencial**, un parámetro por vez, en este orden: topología →
   distancia → inicialización → radio inicial → decaimiento del radio → η
   inicial → decaimiento de η → iteraciones. Criterio: **menor QE media**.
3. Después se prueban **solo dos pares acoplados** con producto cartesiano: (η,
   decaimiento de η) y (radio, decaimiento del radio). Se adoptan solo si bajan
   la QE.
4. **No se usa TE para desempatar.** Cuando hay empates de QE, el TE también
   empata, y en los casi empates el candidato con menor TE tiene peor QE
   (`docs/sweep.md`).
5. **Variar dos parámetros juntos casi no aporta.** En la corrida de 27
   variables con k=4, la mejora fue de 0,24% (p ≈ 0,40) y empeoró el TE y las
   neuronas muertas (`docs/alternativa.md`).

**Resultados finales (PC1..PC10):**

| k | QE | TE | Muertas | Ocupación | Configuración |
|---:|---:|---:|---:|---:|---|
| 3 | 3,110 ± 0,016 | 0,14 | 0/9 | 1,00 | rect, euclidiana, radio 0,4k, η 0,3 exponencial |
| 4 | 2,807 ± 0,022 | 0,14 | 0/16* | 1,00* | rect, **exponencial**, muestras, radio 1,5k exponencial, η 0,5 lineal |
| 4 (euclidiana) | 2,798 ± 0,011 | 0,07 | 0,67 | 0,96 | rect, euclidiana, uniforme, radio 0,3k constante, η 0,3 exponencial |
| 5 | 2,451 ± 0,022 | 0,04 | 3/25* | 0,88* | rect, euclidiana, radio 0,3k, η 0,3 lineal |

\* Valores de la semilla representativa. En promedio sobre las 3 semillas, k=4
tiene 0,67 neuronas muertas y ocupación 0,96, y k=5 tiene 4,33 y 0,83.

- **No hay una grilla final elegida por escrito.** Lo razonable es **k=4**: con
  k=5 aparecen neuronas muertas (50 creadores en 25 neuronas) y k=3 agrupa
  demasiado. Entre las dos variantes de k=4, la **euclidiana** es mejor en QE y
  en TE y además es más estable entre semillas.
- **La QE de PC10 no se compara directamente con la de 27 variables**, porque
  están en espacios distintos. Para compararlas, `kohonen-comparacion.py` lleva
  los pesos de vuelta a las 27 variables ("QE común"): con 27 variables y k=4 da
  3,17; con PC10 y k=4 da 3,19. Es casi lo mismo con menos dimensiones.
- **Estabilidad entre semillas** (ARI, índice de Rand ajustado, implementado a
  mano): ≈ 0,56–0,60 para k=4. El núcleo de cada grupo es estable y los
  creadores de borde cambian de neurona.
- **Alineación con el biplot:** un eje de la grilla sigue a PC1 (R² ≈ 0,87–0,96)
  y PC2 se alinea en parte (≈ 0,4–0,6).
- La interpretación de los perfiles de cada región **no está escrita en ningún
  lado**; la tiene que hacer el grupo. La herramienta para eso es
  `resultados/ej1/ej1.2-perfiles/<corrida>/fig_perfil_neuronas.png`.

Más detalle: `resultados/ej1/ej1.2-comparacion/resumen.txt`,
`resultados/ej1/ej1.2-perfiles/resumen.txt`, `docs/sweep.md`.

---

## 1.3 Oja

**Entrada:** las mismas 27 variables estandarizadas, no las PCs.

**Modelo:**

- Una neurona lineal sin sesgo, con pesos iniciales al azar normalizados.
- **Regla exacta normalizada:** `w ← (w + η·y·x) / ‖w + η·y·x‖`.
- Tasa de aprendizaje `η(t) = 0,5 / (1 + t)`.
- **5 épocas** (250 actualizaciones). Se eligieron a propósito para que la
  convergencia se vea en el gráfico. Con 500 épocas (corrida archivada) el
  índice coincide con PC1 a la perfección y no hay nada que mostrar.
- También se implementó la **forma de Taylor**, `w ← w + η·y·(x − y·w)`, solo
  como comparación. Con η = 0,5 diverge y con η = 0,05 no llega a converger en 5
  épocas. Por eso se usa la regla exacta.
- Criterio de convergencia: `wᵀCw / λ1 ≥ 0,99` y correlación con PC1 ≥ 0,99.
  Se corta la corrida si la norma o la varianza explotan.

**Resultados:**

- Correlación con PC1: **0,99995 ± 0,00004**. Razón de varianza: 0,999.
- El top 5 es idéntico al de PC1: `@futbolsinvueltas`, `@galeriacotidiana`,
  `@brillourbano`, `@crearenreels`, `@golconmate`. Solo hay 3 pares de
  creadores que intercambian posiciones contiguas.

**Para la presentación:** el índice **es PC1**, o sea, un índice de
tamaño/alcance. **No sirve para elegir automáticamente al primero**: el top 5
mezcla fútbol, arte y maquillaje, y no mide afinidad temática ni engagement por
dólar.

Más detalle: `resultados/ej1/oja/epocas_5/resumen.txt`.

---

## 1.4 Decisión para las campañas

**Script:** `src/ej1/campanas.py`, de solo lectura. Los supuestos están en
`src/ej1/campanas.toml` y las salidas en `resultados/ej1/campanas/`.

**Supuestos del grupo:**

1. Solo compra o adopta el público argentino. Por eso el alcance útil de un
   creador es: alcance argentino × % de posts del tópico del cliente. La
   eficiencia se mide en **USD por mil argentinos alcanzados en el tópico**.
2. Quien busca productos de belleza busca tutoriales y opiniones.
3. Guardados + visitas al perfil indican intención de compra.

**Candidatos de cada cliente:** los creadores de la región del mapa cuyo tópico
dominante es el del cliente, más los creadores cuyo propio tema principal es
ese. La segunda parte recupera a los grandes de cada nicho que el mapa agrupa
con los generalistas, como `@crearenreels`.

**Propuesta por cliente** (traducción de las respuestas del grupo):

| Cliente | Criterio |
|---|---|
| Fútbol | Masividad con afinidad alta y barato: alcance + % de fútbol + audiencia argentina − precio |
| Refugios | Descubrimiento + afinidad (perros y audiencia argentina), y luego comentarios y visitas |
| Belleza | Tutoriales, reseñas, guardados y comentarios, con afinidad (maquillaje y audiencia argentina) y alcance |
| Arte | Mezcla: alcance + tutoriales, con afinidad (arte y audiencia argentina) |

**Familias de métricas** (decisión del 27/10/2026; antes belleza y arte usaban
"argentinos alcanzados en el tópico" y fútbol el costo por mil de ellos, que
mezclaban alcance, audiencia argentina y tópico en un solo número):

| Familia | Métricas |
|---|---|
| Masividad | Alcance por publicación (`average_reach`) |
| Afinidad | % de posts del tópico del cliente y % de audiencia argentina |
| Descubrimiento | % del alcance que no sigue al creador |
| Profundidad | Tutoriales, reseñas, guardados, comentarios y visitas |
| Costo | Precio por post (peso negativo) |

Peso 1 para lo principal y 0,5 para lo secundario (`campanas.toml`). Con el
cambio, belleza reemplaza a `@mateconideas` por `@cosmeticaconsciente`; fútbol,
refugios y arte eligen a los mismos 5 (fútbol, en la mayoría de las semillas).
"Argentinos alcanzados en el tópico" y su costo siguen como métricas de
resultado en la comparación con followers.

**Figuras:**

| Figura | Qué muestra |
|---|---|
| `fig_mapa_regiones` | Tópico dominante, tamaño y audiencia argentina de cada neurona. |
| `fig_enfoques` | Por cliente, cada creador: tamaño vs. USD por mil argentinos, % de no seguidores e intención. |
| `fig_followers` | El top 10 por followers: precio, audiencia argentina y tema. |
| `fig_comparacion` | Propuesta vs. followers: USD por mil argentinos, intención por USD y alcance con el mismo presupuesto. |
| `fig_actualizacion` | Qué pasa si los datos cambian de un mes a otro sin reentrenar. |

**Resultados:**

- **Fútbol:**
  - Los chicos y educativos cuestan unos 95 USD por mil argentinos alcanzados.
  - Los grandes con audiencia argentina alta son los más baratos:
    `@golconmate` 19 y `@futbolcononda` 25.
  - Los grandes con audiencia argentina baja son los más caros (125–233).
  - El tamaño se correlaciona +0,57 con el % de no seguidores (masivo y
    descubrimiento van juntos) y −0,92 con la intención.
- **Refugios:** los más baratos por argentino son `@crearenreels` (grande) y
  dos chicos educativos (29–32 USD por mil). Descubrimiento crece con el tamaño
  (r = +0,83).
- **Belleza:** el costo por argentino no depende del tamaño (r = 0,00). Los
  educativos son más baratos (60 contra 90 USD por mil).
- **Arte:** casi no hay disyuntiva entre enfoques. La intención apenas baja
  con el tamaño (r = −0,41).
- **Propuesta vs. followers:**

| Cliente | USD por mil argentinos | Intención cada 100 USD | Alcance con el mismo presupuesto |
|---|---|---|---|
| Fútbol | 87 → 31 | 20 → 71 | 261 mil → 466 mil |
| Refugios | 104 → 27 | 20 → 88 | 218 mil → 257 mil (gasta 10,8 mil) |
| Belleza | 71 → 35 | 25 → 54 | 321 mil → 374 mil |
| Arte | 279 → 51 | 7 → 43 | 81 mil → 267 mil (gasta 17,5 mil) |

  En refugios y arte se acaban los candidatos antes de gastar todo el
  presupuesto. Eso concreta la desventaja de "nicho con pocos creadores".
- **Correlaciones de followers:**
  - precio: +0,82;
  - % de audiencia argentina: 0,00;
  - profundidad, **negativa**: guardados −0,74, comentarios −0,69,
    tutoriales −0,60.
- **Actualización mensual:** con una variación del 10%, un 3% de los creadores
  cambia de región; con 20%, un 10%.

---

## 2.1 QR con Hopfield

**Datos:** 5 QR de 37×37, o sea **N = 1369** neuronas. Negro = 1, blanco = −1.

**La capacidad no es el problema.** p/N = 0,004, muy por debajo de 0,15. El
problema es que **los QR se parecen mucho**:

- el 70% de los píxeles son iguales en los cinco;
- el solapamiento entre pares va de 0,58 a 0,86 (el máximo es food–wine);
- la distancia de Hamming mínima es de 96 bits.

**Decisión: comparar dos reglas de pesos.**

| Regla | Pesos | Puntos fijos | Resultado |
|---|---|---:|---|
| Hebb (la de la teoría) | `W = (1/N)·XᵀX`, diagonal en 0 | 0 de 5 | **Todo cae en el mismo espurio**: el voto mayoritario de los 5 QR, que tiene **menos energía** que los patrones guardados. Con 2 QR funciona; con 3 o más, falla. |
| Pseudoinversa (Personnaz) | `W = Xᵀ(XXᵀ)⁻¹X`, diagonal en 0 | 5 de 5 | Recupera bien; ver la tabla de abajo. |

La pseudoinversa es una **extensión** que se presenta para mostrar que sí hay
una solución viable. **La base teórica sigue siendo Hebb**, y su falla es en sí
misma un resultado: muestra que capacidad y ortogonalidad son cosas distintas.

**Dinámica:** asincrónica secuencial, con un máximo de 100 barridos. Si el campo
local da 0, la neurona conserva su estado. Energía: `E = −½ SᵀWS`.

**Deterioros implementados:** 3 de los 6 que preveía el plan
(`docs/noise.md`). Los niveles se fijaron antes de correr y no se ajustaron
según los resultados.

| Deterioro | Qué hace |
|---|---|
| `ruido` | invierte una fracción de los píxeles al azar |
| `esquina` | pone en blanco un bloque de área `nivel·N` en una esquina al azar |
| `manchas` | invierte `nivel·N` píxeles agrupados en bloques de 3×3 |

**Resultados con pseudoinversa** (Hebb: 0% en todos los niveles):

| Deterioro | 100% de recuperación hasta | Dónde empieza a fallar |
|---|---|---|
| Ruido | 40% | 75% a 45%; 0% a 50% |
| Esquina | 40% del área | 88% a 50% |
| Manchas | 30% | 85% a 40%; 0% a 50% |

- Con 50% de ruido se pierde toda la información: la imagen equivale a una
  imagen al azar.
- **Iteraciones:** la red se estabiliza en ≈ 2 barridos (1 que corrige y 1 que
  verifica) y sube a ≈ 2,8 cerca del punto de falla. La energía siempre baja.
- **Limitaciones para la presentación:**
  - Los patrones están correlacionados.
  - Agregar un menú obliga a recalcular todos los pesos.
  - Una foto real primero hay que alinearla, escalarla a 37×37 y binarizarla.
  - Cuando falla, la red a veces devuelve **otro QR válido** ("incorrecto"):
    es peor que un espurio, porque lleva al menú equivocado.

`docs/structure.md` y `decisiones.md` dicen que 2.1 está "en fase 0", pero eso
está desactualizado: los resultados están en `resultados/ej2/hopfield/qr_hebb/`,
`qr_pseudoinversa/` y `analisis_qr/`.

---

## 2.2 Letras 5×5 (prueba previa)

- Letras **J, O, T, A**: N = 25 neuronas y p = 4 patrones. Regla de Hebb y
  dinámica asincrónica.
- **Ortogonalidad:** ningún par es ortogonal. El solapamiento medio es 0,17 y el
  máximo es 0,36 (O–A). La distancia de Hamming mínima es 8 (O–A).
- **Capacidad:** p/N = 0,16, por encima del límite de 0,15, y aun así **las 4
  letras son estables**. El 15% es una guía estadística, no una regla dura. A
  es la letra con menor margen de estabilidad, porque se parece a O.
- **Ruido** (60 consultas por nivel):

| Ruido | Correctas | Espurios | Incorrectas |
|---:|---:|---:|---:|
| 10% | 98% | 1 | 0 |
| 20% | 78% | 9 | 4 |
| 30% | 60% | 13 | 11 |
| 40% | 25% | 27 | 18 |
| 50% | 8% | 27 | 28 |

- **Estado espurio de ejemplo:** ruido 20%, semilla 0, J de origen. Es estable,
  no corresponde a ninguna letra (la más cercana es T, a distancia 4) y tiene
  energía −11,68.
- T es la letra más robusta y A la más frágil.
- Gráficos: `resultados/ej2/hopfield/figuras/` (`src/ej2/hopfield-figuras.py`).

---

## Pendientes

1. Validar los supuestos de 1.4 (sección A) e interpretar los gráficos de
   `resultados/ej1/campanas/`.
2. Elegir la grilla final de Kohonen (recomendado: k=4 euclidiana) y escribir
   qué perfil reúne cada región.
3. Cerrar el umbral de PCA (90%) en `decisiones.md`.
4. Actualizar `structure.md` y `decisiones.md`: 2.1 ya está hecho, hay
   secciones duplicadas y faltan los scripts `kohonen-comparacion.py`,
   `kohonen-perfiles.py` y `hopfield-qr-analisis.py`.
