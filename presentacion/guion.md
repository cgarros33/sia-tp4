# Guion de la presentación — TP4 SIA: aprendizaje no supervisado

Duración objetivo: **≈ 27 minutos**, 29 diapositivas + backup. Formato 16:9.

## Indicaciones de diseño

- El **título de cada diapositiva es la conclusión**, no el tema.
- Una idea por diapositiva: el gráfico ocupa la mayor parte; como mucho 3
  viñetas cortas en pantalla. El resto va en las notas del orador.
- Las figuras están en `figuras/` con el número de la diapositiva. Usarlas sin
  recortar salvo donde se indica.
- Idioma: español rioplatense. Números con coma decimal (0,82).
- Colores de las figuras: fútbol verde agua, perros amarillo, maquillaje
  naranja, arte azul; "followers" gris y "nuestra propuesta" violeta. Mantener
  esa asociación si se agregan elementos.

---

## Introducción (1')

### 0 · Portada (15")
- TP4 — Métodos de aprendizaje no supervisado. Sistemas de Inteligencia
  Artificial, ITBA, 2º cuatrimestre 2026.
- Integrantes: Celestino Garrós (64375), Leo Weitz (64365), Federico Ignacio
  Ruckauf (64356), Matías Romanato (62072).

### 1 · Elegir por followers es elegir por tamaño: ¿qué más hay en los datos? (45")
- **En pantalla:** el problema (las campañas no recuperan la inversión) y la
  agenda: PCA → Kohonen → Oja → decisiones → Hopfield.
- **Notas:** "Vamos a mostrar que followers mide una sola dimensión de las
  varias que importan, y cómo usar las otras."

## 1.1 Exploración y PCA (3'30")

### 2 ·  Variables redundantes (1')
- **Figura:** `figuras/02_correlaciones.png`
- **En pantalla:** se sacan 4 variables redundantes exactas (precio en ARS,
  alcance argentino estimado, engagement total, % orgánico). El bloque de
  tamaño está casi duplicado (reach, impressions y views: r = 0,99).
- **Notas:** las variables están ordenadas por su carga en PC1: arriba un bloque
  "masivo" (alcance, videos cortos, tendencias, followers, precio), abajo uno
  "profundo" (textos largos, guardados, comentarios, visitas, tutoriales) y
  entre ambos correlación negativa (azul). Es un adelanto de PC1. Se estandariza
  porque las escalas van de cientos de miles a porcentajes.

### 3 · 10 componentes explican el 91,5% de la variación (1')
- **Figura:** `figuras/03_scree.png`
- **En pantalla:** Kaiser sugiere 6, el codo 6–7; 10 es conservador.
- **Notas:** la matriz tiene rango 25 porque tópicos y tipos de contenido suman
  100% cada uno.

### 4 · La mitad de la variación es "masivo vs. profundo". El resto es especificacion de nichos (1'30")
- **Figuras:** `figuras/04a_biplot.png` + `figuras/04b_cargas.png`
- **En pantalla:**
  - PC1 (50%): reach, views, tendencias ↔ textos largos, guardados,
    comentarios, visitas al perfil.
  - PC2: arte/showcase ↔ audiencia argentina/perros. PC3: fútbol patrocinado.
    PC4: maquillaje ↔ fútbol.
- **Notas:** followers correlaciona 0,87 con PC1: **elegir por followers es
  mirar solo PC1**. PC1 ya contiene la tensión de la consigna (masivo vs.
  profundo).

## 1.2 Red de Kohonen (6'15")

### 5 · Cómo entrenamos y medimos el mapa (1')
- **Figura:** esquema a dibujar: 50 creadores → 10 componentes → grilla 4×4 de
  neuronas → cada creador va a su neurona más parecida.
- **En pantalla:** métricas: error de cuantización (QE, qué tan lejos queda cada
  creador de su neurona), error topográfico (TE), neuronas sin creadores.
  3 semillas, media ± desvío.
- **Notas:** los scores del PCA se usan sin reestandarizar para que PC1 conserve
  su peso.

### 6 · Barrimos un hiperparámetro por vez con la grilla fija (1'15")
- **Figura:** `figuras/06_barrido_hiperparametros.png`
- **En pantalla:** si el tamaño de grilla se deja libre, gana el más grande
  (el mapa memoriza). Variar dos parámetros juntos no mejoró nada fuera del
  ruido (p ≈ 0,40).
- **Notas:** eje compartido para comparar pasos. La topología sí importa
  (rectangular 2,82 vs. hexagonal 2,90). En radio, η e iteraciones las
  diferencias entre los mejores valores son menores que el desvío; la
  inicialización y el decaimiento del radio no cambian nada (con radio inicial
  0,3·k = 1,2 el radio casi no decrece). La distancia exponencial ganaba por
  0,007 de QE (menos que su desvío) y con el doble de TE; por eso este barrido
  de control fija la distancia euclidiana, y es el mapa final.

### 7 · Balanceando error y neuronas vacías (45")
- **Figura:** `figuras/07_metricas_por_grilla.png`
- **En pantalla:** la QE baja con más neuronas, pero en 5×5 quedan ≈ 4 vacías.
  4×4 euclidiana: QE 2,80 ± 0,01, TE 0,07, ≈ 3 creadores por neurona.
- **Notas:** medida en las 27 variables, la QE con 10 PCs (3,19) es casi la
  misma que con todas (3,17): la reducción no perdió información útil.

### 8 · Tamaño en un eje, tópico en el otro (1'15")
- **Figura:** `figuras/08_mapa_regiones.png`
- **En pantalla:** arriba los grandes (400–600 mil seguidores), abajo los chicos
  (17–130 mil). Arte abajo a la izquierda, fútbol en la columna 1, perros a la
  derecha, maquillaje arriba. La neurona (0,0) es mixta: grandes generalistas.
- **Notas:** cada cliente tiene su región; se usa en 1.4.

### 9 · La Matriz U: Frontera entre los grandes y el resto (1')
- **Figura:** `figuras/09_matriz_u.png`
- **En pantalla:** fila 1 con valores altos (2,8–3,0) separa a los grandes. La
  neurona vacía (1,0) está en la frontera. Valles abajo (1,3–1,6): creadores
  chicos muy parecidos.
- **Notas:** **distancia alta = cambio brusco de perfil; baja = creadores
  intercambiables** (sirve para encontrar sustitutos).

### 10 · Coincide con el biplot y agrega lo que PC1–PC2 no separan (1')
- **Figura:** `figuras/10_mapa_sobre_biplot.png`
- **En pantalla:** la grilla se despliega sobre el biplot: fila 0 a la derecha
  (PC1 alto), arte arriba, perros abajo. El tamaño explica el 96% de la
  variación de PC1 a lo largo de la grilla.
- **Notas:** como el tamaño domina, los grandes de un nicho caen con los
  generalistas (ej.: @crearenreels, 67% perros, en la neurona mixta). Se
  resuelve en 1.4.

## 1.3 Regla de Oja (2')

### 11 · La regla de Oja aprende PC1 (1')
- **Figuras:** `figuras/11a_oja_convergencia.png` + `figuras/11b_oja_vs_pc1.png`
- **En pantalla:** correlación 0,99995 con PC1. Regla exacta, η = 0,5/(1+t),
  5 épocas para que se vea la convergencia.
- **Notas:** la aproximación de Taylor diverge con η alto.

### 12 · El índice mide tamaño: elegir siempre el máximo es un error (1')
- **Figura:** `figuras/12_oja_vs_negocio.png`
- **En pantalla:** el #1 del índice es @futbolsinvueltas: 27% de audiencia
  argentina y USD 4.655 por publicación. El top 5 mezcla tópicos.
- **Notas:** es otra forma de elegir por followers; para un refugio de perros
  el #1 no sirve.

## 1.4 Decisión para las campañas (7'15")

### 13 · Supuesto central: solo compra el público argentino (30")
- **Figura:** ninguna; 3 supuestos y 2 métricas.
- **En pantalla:**
  - Alcance útil = alcance argentino × % de posts del tópico del cliente.
  - Eficiencia = **USD por cada mil argentinos alcanzados en el tópico**.
  - Intención de compra = guardados + visitas al perfil.
- **Notas:** quien busca belleza busca tutoriales y opiniones. Candidatos de
  cada cliente: su región del mapa + creadores cuyo tema principal es el del
  cliente (para no perder a @crearenreels).

### 14 · Enfoques de campaña (1')
- **Figura:** `figuras/14_enfoques_todos.png`
- **En pantalla:** cada fila es un enfoque, cada punto un creador (color =
  % de audiencia argentina).
  - Descubrimiento crece con el tamaño (r +0,38 a +0,83).
  - Intención cae con el tamaño (r −0,41 a −0,92).
  - El costo por argentino **no depende del tamaño**: depende de la audiencia
    argentina.

### 15 · Fútbol: masivo, eligiendo creadores con audiencia argentina alta (1'15")
- **Figuras:** `figuras/15_enfoques_futbol.png` + `figuras/08_mapa_regiones.png`
  (chica, resaltando la columna 1)
- **En pantalla:**
  - Kohonen separa 3 zonas de fútbol: grandes, medianos con muchos no
    seguidores (pero 41% argentinos) y chicos educativos.
  - Más baratos por argentino: @golconmate 19 USD, @futbolcononda 25,
    @tribunacreativa 38 (grandes, muy argentinos).
  - Educativos: ≈ 95 USD por mil y llegan a pocos → complemento.
- **Notas:** los grandes con poca audiencia argentina son los más caros
  (125–233). Los educativos generan casi el doble de intención por persona.

### 16 · Refugios, belleza y arte: la mezcla gana cuando lo educativo es barato (1'15")
- **Figura:** `figuras/16_enfoques_refugios_belleza_arte.png`
- **En pantalla:**
  - **Refugios: descubrimiento + profundidad.** @crearenreels (24 USD, 76% no
    seguidores) + chicos educativos muy argentinos (@huellasyconsejos 29,
    @consejoscaninos 32) que generan más consultas.
  - **Belleza: profundidad.** Educativos más baratos (60 vs. 90 USD por mil),
    p. ej. @brochasymates; sumar un grande argentino (@maquillajereal 27).
  - **Arte: mezcla.** Grandes argentinos (@galeriacotidiana 43) + educativos
    argentinos (@colorenmovimiento 53): lo educativo cuesta lo mismo o menos.
- **Notas:** a diferencia de fútbol, en belleza y arte lo educativo no es solo
  un complemento: cuesta lo mismo o menos por argentino y le queda bien al
  producto.

### 17 · ¿Qué compra hoy la agencia al elegir por followers? (1')
- **Figura:** `figuras/17_que_elige_followers.png`
- **En pantalla:** top 10: 6 maquillaje, 3 fútbol, 1 perros, **0 arte**.
  Followers: r = +0,82 con precio, 0 con audiencia argentina, **−0,74 con
  guardados**.
- **Notas:** @huellasyconsejos (94% argentina, USD 75) vs. @futbolsinvueltas
  (27%, USD 4.655). Elegir por followers trae a los menos profundos.

### 18 · Con nuestra propuesta cada dólar llega a más argentinos interesados (1')
- **Figura:** `figuras/18_propuesta_vs_followers.png`
- **En pantalla:** 1,8 a 5,5 veces más barato por argentino; 1,9 a 6,2 veces
  más intención por dólar; con el mismo presupuesto llega a más en los 4
  clientes.
- **Notas:** en refugios y arte se acaban los candidatos antes de gastar todo
  (10,8 y 17,5 mil USD): la desventaja de un nicho con pocos creadores.

### 19 · Ventajas y desventajas de cada estrategia (30")
- **Figura:** tabla de dos columnas.
- **Followers.** Ventajas: simple, maximiza alcance general. Desventajas: caro,
  ignora audiencia argentina y tópico, trae a los menos profundos.
- **Propuesta.** Ventajas: más barata y afín, encuentra creadores de nicho,
  sustitutos vía Matriz U, un cliente nuevo de un rubro existente solo elige su
  región. Desventajas: depende de los datos, requiere mantenimiento, en nichos
  chicos se acaban los candidatos, un generalista grande con 30% del tópico
  puede llegar a más que un especialista.

### 20 · Mantenimiento: el mapa aguanta la actualización mensual (45")
- **Figura:** `figuras/20_actualizacion_mensual.png`
- **En pantalla:**
  - Campaña nueva de un rubro existente: no se reentrena; se elige región y
    métricas.
  - Rubro o variable nueva: rehacer PCA y Kohonen.
  - Pocos creadores nuevos: se proyectan con el PCA y se ubican en el mapa;
    muchos: reentrenar.
  - Con 10% de variación mensual solo un 3% cambia de región.
- **Notas:** al reentrenar el mapa puede quedar rotado: hay que volver a nombrar
  las regiones.

## 2. Memoria asociativa: Hopfield (6'15")

### 21 · Letras: 4 patrones estables aunque se supera el 15% (1')
- **Figura:** `figuras/21_letras_ortogonalidad.png`
- **En pantalla:** p/N = 0,16 > 0,15 y las 4 son estables. Ningún par es
  ortogonal; O–A es el más parecido (0,36).
- **Notas:** el 15% es una guía estadística, no un umbral exacto. A tiene el
  menor margen (0,16) por parecerse a O.

### 22 · Con mucho ruido aparecen estados espurios (1')
- **Figura:** `figuras/22_letras_ruido_espurio.png`
- **En pantalla:** recupera 98% con 10% de ruido y 8% con 50%. Desde 40%
  dominan los espurios y las letras equivocadas.
- **Notas:** el espurio del ejemplo es estable, no es ninguna letra (a 4 píxeles
  de T) y tiene la misma energía que J.

### 23 · Los 5 QR se parecen demasiado entre sí (45")
- **Figura:** `figuras/23_qr_correlacion.png`
- **En pantalla:** N = 1.369, p/N = 0,004: **la capacidad no es el problema**.
  El 70% de los píxeles son iguales en los cinco (solapamiento 0,58–0,86).

### 24 · Hebb no recupera ningún QR, la pseudoinversa los 5 (1')
- **Figura:** `figuras/24_qr_capacidad.png`
- **En pantalla:** Hebb: 0/5 puntos fijos; todos caen al "promedio" de los QR.
  Con 2 QR funciona, con 3 o más falla. Pseudoinversa W = Xᵀ(XXᵀ)⁻¹X: 5/5.
- **Notas:** capacidad y ortogonalidad son cosas distintas.

### 25 · La pseudoinversa recupera hasta un 40% de deterioro (1')
- **Figuras:** `figuras/25a_qr_recuperacion.png` + `figuras/25b_qr_ejemplos.png`
- **En pantalla:** 100% hasta 40% de ruido o de esquina rota. Manchas fallan
  desde 35%. Con 50% de ruido no queda información.

### 26 · La energía siempre baja y converge en ≈ 2 barridos (45")
- **Figuras:** `figuras/26a_qr_energia.png` + `figuras/26b_qr_barridos.png`
- **En pantalla:** con Hebb la energía termina **por debajo** de la del QR
  original: el espurio es más estable que el patrón verdadero.

### 27 · Limitaciones: cuando falla, puede devolver otro menú (45")
- **Figura:** `figuras/27_qr_fallas.png`
- **En pantalla:**
  - Con media esquina rota, "food" se recupera como "wine" (el par más
    parecido, 96 píxeles): el lector abre la carta equivocada.
  - Patrones correlacionados; agregar un menú obliga a recalcular los pesos;
    una foto real necesita alineación, escala y binarización.
- **Notas:** el QR ya trae corrección de errores (≈ 30%); Hopfield aporta si se
  rompen los patrones de posición de las esquinas.

## Cierre (45")

### 28 · Conclusiones
- **Ejercicio 1:** followers mide tamaño. PCA y Kohonen muestran los nichos;
  Oja vuelve a medir tamaño. Con supuestos explícitos la selección es 2 a 6
  veces más eficiente.
- **Ejercicio 2:** en Hopfield importa la ortogonalidad de los patrones, no solo
  su cantidad.

---

## Backup (solo si preguntan)

| Archivo | Para qué |
|---|---|
| `figuras/B1_perfil_neuronas.png` | Perfil de cada neurona en las variables originales |
| `figuras/B2_creadores_por_neurona.png` | Qué creadores hay en cada neurona |
| `figuras/B3_estabilidad_semillas.png` | Estabilidad del mapa entre semillas (ARI) |
| `figuras/B4_oja_sensibilidad.png` | Sensibilidad de Oja a η y su decaimiento |
| `figuras/B5_oja_reglas.png` | Regla exacta vs. aproximación de Taylor |

## Origen de las figuras

Las figuras son copias de `resultados/` y se regeneran con:
`src/ej1/eda-pca.py`, `src/ej1/figuras-presentacion.py` (02 y 06),
`src/ej1/kohonen-perfiles.py`,
`src/ej1/kohonen-comparacion.py`, `src/ej1/oja.py`, `src/ej1/campanas.py`,
`src/ej2/hopfield-qr-analisis.py` y `src/ej2/hopfield-figuras.py`.
