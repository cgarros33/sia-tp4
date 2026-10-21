1. # Campaña marketing:

Busco content creators \-\> ahora filtro solo por followers pero no me alcanza   
Clientes \-\> refugio corte nickson(adopciones) ; ropa futbol ; cosmeticos-ish ; arte (3x ventas)  
Creadores.csv con parametros a cagarse de stats de los posteos de esta gente, incluyendo con topics para cada uno de los clientes, organico o no, % de modalidades de publicacion, formato, largo, etc

MUY IMPORTANTE: POR QUÉ VARIABLE  TRATAMOS DE AGRUPAR? (y/o como vamos a formar el indice de la parte 3 \-\> most likely ordenar por engagement/\$ asi con la minima plata tengo el maximo reach, ver con este analisis de correlacion cual es el mejor proxy de engagement, por sesgos de la vida real supongo que puede estar tirado a comentarios pero hay que ver que escupe el dataset)

PARTE 1 \-\> EDA (exploratory data analysis) y PCA (primary component analysis) (me quedo con las variables que son señal y descarto el ruido)

* Armo una matriz de covarianzas \-\> matriz simetrica def pos me da asociacion lineal positiva y negativa entre variables entre todo mi conjunto de datos  
* De las q vars originales agarrar p (SON 33 VARIABLES ORIGINALES y 50 filas/influenceers) \-\> tamaño adecuado de q? La intuicion me dice que va a estar cerca de 10  
* El algoritmo analiza variabilidad, correlacions y me pide setear EL MINIMO PORCENTAJE DE LAS VARIANZAS QUE YO QUIERO AGARRAR \-\> analizar cual porcentaje parece adecuado para esto, el ejemplo que da la teórica es 95%, yo haría 90 95 y 99 para ver cuantas variables da cada una como resultado y ahi decidir

PARTE 2 \-\> COJONES (kohonen) : implementar red para agrupar los creadores segun todo lo que me dieron. 

Analizar:

1) Hiperparametros y justificar eleccion final (IMPORTANTE: variar de a un parametro con todo el resto fijo hasta elegir el que nos gusta, esto nos lo marco euge. Preguntarle a claudio si puede haber alguno de los factores que vaya en combo como para analizarlos en combo producto cartesiano muchas opciones)   
2) Grilla de neuronas \- distribucion creadores contenido en la grilla y matriz U (leer que mierda era la matriz U)  
3) Interpretar resultados \-\> ejemplos y mas metricas REVER una vez que esté hecho contestar todas las preguntas y  ver si tiene

PARTE 3 \-\> INDICE (Oja)

* Implementar regla oja sobre datos  
* ver que representa este indice (seria la primera componente principal osea la variable mas indicativa del resto creo. Esto es una medida valida para elegir para el negocio sin mas contexto?)  
* como interpretar  
* podemos quedarnos solo con ese indice como forma de elegir agarrando el mas alto?? Depende de que represente

Qué hace Oja?? \-\> es una proyeccion sobre la direccion de maxima variacion de los datos (eje 1ra componente principal) pero como eso diverge normaliza en base a norma tamaño anterior. 

Converge 

PARTE 4 \-\> DECISION para las campañas

Interpretar todos los pasos intermedios para ver que conviene priorizar en cada momento

* Fulbito \-\> priorizo masividad, descubrimiento o profundidad educativa? Hunch says reach pero hay que ver que resulta de analizar el dataset que no se leer csvs de 33 columnas asi facil  
* Comparar lo de arriba contra followers, desventajas y ventajas  
* Mantenimiento sistema : nueva campaña que hago? Tengo que reanalizar todo? Que cambia si cambia el numero de variables o de creadores de contenido? 


  
NOTA: el foco no esta en que tenga sentido lo de marketing obviamente, esta en el aprendizaje no supervisado, si necesito un supuesto medio debil no es tan grave

2. # Memoria asociativa (qrs) (falta completar cosas)

* Problema \-\> esquina qr rota  
* 5 cartas digitales : food wine beverages dessert specials  
* Idea a probar : Hopfield asociativa (sin contar el numero de pixeles y/o bits que codifique los qrs 0,15 \<\< \#bits asi que voy a estar chill con que entren los 5, desconozco a priori tema estados espurios) SON 37x37  
* Tengo que representarlos como matriz binaria de 37 x 37, estan en datos/hopfield  
* Almacenar los 5 vec en la red \-\> construir patrón consulta

Analizar:

1) Alteracion QRs: ruido aleatorio (niveles de ruido \+ simulacion de deterioro) \- imprimir alguno y cagarlo a trompadas a ver como queda?  
2) Cantidad de iteraciones y energia en cada paso de la red (hasta llegar al estado estable y/o espurio si la cagamos con eso supongo  
3) Limitaciones??? Cuando se rompe? Que tan lejos se puede llevar sin preprocesamiento a nivel de mala calidad de la imagen cuando claramente no es el formato mas adecuado? Que ahgo si la foto tiene mas pixeles de los necesarios? Se puede pasar sin preprocesamiento?  
* RECOMENDACION CATEDRA \-\> arrancar haciendo letras del abecedario y tratar de detectar con eso (Almacenar 4 \-\> asociar matriz 5x5 a letra)  
* Analisis ortogonalidad (tratar de entender) y de capacidad de almacenamiento de la red (25\*0.15 \= 4.25 que es lo que nos dicen que hagamos)

RANDOM BULLSHIT GO

Teoria artificializando.pdf \-\> pgs 101-132 inclusive

Voy  a intentar usar las skills de Python Pro y de The Fool de [https\://github.com/farmage/opencode-skills/](https://github.com/farmage/opencode-skills/)

Voy a tratar de mantener tamaños de commits que me de la cabeza leer a mi leer

