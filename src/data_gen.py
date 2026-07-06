"""Generate the toy QA dataset used by every experiment.

The dataset is hand-written into `data/toy_qa.jsonl` so the file is committed
to the repo and the experiment is fully reproducible without a network
connection. The examples below are split into two categories:

  * `expected_short`  – 40% of the rows; the answer is a single word / number
  * `expected_long`   – 60% of the rows; the answer is a free-form sentence

Difficulty labels (`easy` / `medium` / `hard`) are used to break down the
final metrics in the analysis stage.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Every row is a dict with the schema declared in the spec. A difficulty
# field is included so the analysis can slice the results.
_SHORT_ROWS: list[dict[str, Any]] = [
    {"id": "s01", "question": "¿Cuál es la capital de Francia?", "expected_short": "París", "difficulty": "easy"},
    {"id": "s02", "question": "¿Cuántos lados tiene un hexágono?", "expected_short": "6", "difficulty": "easy"},
    {"id": "s03", "question": "¿En qué año llegó el hombre a la Luna?", "expected_short": "1969", "difficulty": "easy"},
    {"id": "s04", "question": "¿Cuál es el símbolo químico del oro?", "expected_short": "Au", "difficulty": "easy"},
    {"id": "s05", "question": "¿Cuántos continentes hay?", "expected_short": "7", "difficulty": "easy"},
    {"id": "s06", "question": "¿Quién escribió 'Cien años de soledad'?", "expected_short": "Gabriel García Márquez", "difficulty": "medium"},
    {"id": "s07", "question": "¿Cuál es el río más largo del mundo?", "expected_short": "Amazonas", "difficulty": "medium"},
    {"id": "s08", "question": "¿En qué país se encuentra la Torre de Pisa?", "expected_short": "Italia", "difficulty": "easy"},
    {"id": "s09", "question": "¿Cuántos huesos tiene el cuerpo humano adulto?", "expected_short": "206", "difficulty": "medium"},
    {"id": "s10", "question": "¿Cuál es el océano más grande?", "expected_short": "Pacífico", "difficulty": "easy"},
    {"id": "s11", "question": "¿Quién pintó la Mona Lisa?", "expected_short": "Leonardo da Vinci", "difficulty": "easy"},
    {"id": "s12", "question": "¿Cuál es el idioma más hablado del mundo por nativos?", "expected_short": "Mandarín", "difficulty": "medium"},
    {"id": "s13", "question": "¿En qué año cayó el Muro de Berlín?", "expected_short": "1989", "difficulty": "medium"},
    {"id": "s14", "question": "¿Cuál es el metal más ligero?", "expected_short": "Litio", "difficulty": "medium"},
    {"id": "s15", "question": "¿Cuántos planetas tiene el sistema solar?", "expected_short": "8", "difficulty": "easy"},
    {"id": "s16", "question": "¿Cuál es la moneda de Japón?", "expected_short": "Yen", "difficulty": "easy"},
    {"id": "s17", "question": "¿Quién propuso la teoría de la relatividad?", "expected_short": "Einstein", "difficulty": "easy"},
    {"id": "s18", "question": "¿Cuál es el animal terrestre más rápido?", "expected_short": "Guepardo", "difficulty": "medium"},
    {"id": "s19", "question": "¿En qué continente está Egipto?", "expected_short": "África", "difficulty": "easy"},
    {"id": "s20", "question": "¿Cuántos segundos tiene una hora?", "expected_short": "3600", "difficulty": "easy"},
    {"id": "s21", "question": "¿Quién fue el primer presidente de Estados Unidos?", "expected_short": "George Washington", "difficulty": "medium"},
    {"id": "s22", "question": "¿Cuál es el gas más abundante en la atmósfera?", "expected_short": "Nitrógeno", "difficulty": "medium"},
    {"id": "s23", "question": "¿En qué país se originaron los Juegos Olímpicos?", "expected_short": "Grecia", "difficulty": "medium"},
    {"id": "s24", "question": "¿Cuál es el hueso más largo del cuerpo humano?", "expected_short": "Fémur", "difficulty": "medium"},
    {"id": "s25", "question": "¿Cuántas cuerdas tiene una guitarra estándar?", "expected_short": "6", "difficulty": "easy"},
    {"id": "s26", "question": "¿Cuál es el planeta más cercano al Sol?", "expected_short": "Mercurio", "difficulty": "easy"},
    {"id": "s27", "question": "¿En qué año se descubrió América?", "expected_short": "1492", "difficulty": "easy"},
    {"id": "s28", "question": "¿Cuál es la fórmula química del agua?", "expected_short": "H2O", "difficulty": "easy"},
    {"id": "s29", "question": "¿Cuántas caras tiene un cubo?", "expected_short": "6", "difficulty": "easy"},
    {"id": "s30", "question": "¿Cuál es el autor de 'Don Quijote de la Mancha'?", "expected_short": "Miguel de Cervantes", "difficulty": "medium"},
]

_LONG_ROWS: list[dict[str, Any]] = [
    {
        "id": "l01",
        "question": "Explica la diferencia entre aprendizaje supervisado y no supervisado en dos frases.",
        "expected_long": (
            "El aprendizaje supervisado utiliza datos etiquetados para entrenar al modelo con ejemplos "
            "de entrada y salida esperada, mientras que el no supervisado trabaja con datos sin etiquetas "
            "y busca patrones o agrupamientos. El primero se usa típicamente para clasificación y regresión, "
            "mientras que el segundo se aplica a clustering y reducción de dimensionalidad."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l02",
        "question": "¿Por qué el agua salada hierve a una temperatura más alta que el agua dulce?",
        "expected_long": (
            "El agua salada hierve a una temperatura más alta porque las sales disueltas elevan el punto "
            "de ebullición mediante un fenómeno llamado elevación ebulloscópica. Los iones de la sal "
            "interfieren con la evaporación de las moléculas de agua, requiriéndose más energía para "
            "que la presión de vapor iguale a la presión atmosférica."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l03",
        "question": "Describe brevemente la causa principal del efecto invernadero.",
        "expected_long": (
            "El efecto invernadero se produce cuando ciertos gases de la atmósfera, principalmente dióxido "
            "de carbono y metano, atrapan parte de la radiación infrarroja emitida por la superficie terrestre. "
            "Esto eleva la temperatura media del planeta más allá de lo que se observaría sin atmósfera. "
            "La actividad humana, especialmente la quema de combustibles fósiles, ha incrementado "
            "significativamente la concentración de estos gases."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l04",
        "question": "¿Qué es una API REST y en qué se diferencia de una API SOAP?",
        "expected_long": (
            "Una API REST es un estilo arquitectónico que utiliza los métodos estándar de HTTP (GET, POST, "
            "PUT, DELETE) para interactuar con recursos identificados por URLs. SOAP, en cambio, es un "
            "protocolo más rígido basado en XML con un contrato formal (WSDL). REST es generalmente más "
            "ligero, stateless y fácil de cachear, mientras que SOAP ofrece características avanzadas "
            "como transacciones y seguridad estandarizada."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l05",
        "question": "Explica qué es la entropía en el contexto de la termodinámica.",
        "expected_long": (
            "La entropía es una magnitud física que mide el grado de desorden o de incertidumbre de un "
            "sistema. En termodinámica, la segunda ley establece que la entropía total del universo "
            "siempre aumenta en procesos espontáneos, lo que determina la dirección temporal de los "
            "fenómenos naturales. Cuanto mayor es la entropía, menos energía útil queda disponible "
            "para realizar trabajo."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l06",
        "question": "¿Qué es el overfitting y cómo se puede mitigar?",
        "expected_long": (
            "El overfitting ocurre cuando un modelo aprende los detalles y el ruido del conjunto de "
            "entrenamiento hasta el punto de perjudicar su rendimiento sobre datos nuevos. Se mitiga "
            "con técnicas como regularización (L1/L2), dropout en redes neuronales, validación cruzada, "
            "early stopping y aumento de datos. La clave es mantener un equilibrio entre el sesgo y la "
            "varianza del modelo."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l07",
        "question": "Describe brevemente qué hace un sistema operativo.",
        "expected_long": (
            "Un sistema operativo es el software que gestiona los recursos de hardware de un computador "
            "y provee servicios a las aplicaciones de usuario. Sus funciones principales incluyen la "
            "administración de procesos, memoria, dispositivos de entrada/salida y el sistema de archivos. "
            "Actúa como intermediario entre el usuario y el hardware, abstrayendo los detalles de bajo nivel."
        ),
        "difficulty": "easy",
    },
    {
        "id": "l08",
        "question": "¿Por qué se producen las estaciones del año?",
        "expected_long": (
            "Las estaciones del año se producen por la inclinación del eje de rotación de la Tierra, "
            "que está inclinado aproximadamente 23,5 grados respecto al plano de su órbita alrededor "
            "del Sol. Esta inclinación hace que diferentes regiones del planeta reciban cantidades "
            "distintas de luz solar a lo largo del año, generando variaciones climáticas cíclicas."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l09",
        "question": "Explica qué es un transformador en el contexto del aprendizaje profundo.",
        "expected_long": (
            "Un transformador es una arquitectura de red neuronal introducida en 2017 que utiliza "
            "mecanismos de atención para procesar secuencias de datos en paralelo. A diferencia de las "
            "redes recurrentes, no requiere procesar los elementos en orden, lo que acelera el "
            "entrenamiento. Es la base de modelos de lenguaje como GPT y BERT, y revolucionó el "
            "procesamiento de lenguaje natural y otras áreas."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l10",
        "question": "¿Qué es la deuda técnica y por qué importa en el desarrollo de software?",
        "expected_long": (
            "La deuda técnica es el costo futuro implícito en elegir una solución fácil hoy en lugar de "
            "un enfoque mejor pero más costoso a corto plazo. Se manifiesta como código difícil de mantener, "
            "falta de documentación o atajos en el diseño. Importa porque disminuye la velocidad del equipo "
            "con el tiempo y puede hacer que los cambios pequeños se vuelvan caros y arriesgados."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l11",
        "question": "Explica brevemente la diferencia entre IPv4 e IPv6.",
        "expected_long": (
            "IPv4 utiliza direcciones de 32 bits, lo que permite aproximadamente 4.300 millones de "
            "direcciones únicas, mientras que IPv6 emplea 128 bits y ofrece una cantidad prácticamente "
            "ilimitada. IPv6 también simplifica el enrutamiento, mejora la seguridad mediante IPsec y "
            "elimina la necesidad de NAT. La transición es gradual y ambos protocolos coexisten en la "
            "actualidad."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l12",
        "question": "¿Qué es un vector de embedding y para qué se utiliza?",
        "expected_long": (
            "Un vector de embedding es una representación numérica densa de un dato (palabra, imagen, "
            "concepto) en un espacio de menor dimensión donde las relaciones semánticas se preservan como "
            "distancias o ángulos. Se utiliza para capturar similitud, alimentar modelos de aprendizaje "
            "automático y permitir búsquedas semánticas eficientes. Es la base de técnicas modernas como "
            "RAG y recommendation systems."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l13",
        "question": "Explica por qué el cielo se ve azul durante el día.",
        "expected_long": (
            "El cielo se ve azul debido a un fenómeno llamado dispersión de Rayleigh, que ocurre cuando "
            "la luz solar interactúa con las moléculas de la atmósfera. Las longitudes de onda más "
            "cortas, como el azul y el violeta, se dispersan mucho más que las largas como el rojo. Como "
            "nuestros ojos son más sensibles al azul y parte del violeta se filtra en la atmósfera "
            "superior, percibimos el cielo como azul."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l14",
        "question": "Describe qué es un ataque de phishing y cómo protegerse.",
        "expected_long": (
            "Un ataque de phishing es una técnica de ingeniería social en la que un atacante se hace pasar "
            "por una entidad legítima para robar credenciales, datos personales o instalar malware. "
            "Comúnmente se distribuye por correo electrónico, mensajes o sitios web falsos. Para "
            "protegerse conviene verificar la URL, no hacer clic en enlaces sospechosos, activar la "
            "autenticación de dos factores y mantener la educación continua sobre nuevas técnicas."
        ),
        "difficulty": "easy",
    },
    {
        "id": "l15",
        "question": "¿Qué es el teorema CAP en sistemas distribuidos?",
        "expected_long": (
            "El teorema CAP establece que en un sistema distribuido de almacenamiento de datos es "
            "impossible garantizar simultáneamente consistencia, disponibilidad y tolerancia a "
            "particiones. Bajo una partición de red, el sistema debe elegir entre servir respuestas "
            "consistentes (posiblemente rechazando peticiones) o mantener la disponibilidad (a costa de "
            "servir datos posiblemente desactualizados). Esta compensación guía el diseño de bases de "
            "datos modernas."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l16",
        "question": "Explica brevemente qué es la fotosíntesis.",
        "expected_long": (
            "La fotosíntesis es el proceso bioquímico mediante el cual las plantas, algas y algunas "
            "bacterias convierten luz solar, agua y dióxido de carbono en glucosa y oxígeno. Ocurre "
            "principalmente en los cloroplastos, donde la clorofila captura la energía luminosa. Es "
            "fundamental para la vida en la Tierra porque produce el oxígeno que respiramos y es la "
            "base de casi todas las cadenas alimenticias."
        ),
        "difficulty": "easy",
    },
    {
        "id": "l17",
        "question": "¿Qué es un contenedor en el contexto de la ingeniería de software?",
        "expected_long": (
            "Un contenedor es una unidad estándar de software que empaqueta el código de una aplicación "
            "junto con sus dependencias, bibliotecas y configuración, de modo que se ejecute de forma "
            "consistente en distintos entornos. A diferencia de las máquinas virtuales, los contenedores "
            "comparten el kernel del sistema operativo y son mucho más ligeros. Docker y Kubernetes son "
            "las herramientas más populares para gestionar contenedores."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l18",
        "question": "Explica qué es la recursión y cuándo es preferible a un enfoque iterativo.",
        "expected_long": (
            "La recursión es una técnica de programación donde una función se llama a sí misma para "
            "resolver un problema dividiéndolo en subproblemas más pequeños del mismo tipo. Es preferible "
            "a un enfoque iterativo cuando la estructura del problema es naturalmente recursiva, como en "
            "el recorrido de árboles, algoritmos divide y vencerás o backtracking. Sin embargo, en "
            "problemas simples o de alto rendimiento suele preferirse la iteración para evitar el "
            "overhead y el riesgo de desbordamiento de pila."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l19",
        "question": "Describe qué es el ROI y por qué se usa en evaluación de proyectos.",
        "expected_long": (
            "El ROI (Return on Investment) es una métrica financiera que mide la rentabilidad de una "
            "inversión comparando el beneficio neto obtenido con el coste total invertido. Se calcula "
            "como (beneficio - coste) / coste, y se expresa habitualmente como porcentaje. Es útil en "
            "la evaluación de proyectos porque permite comparar de forma homogénea inversiones de "
            "diferente tamaño y duración, facilitando la toma de decisiones estratégicas."
        ),
        "difficulty": "easy",
    },
    {
        "id": "l20",
        "question": "¿Por qué algunos animales hibernan durante el invierno?",
        "expected_long": (
            "Algunos animales hibernan para sobrevivir periodos de escasez de alimento y temperaturas "
            "extremas. Durante la hibernación, su metabolismo se reduce drásticamente, su temperatura "
            "corporal desciende y dependen principalmente de las reservas de grasa acumuladas. Esto les "
            "permite conservar energía hasta que las condiciones ambientales mejoren. No todos los "
            "animales tienen la capacidad fisiológica de hibernar; es una adaptación evolucionada en "
            "ciertas especies."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l21",
        "question": "Explica qué es la computación cuántica y en qué se diferencia de la clásica.",
        "expected_long": (
            "La computación cuántica utiliza qubits, que pueden representar simultáneamente 0 y 1 "
            "gracias a la superposición, lo que permite explorar muchos estados en paralelo. También "
            "aprovecha el entrelazamiento, que correlaciona qubits para realizar operaciones coordinadas. "
            "A diferencia de la computación clásica, es especialmente potente para problemas como la "
            "simulación de moléculas, la optimización combinatoria y ciertos algoritmos criptográficos. "
            "Sin embargo, es muy sensible al ruido y a la decoherencia, lo que limita su escala actual."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l22",
        "question": "Explica brevemente qué es el Big Data y mencione sus tres V principales.",
        "expected_long": (
            "Big Data se refiere a conjuntos de datos cuyo volumen, velocidad o variedad exceden la "
            "capacidad de las herramientas tradicionales de procesamiento. Las tres V principales son: "
            "volumen, que describe la cantidad masiva de datos; velocidad, que mide la rapidez con que "
            "se generan y procesan; y variedad, que se refiere a los múltiples formatos (texto, imagen, "
            "vídeo, sensores). Su análisis permite extraer patrones y tomar decisiones basadas en "
            "evidencia a escalas antes impensables."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l23",
        "question": "¿Qué es un dataset balanceado y por qué importa en clasificación?",
        "expected_long": (
            "Un dataset está balanceado cuando las clases de la variable objetivo tienen una "
            "representación aproximadamente igual. Importa en clasificación porque los modelos tienden "
            "a favorecer la clase mayoritaria, lo que puede inflar métricas como la accuracy sin que "
            "el modelo sea realmente útil. Para corregirlo se usan técnicas como oversampling de la "
            "clase minoritaria, undersampling de la mayoritaria o el uso de pesos por clase en la "
            "función de pérdida."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l24",
        "question": "Explica la diferencia entre HTTP y HTTPS.",
        "expected_long": (
            "HTTP (HyperText Transfer Protocol) es el protocolo de comunicación entre clientes y "
            "servidores web en su forma básica, transmitiendo datos en texto plano. HTTPS es la versión "
            "segura que añade una capa de cifrado TLS/SSL, garantizando confidencialidad e integridad "
            "de los datos intercambiados. Hoy en día HTTPS es el estándar y los navegadores marcan los "
            "sitios sin cifrar como no seguros, lo que afecta también al posicionamiento SEO."
        ),
        "difficulty": "easy",
    },
    {
        "id": "l25",
        "question": "¿Qué es el gradiente descendente y por qué es central en el aprendizaje automático?",
        "expected_long": (
            "El gradiente descendente es un algoritmo de optimización iterativo que ajusta los "
            "parámetros de un modelo en la dirección opuesta al gradiente de la función de pérdida. "
            "Es central en el aprendizaje automático porque permite minimizar el error del modelo de "
            "forma eficiente, incluso en espacios de alta dimensión. Variantes como SGD, Adam o RMSProp "
            "difieren en cómo escalan y combinan los gradientes, pero comparten el mismo principio "
            "fundamental."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l26",
        "question": "Explica por qué los océanos son salados.",
        "expected_long": (
            "Los océanos son salados principalmente porque el agua de lluvia erosiona las rocas de la "
            "corteza terrestre, disolviendo sales minerales que son transportadas por los ríos hasta el "
            "mar. A lo largo de millones de años, este proceso ha concentrado iones como sodio, cloruro, "
            "magnesio y sulfato en el agua oceánica. La evaporación y la precipitación mantienen un "
            "equilibrio dinámico, pero la salinidad media se mantiene alrededor de 35 gramos por litro."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l27",
        "question": "¿Qué es la regresión lineal y cuándo es apropiado usarla?",
        "expected_long": (
            "La regresión lineal es un modelo estadístico que asume una relación lineal entre una "
            "variable dependiente y una o más variables independientes. Es apropiado cuando la relación "
            "subyacente es aproximadamente lineal, no hay multicolinealidad severa y los residuos son "
            "homocedásticos. Se valora por su interpretabilidad, pero no captura relaciones complejas, "
            "por lo que en problemas no lineales se prefieren modelos más flexibles como árboles o "
            "redes neuronales."
        ),
        "difficulty": "easy",
    },
    {
        "id": "l28",
        "question": "Explica qué es un ORM y por qué se usa en el desarrollo backend.",
        "expected_long": (
            "Un ORM (Object-Relational Mapping) es una técnica que permite mapear tablas de una base de "
            "datos relacional a objetos de un lenguaje de programación, abstrayendo las consultas SQL. "
            "Se usa en el desarrollo backend porque aumenta la productividad, mejora la mantenibilidad y "
            "reduce errores de inyección SQL al evitar concatenar strings. Ejemplos populares incluyen "
            "SQLAlchemy en Python, Hibernate en Java y Entity Framework en .NET."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l29",
        "question": "¿Qué es el sesgo de confirmación y cómo afecta al análisis de datos?",
        "expected_long": (
            "El sesgo de confirmación es la tendencia humana a buscar, interpretar y recordar "
            "información de forma que confirme nuestras creencias previas. En el análisis de datos puede "
            "llevar a seleccionar variables convenientes, ignorar evidencia contradictoria o mal "
            "interpretar correlaciones como causalidades. Para mitigarlo se recomienda definir hipótesis "
            "antes de explorar los datos, usar protocolos pre-registrados y someter los hallazgos a "
            "revisión por pares."
        ),
        "difficulty": "medium",
    },
    {
        "id": "l30",
        "question": "Describe qué es un test A/B y para qué se utiliza en productos digitales.",
        "expected_long": (
            "Un test A/B es un experimento controlado en el que se muestra aleatoriamente a los usuarios "
            "dos variantes (A y B) de un producto para comparar su comportamiento. Se utiliza para tomar "
            "decisiones de diseño basadas en datos, como probar el color de un botón, la ubicación de un "
            "elemento o el copy de un mensaje. Para que sea válido se requiere un tamaño de muestra "
            "suficiente, una métrica de éxito clara y una duración adecuada para captar variaciones "
            "cíclicas."
        ),
        "difficulty": "easy",
    },
    {
        "id": "l31",
        "question": "Explica qué es la criptografía de clave pública y por qué es importante.",
        "expected_long": (
            "La criptografía de clave pública usa un par de claves: una pública, que cualquiera puede "
            "conocer, y otra privada, que solo el propietario posee. Permite cifrar mensajes, firmar "
            "digitalmente y establecer canales seguros sin compartir secretos previamente. Es "
            "importante porque sustenta protocolos como HTTPS, las firmas digitales y las criptomonedas, "
            "haciendo posible la comunicación segura a escala global en internet."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l32",
        "question": "¿Qué es el aprendizaje por refuerzo y en qué problemas se aplica?",
        "expected_long": (
            "El aprendizaje por refuerzo es un paradigma de machine learning en el que un agente "
            "aprende a tomar decisiones interactuando con un entorno y recibiendo recompensas o castigos. "
            "A diferencia del supervisado, no requiere ejemplos etiquetados; aprende por ensayo y error. "
            "Se aplica con éxito en problemas como juegos (ajedrez, Go), robótica, optimización de "
            "tráfico, recomendación personalizada y control industrial. Algoritmos populares incluyen "
            "Q-learning, DQN y PPO."
        ),
        "difficulty": "hard",
    },
    {
        "id": "l33",
        "question": "Explica brevemente por qué se congelan los océanos en los polos pero no en el ecuador.",
        "expected_long": (
            "Los océanos se congelan en los polos y no en el ecuador principalmente por la diferencia "
            "de temperatura. En los polos la radiación solar llega con un ángulo bajo, distribuyéndose "
            "en una superficie mayor y atravesando más atmósfera, lo que reduce la energía recibida. "
            "En el ecuador los rayos inciden perpendicularmente, entregando mayor densidad de energía "
            "por unidad de área. Esto genera un gradiente térmico que mantiene los polos bajo cero "
            "mientras el ecuador permanece cálido."
        ),
        "difficulty": "easy",
    },
]


def build_dataset(seed: int = 42) -> list[dict[str, Any]]:
    """Return the full dataset with both `expected_short` and `expected_long` filled.

    For short rows the `expected_long` field is set to the short answer wrapped
    in a one-sentence justification, so the CROP pipeline always has a long
    target to work with if it ends up routed to a `long` row by mistake.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for r in _SHORT_ROWS:
        long_fallback = f"{r['expected_short']} (respuesta corta del conjunto de referencia)."
        rows.append(
            {
                "id": r["id"],
                "question": r["question"],
                "expected_short": r["expected_short"],
                "expected_long": long_fallback,
                "difficulty": r["difficulty"],
            }
        )
    for r in _LONG_ROWS:
        rows.append(
            {
                "id": r["id"],
                "question": r["question"],
                "expected_short": r["expected_long"].split(".")[0].strip(),
                "expected_long": r["expected_long"],
                "difficulty": r["difficulty"],
            }
        )
    rng.shuffle(rows)
    return rows


def write_dataset(path: Path | str, seed: int = 42) -> Path:
    """Write the dataset to disk as JSONL and return the path."""
    rows = build_dataset(seed=seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def load_dataset(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file previously written by `write_dataset`."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    # CLI: `python -m src.data_gen` writes the file.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/toy_qa.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    written = write_dataset(args.out, seed=args.seed)
    print(f"wrote {len(load_dataset(written))} rows to {written}")
