"""
Funciones auxiliares: calculo de angulos entre articulaciones,
definicion de landmarks/conexiones del esqueleto, y mapeo de
articulaciones genericas (hip, knee, etc.) a sus indices izquierdo/derecho.

Nota: uso los indices numericos directos (0-32) en vez del enum
mp.solutions.pose.PoseLandmark porque esa API "vieja" puede no
estar disponible en versiones recientes de mediapipe. Los indices
son siempre los mismos (33 landmarks, mismo modelo BlazePose por
debajo), asi que hardcodearlos es mas robusto.
"""

import numpy as np

NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Mapea un nombre generico de articulacion a (indice_izquierdo, indice_derecho).
# Sirve para elegir dinamicamente que lado del cuerpo usar segun cual
# este mas visible para la camara (clave cuando la persona esta de costado,
# porque el lado "de atras" queda tapado y solo conviene usar el lado cercano).
JOINTS = {
    "ear": (LEFT_EAR, RIGHT_EAR),
    "shoulder": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "elbow": (LEFT_ELBOW, RIGHT_ELBOW),
    "wrist": (LEFT_WRIST, RIGHT_WRIST),
    "hip": (LEFT_HIP, RIGHT_HIP),
    "knee": (LEFT_KNEE, RIGHT_KNEE),
    "ankle": (LEFT_ANKLE, RIGHT_ANKLE),
    "heel": (LEFT_HEEL, RIGHT_HEEL),
    "toe": (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
}

POSE_CONNECTIONS = [
    (LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT_INDEX), (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT_INDEX), (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
]


def calculate_angle(a, b, c):
    """
    Calcula el angulo en grados formado en el punto b, entre los
    segmentos b-a y b-c. a, b, c son tuplas o listas (x, y) en pixeles.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (np.arctan2(c[1] - b[1], c[0] - b[0])
               - np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def distance(a, b):
    """Distancia euclidea en pixeles entre dos puntos (x, y)."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def midpoint(a, b):
    """Punto medio entre dos puntos (x, y)."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def facing_sign(heel, toe):
    """
    Devuelve 1.0 si la persona mira/avanza hacia +x (punta del pie a la
    derecha del talon) o -1.0 si mira hacia -x. Sirve para saber, estando
    de costado, cual es la direccion "hacia adelante" del cuerpo sin
    depender de si la persona se paro mirando a izquierda o derecha.
    """
    return 1.0 if toe[0] >= heel[0] else -1.0


def forward_offset(from_point, to_point, sign):
    """
    Distancia (con signo) en el eje x entre dos puntos, medida en la
    direccion "hacia adelante" del cuerpo. Positivo = to_point esta
    por delante de from_point.
    """
    return (to_point[0] - from_point[0]) * sign


def back_bow(mask, shoulder, hip, knee, samples=24, max_depth=0.6):
    """
    Mide cuanto se arquea el contorno de la espalda respecto de la linea
    recta hombro-cadera, usando la mascara de segmentacion de la persona.

    Los landmarks de pose no tienen puntos de columna, asi que el
    redondeo de espalda no se puede sacar de ellos. La silueta si lo
    muestra: con la espalda neutra el borde de la espalda va practicamente
    recto y paralelo a la linea hombro-cadera, y cuando se redondea ese
    borde se abomba hacia afuera.

    Como se calcula: se recorre la linea hombro-cadera y desde cada punto
    se tira un rayo perpendicular hacia el lado de la espalda (el opuesto
    a la rodilla, que queda del lado de la panza) hasta salir de la
    mascara. Eso da el perfil del borde de la espalda. Al perfil se le
    resta la recta que une sus propios extremos, de modo que lo que queda
    es la CURVATURA y no el grosor del torso ni la inclinacion. El
    resultado se divide por el largo del torso para que no dependa de la
    distancia a la camara ni del tamaño de la persona.

    Devuelve la flecha del arco como fraccion del largo del torso
    (0 = recta, mas alto = mas redondeada), o None si no se pudo medir.
    """
    s = np.asarray(shoulder, dtype=float)
    h_pt = np.asarray(hip, dtype=float)
    k = np.asarray(knee, dtype=float)

    chord = h_pt - s
    torso_len = float(np.hypot(chord[0], chord[1]))
    if torso_len < 25:
        return None

    along = chord / torso_len
    normal = np.array([-along[1], along[0]])
    # La espalda esta del lado opuesto a la rodilla.
    if float(np.dot(k - s, normal)) > 0:
        normal = -normal

    height, width = mask.shape
    depth = int(torso_len * max_depth)
    if depth < 5:
        return None

    t = np.linspace(0.0, 1.0, samples + 1)
    base = s[None, :] + chord[None, :] * t[:, None]
    steps = np.arange(depth, dtype=float)
    pts = base[:, None, :] + normal[None, None, :] * steps[None, :, None]

    xs = np.clip(np.rint(pts[..., 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(pts[..., 1]).astype(int), 0, height - 1)
    in_image = ((pts[..., 0] >= 0) & (pts[..., 0] < width)
                & (pts[..., 1] >= 0) & (pts[..., 1] < height))
    in_body = (mask[ys, xs] >= 0.5) & in_image

    # Solo sirven los rayos que arrancan dentro del cuerpo y que llegan a
    # salir: si el rayo nunca sale, el borde quedo fuera del alcance y la
    # medicion de ese punto no es confiable.
    exited = ~in_body
    usable = in_body[:, 0] & exited.any(axis=1)
    if usable.sum() < samples * 0.7:
        return None

    edge = np.argmax(exited, axis=1).astype(float)[usable]
    t_ok = t[usable]

    inner = (t_ok >= 0.12) & (t_ok <= 0.88)
    if inner.sum() < 5:
        return None

    # Recta entre los extremos del propio perfil: aisla la curvatura.
    start = float(np.interp(0.12, t_ok, edge))
    end = float(np.interp(0.88, t_ok, edge))
    straight = start + (end - start) * (t_ok - 0.12) / 0.76

    return float(np.max((edge - straight)[inner])) / torso_len
