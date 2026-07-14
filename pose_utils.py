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
