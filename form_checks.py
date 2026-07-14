"""
Chequeos de forma especificos de cada ejercicio, ademas del angulo
de profundidad basico. Todas las distancias se normalizan contra una
medida propia del cuerpo (largo de pierna o de torso) para que los
umbrales no dependan de que tan lejos este la persona de la camara.

MediaPipe no da puntos de la columna, asi que "espalda recta" se
aproxima con el punto de la oreja: si la cabeza se mantiene alineada
con la linea hombro-cadera, la espalda alta y el cuello se estan
manteniendo rectos; si esa alineacion se rompe (la cabeza "cae"
respecto de esa linea), es senal de que la espalda se esta
redondeando. Para que esa linea hombro-cadera sea mas fiel a la
columna real (y no al hombro/cadera del lado cercano a la camara,
que se desplaza lateralmente segun como este rotado el torso), se usa
el punto medio entre hombro izquierdo y derecho, y entre cadera
izquierda y derecha (shoulder_center / hip_center), en vez de un solo
lado.

Ademas, en vez de comparar solo contra un umbral fijo (que no detecta
a alguien cuya postura de pie "normal" ya es medio encorvada, ni a
alguien que arranca con buena pinta pero se redondea de mas al bajar),
se compara contra la postura de pie de la propia persona: mientras
esta arriba/parada se va promediando su angulo cabeza-hombro-cadera
como "postura neutra", y despues se controla cuanto se aleja de esa
referencia durante la bajada.
"""

from pose_utils import calculate_angle, distance, midpoint, forward_offset

GOOD = True
BAD = False

# Umbrales (fracciones del largo de pierna/torso usado como referencia).
SQUAT_KNEE_OVER_TOE = 0.12
SQUAT_SHOULDER_MIDFOOT_DEV = 0.18

# Espalda/cuello: si se aleja mas de esto (en grados) de la postura
# neutra de pie de la persona, se considera que se esta redondeando.
DEADLIFT_BACKNECK_DROP = 15
# Piso absoluto: por debajo de esto se considera espalda redondeada
# incluso sin una postura neutra de referencia (p. ej. el video arranca
# ya agachado). Calibrado contra fotos reales de espalda claramente
# arqueada/redondeada en el arranque del peso muerto, que dan angulos
# cabeza-hombro-cadera de entre 130 y 145 grados.
DEADLIFT_BACKNECK_ABS_FLOOR = 155

DEADLIFT_LOCKOUT_MIN_ANGLE = 170
DEADLIFT_BAR_OFFSET_MIN = 0.02
DEADLIFT_BAR_OFFSET_MAX = 0.30


def is_back_rounded(backneck_angle, neutral_backneck=None):
    """True si el angulo actual cabeza-hombro-cadera indica espalda
    redondeada: o esta directamente por debajo del piso absoluto, o
    cayo demasiado respecto de la postura neutra de la persona (si ya
    la conocemos)."""
    if backneck_angle < DEADLIFT_BACKNECK_ABS_FLOOR:
        return True
    if neutral_backneck is not None and (neutral_backneck - backneck_angle) > DEADLIFT_BACKNECK_DROP:
        return True
    return False


def initial_metrics(exercise_key):
    if exercise_key == "squat":
        return {
            "min_depth_angle": 180.0,
            "max_knee_forward": float("-inf"),
            "max_shoulder_dev": 0.0,
        }
    if exercise_key == "deadlift":
        return {
            "min_depth_angle": 180.0,
            "back_rounded": False,
            "max_hip_angle": 0.0,
            "min_hip_angle_seen": 180.0,
            "bar_offset_at_bottom": None,
        }
    return {}


def update_metrics(exercise_key, metrics, points, facing_sign, stage, rep_angle_value, depth_angle_value,
                    neutral_backneck=None):
    if exercise_key == "squat":
        if stage == "down":
            metrics["min_depth_angle"] = min(metrics["min_depth_angle"], depth_angle_value)

            leg_len = distance(points["hip"], points["ankle"]) or 1.0
            forward = forward_offset(points["toe"], points["knee"], facing_sign)
            metrics["max_knee_forward"] = max(metrics["max_knee_forward"], forward / leg_len)

            midfoot = midpoint(points["ankle"], points["toe"])
            dev = abs(points["shoulder"][0] - midfoot[0]) / leg_len
            metrics["max_shoulder_dev"] = max(metrics["max_shoulder_dev"], dev)

    elif exercise_key == "deadlift":
        backneck_angle = calculate_angle(points["ear"], points["shoulder_center"], points["hip_center"])
        if is_back_rounded(backneck_angle, neutral_backneck):
            metrics["back_rounded"] = True
        metrics["max_hip_angle"] = max(metrics["max_hip_angle"], rep_angle_value)

        if stage == "down":
            metrics["min_depth_angle"] = min(metrics["min_depth_angle"], depth_angle_value)

            if rep_angle_value < metrics["min_hip_angle_seen"]:
                metrics["min_hip_angle_seen"] = rep_angle_value
                torso_len = distance(points["shoulder"], points["hip"]) or 1.0
                offset = forward_offset(points["wrist"], points["shoulder"], facing_sign)
                metrics["bar_offset_at_bottom"] = offset / torso_len

    return metrics


def evaluate_rep(exercise_key, config, metrics):
    """Devuelve una lista de (mensaje, GOOD/BAD) para la repeticion recien contada."""
    lines = []
    depth = metrics["min_depth_angle"]
    depth_min, depth_max = config["depth_range"]
    if depth < depth_min:
        lines.append((config["too_deep_msg"], BAD))
    elif depth > depth_max:
        lines.append((config["too_shallow_msg"], BAD))
    else:
        lines.append((config["good_msg"], GOOD))

    if exercise_key == "squat":
        if metrics["max_knee_forward"] > SQUAT_KNEE_OVER_TOE:
            lines.append(("La rodilla pasa la punta del pie, llevala mas atras", BAD))
        if metrics["max_shoulder_dev"] > SQUAT_SHOULDER_MIDFOOT_DEV:
            lines.append(("Alinea los hombros con el medio del pie", BAD))

    elif exercise_key == "deadlift":
        if metrics["back_rounded"]:
            lines.append(("Espalda y cuello rectos, se estan redondeando", BAD))

        if metrics["max_hip_angle"] < DEADLIFT_LOCKOUT_MIN_ANGLE:
            lines.append(("Termina de extender bien la cadera arriba", BAD))

        offset = metrics["bar_offset_at_bottom"]
        if offset is not None:
            if offset < DEADLIFT_BAR_OFFSET_MIN:
                lines.append(("Lleva los hombros un poco mas adelante de la barra", BAD))
            elif offset > DEADLIFT_BAR_OFFSET_MAX:
                lines.append(("No adelantes tanto los hombros respecto de la barra", BAD))

    return lines
