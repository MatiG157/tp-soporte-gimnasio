"""
Chequeos de forma especificos de cada ejercicio, ademas del angulo
de profundidad basico. Todas las distancias se normalizan contra una
medida propia del cuerpo (largo de pierna o de torso) para que los
umbrales no dependan de que tan lejos este la persona de la camara.

Sobre el chequeo de espalda en peso muerto
------------------------------------------
La espalda redondeada se detecta con la SILUETA de la persona (mascara
de segmentacion), no con los landmarks: ver back_bow() en pose_utils.

Por que no alcanza con los landmarks: MediaPipe no da puntos de columna,
y lo unico parecido es la linea oreja-hombro-cadera, que en realidad
mide si la cabeza acompaña al torso. Eso no es redondeo de espalda.
Medido sobre video real, un peso muerto con la espalda claramente
redondeada dio un angulo MAS alto (mediana 171 grados) que uno bien
hecho (162), porque al redondear la cabeza sigue la curva y los tres
puntos quedan alineados igual. Por eso ese angulo quedo solo como
chequeo de cabeza/cuello, con umbral conservador, y el redondeo de
espalda se mide aparte con la silueta.

El abombamiento de la silueta si separa bien (medido sobre los videos
de prueba, mediana por repeticion):

    espalda plana            0.020
    espalda redondeada       0.062
    espalda muy redondeada   0.108

Limitacion: la mascara sigue el contorno de la ropa. Un buzo holgado
suma volumen en la espalda alta y puede inflar la medicion, asi que
conviene filmar con ropa ajustada.
"""

from pose_utils import calculate_angle, distance, midpoint, forward_offset

GOOD = True
BAD = False

# Umbrales (fracciones del largo de pierna/torso usado como referencia).
SQUAT_KNEE_OVER_TOE = 0.12
SQUAT_SHOULDER_MIDFOOT_DEV = 0.18

# Por debajo de este angulo de cadera consideramos que la persona esta
# en la parte cargada del movimiento (inclinada sobre la barra). Los
# chequeos de espalda y de posicion de barra solo miran esos cuadros:
# fuera de ahi la geometria es otra y solo aporta falsos positivos.
DEADLIFT_HINGE_ANGLE = 120

# Cuanto puede caer (en grados) el angulo cabeza-hombro-cadera respecto
# de la postura neutra de pie de la persona antes de avisar. Medido
# sobre video: con tecnica correcta la caida sostenida no pasa de ~20
# grados (peor caso 28 en un pico aislado de 4 cuadros).
DEADLIFT_BACKNECK_DROP = 28
# Cuadros SEGUIDOS que tienen que superar esa caida para avisar.
DEADLIFT_BACKNECK_FRAMES = 5

# Abombamiento de la silueta de la espalda (fraccion del largo del torso,
# ver back_bow en pose_utils). Medianas medidas sobre video: 0.020 con
# espalda plana, 0.062 redondeada y 0.108 muy redondeada. Los umbrales
# van en el medio de esos grupos.
DEADLIFT_BACK_BOW = 0.045
DEADLIFT_BACK_BOW_SEVERE = 0.085
# Por encima de esto la medicion es imposible (la espalda no se arquea
# medio torso): es un cuadro con la mascara o los landmarks rotos.
DEADLIFT_BACK_BOW_MAX_VALID = 0.35
# Cuadros medidos como minimo antes de opinar sobre la espalda.
DEADLIFT_BACK_BOW_MIN_SAMPLES = 5

# El lockout se mide sobre el angulo de cadera sin suavizar: el suavizado
# recorta los picos y hacia que este chequeo saltara aun con la cadera
# bien extendida. Tiene que ser >= angle_up del ejercicio, si no la
# repeticion se cuenta pero igual se avisa que falto extension.
DEADLIFT_LOCKOUT_MIN_ANGLE = 165

# Offset hombro-muñeca (fraccion del largo del torso, positivo = hombro
# por delante de la barra), medido como mediana de la parte cargada.
# Con los hombros detras de la barra la espalda trabaja en desventaja.
DEADLIFT_BAR_OFFSET_MIN = 0.0
DEADLIFT_BAR_OFFSET_MAX = 0.45


def _percentile(values, q):
    """Percentil q (0-100) de una lista, por interpolacion lineal.

    Se usa en vez del minimo crudo porque el minimo lo define un solo
    cuadro y basta un mal frame de deteccion para arruinar la medicion
    de toda la repeticion.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _median(values):
    return _percentile(values, 50)


def backneck_drop(backneck_angle, neutral_backneck):
    """Cuanto cayo el angulo cabeza-hombro-cadera respecto de la postura
    neutra de pie de la persona. 0 si todavia no hay neutra medida."""
    if neutral_backneck is None:
        return 0.0
    return neutral_backneck - backneck_angle


def initial_metrics(exercise_key):
    if exercise_key == "squat":
        return {
            "min_depth_angle": 180.0,
            "max_knee_forward": float("-inf"),
            "max_shoulder_dev": 0.0,
        }
    if exercise_key == "deadlift":
        return {
            "depth_samples": [],
            "back_rounded": False,
            "backneck_streak": 0,
            "backneck_worst_drop": 0.0,
            "max_hip_angle": 0.0,
            "bar_offsets": [],
            "back_bows": [],
        }
    return {}


def rounded_back_level(bows):
    """0 = espalda bien, 1 = redondeada, 2 = muy redondeada.

    Usa la mediana de la parte cargada: cuadros sueltos con la mascara
    mal segmentada dan valores absurdos y no tienen que decidir nada.
    """
    usable = [b for b in bows if b is not None and b <= DEADLIFT_BACK_BOW_MAX_VALID]
    if len(usable) < DEADLIFT_BACK_BOW_MIN_SAMPLES:
        return 0
    bow = _median(usable)
    if bow > DEADLIFT_BACK_BOW_SEVERE:
        return 2
    if bow > DEADLIFT_BACK_BOW:
        return 1
    return 0


def update_metrics(exercise_key, metrics, points, facing_sign, stage, rep_angle_value, depth_angle_value,
                    neutral_backneck=None, rep_angle_raw=None, back_bow_value=None):
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
        # El lockout se juzga sobre el angulo crudo (el suavizado recorta
        # el pico y hace que parezca que nunca se extendio del todo).
        lockout_angle = rep_angle_raw if rep_angle_raw is not None else rep_angle_value
        metrics["max_hip_angle"] = max(metrics["max_hip_angle"], lockout_angle)

        # Parte cargada del movimiento: solo ahi tienen sentido los
        # chequeos de espalda y de barra.
        in_hinge = rep_angle_value < DEADLIFT_HINGE_ANGLE

        # Redondeo de espalda: solo en la parte cargada, donde la
        # geometria del rayo perpendicular tiene sentido.
        if in_hinge and back_bow_value is not None:
            metrics["back_bows"].append(back_bow_value)

        if in_hinge and neutral_backneck is not None:
            backneck_angle = calculate_angle(points["ear"], points["shoulder_center"], points["hip_center"])
            drop = backneck_drop(backneck_angle, neutral_backneck)
            if drop > DEADLIFT_BACKNECK_DROP:
                metrics["backneck_streak"] += 1
                metrics["backneck_worst_drop"] = max(metrics["backneck_worst_drop"], drop)
                if metrics["backneck_streak"] >= DEADLIFT_BACKNECK_FRAMES:
                    metrics["back_rounded"] = True
            else:
                metrics["backneck_streak"] = 0
        else:
            metrics["backneck_streak"] = 0

        if stage == "down":
            metrics["depth_samples"].append(depth_angle_value)
            if in_hinge:
                torso_len = distance(points["shoulder"], points["hip"]) or 1.0
                offset = forward_offset(points["wrist"], points["shoulder"], facing_sign)
                metrics["bar_offsets"].append(offset / torso_len)

    return metrics


def evaluate_rep(exercise_key, config, metrics, truncated=False):
    """Devuelve una lista de (mensaje, GOOD/BAD) para la repeticion recien contada.

    truncated: el video corto antes de terminar de medir la subida. En ese
    caso no se opina sobre el lockout, porque la extension final de cadera
    quedo sin registrar y marcaria falta de extension en repeticiones que
    en realidad terminan bien.
    """
    lines = []
    depth_min, depth_max = config["depth_range"]

    if exercise_key == "deadlift":
        # Percentil bajo en vez del minimo: mas estable frente a un
        # cuadro suelto mal detectado.
        depth = _percentile(metrics["depth_samples"], 10)
    else:
        depth = metrics["min_depth_angle"]

    if depth is None:
        lines.append(("No llegue a medir la flexion de piernas", BAD))
    elif depth < depth_min:
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
        level = rounded_back_level(metrics["back_bows"])
        if level == 2:
            lines.append(("Espalda MUY redondeada, para y baja el peso", BAD))
        elif level == 1:
            lines.append(("Espalda redondeada, saca pecho y traba la espalda", BAD))

        # Aviso aparte: la cabeza puede caer con la espalda bien.
        if metrics["back_rounded"]:
            lines.append(("Cabeza y cuello alineados con la espalda", BAD))

        if not truncated and metrics["max_hip_angle"] < DEADLIFT_LOCKOUT_MIN_ANGLE:
            lines.append(("Termina de extender bien la cadera arriba", BAD))

        # Mediana de la parte cargada: un solo cuadro (antes se usaba el
        # del angulo minimo de cadera) es demasiado ruidoso.
        offset = _median(metrics["bar_offsets"])
        if offset is not None:
            if offset < DEADLIFT_BAR_OFFSET_MIN:
                lines.append(("Lleva los hombros un poco mas adelante de la barra", BAD))
            elif offset > DEADLIFT_BAR_OFFSET_MAX:
                lines.append(("No adelantes tanto los hombros respecto de la barra", BAD))

    return lines
