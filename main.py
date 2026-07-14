"""
Corrector de postura con OpenCV + MediaPipe Pose Landmarker.
Sentadilla y peso muerto se analizan siempre de costado (perfil):
se usan solo las articulaciones del lado del cuerpo mas visible para
la camara (punta del pie, tobillo, rodilla, cadera, hombro, cabeza y,
en el peso muerto, tambien la mano), que es lo que realmente se puede
ver bien parado de costado.

Se puede analizar en vivo con la camara o pasar un video ya grabado.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_utils import calculate_angle, facing_sign, midpoint, POSE_CONNECTIONS, JOINTS
from exercises import EXERCISES
import form_checks

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Cuanto pesa el frame nuevo al suavizar el angulo principal (0-1). Mas
# bajo = mas estable pero mas lento para reaccionar; mas alto = mas
# sensible pero mas jitter. Sirve para que el ruido de deteccion frame
# a frame no dispare cambios de estado o repeticiones falsas.
SMOOTHING_ALPHA = 0.4

# Cuadros seguidos que hacen falta para confirmar un cambio de estado
# (subida/bajada), para no contar repeticiones por un pico de ruido.
DEBOUNCE_FRAMES = 3

# Visibilidad promedio minima (0-1) de las articulaciones usadas para
# confiar en el frame. Si esta por debajo, no se actualiza nada y se
# le pide al usuario que se acomode.
VISIBILITY_THRESHOLD = 0.5

# Cuanto pesa cada frame nuevo al promediar la postura "neutra" de pie
# (cabeza-hombro-cadera) contra la que se compara la espalda del peso
# muerto. Bajo a proposito: es una referencia que se arma de a poco
# mientras la persona esta parada, no algo que deba saltar de golpe.
NEUTRAL_BACKNECK_ALPHA = 0.05
# Cuadros parado/arriba necesarios antes de confiar en esa referencia.
MIN_BASELINE_FRAMES = 15


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo de pose (solo pasa la primera vez)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Listo.")


def draw_skeleton(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def choose_side(landmarks, joint_names, current_side):
    """Elige que lado del cuerpo (0=izq, 1=der) usar segun visibilidad,
    con histeresis para no estar saltando de lado por ruido."""
    left_vis = sum(landmarks[JOINTS[j][0]].visibility for j in joint_names) / len(joint_names)
    right_vis = sum(landmarks[JOINTS[j][1]].visibility for j in joint_names) / len(joint_names)

    if current_side is None:
        current_side = 0 if left_vis >= right_vis else 1
    elif current_side == 0 and right_vis > left_vis + 0.15:
        current_side = 1
    elif current_side == 1 and left_vis > right_vis + 0.15:
        current_side = 0

    avg_vis = left_vis if current_side == 0 else right_vis
    return current_side, avg_vis


def draw_hud(frame, config, counter, feedback_lines, bx1, by1, bx2, by2, side_label, end_msg=None):
    box_h = 85 + 24 * max(1, len(feedback_lines))
    cv2.rectangle(frame, (0, 0), (380, box_h), (30, 30, 30), -1)
    cv2.putText(frame, config["name"], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {counter}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y = 80
    for text, is_good in feedback_lines:
        color = (0, 255, 0) if is_good else (0, 0, 255)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 24

    if side_label:
        cv2.putText(frame, f"Lado: {side_label}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    if end_msg:
        cv2.putText(frame, end_msg, (10, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 200), -1)
    cv2.putText(frame, "Cerrar", (bx1 + 15, by1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def run(exercise_key, video_path=None):
    config = EXERCISES[exercise_key]
    joint_names = config["joints"]
    ensure_model()

    is_video_file = video_path is not None
    source = video_path if is_video_file else 0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("No pude abrir la camara." if not is_video_file else f"No pude abrir el video: {video_path}")
        return

    # Los metadatos de fps de un video subido pueden venir en 0, NaN o
    # con valores absurdos (algunos codecs/celulares guardan cualquier
    # cosa ahi); sanitizamos para no dividir por 0 ni mandar timestamps
    # invalidos a mediapipe.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (0 < fps <= 240):
        fps = 30.0

    window_name = "Corrector de postura"
    state = {"close": False}
    layout = {"w": None, "h": None, "bx1": 0, "by1": 0, "bx2": 0, "by2": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and layout["bx1"] <= x <= layout["bx2"] and layout["by1"] <= y <= layout["by2"]:
            state["close"] = True

    def update_layout(frame):
        """Recalcula tamano de ventana y boton a partir del frame real
        (no de metadatos del video, que a veces no coinciden con lo que
        realmente se decodifica). Se llama de nuevo si la resolucion
        cambia entre frames, para no romper con videos de fuentes
        distintas."""
        fh, fw = frame.shape[:2]
        if fw == layout["w"] and fh == layout["h"]:
            return
        layout["w"], layout["h"] = fw, fh
        btn_w, btn_h = 110, 40
        layout["bx1"], layout["by1"] = fw - btn_w - 10, 10
        layout["bx2"], layout["by2"] = fw - 10, 10 + btn_h

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    counter = 0
    stage = "up"
    down_streak = 0
    up_streak = 0
    metrics = form_checks.initial_metrics(exercise_key)
    feedback_lines = [(config["setup_msg"], True)]
    current_side = None
    smoothed_angle = None
    neutral_backneck = None
    neutral_backneck_frames = 0
    start_time = time.time()
    frame_index = 0
    last_timestamp_ms = -1
    side_label = None
    last_frame = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if is_video_file and last_frame is not None:
                    end_msg = f"Fin del video. Repeticiones: {counter}. Presiona una tecla para cerrar."
                    draw_hud(last_frame, config, counter, feedback_lines,
                              layout["bx1"], layout["by1"], layout["bx2"], layout["by2"],
                              side_label, end_msg=end_msg)
                    cv2.imshow(window_name, last_frame)
                    cv2.waitKey(0)
                break

            try:
                if not is_video_file:
                    frame = cv2.flip(frame, 1)

                update_layout(frame)
                w, h = layout["w"], layout["h"]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                if is_video_file:
                    timestamp_ms = int(frame_index * 1000 / fps)
                else:
                    timestamp_ms = int((time.time() - start_time) * 1000)
                # mediapipe exige timestamps estrictamente crecientes; nos
                # aseguramos de eso pase lo que pase con el fps del video.
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                frame_index += 1

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                side_label = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    current_side, avg_vis = choose_side(landmarks, joint_names, current_side)
                    side_label = "izquierdo" if current_side == 0 else "derecho"
                    draw_skeleton(frame, landmarks, w, h)

                    if avg_vis < VISIBILITY_THRESHOLD:
                        feedback_lines = [("Reacomodate, no se ven bien las articulaciones de ese lado", False)]
                    else:
                        points = {
                            name: (landmarks[JOINTS[name][current_side]].x * w,
                                   landmarks[JOINTS[name][current_side]].y * h)
                            for name in joint_names
                        }
                        # linea hombro-cadera "centrada" (promedio de los
                        # dos lados) para el chequeo de espalda/cuello del
                        # peso muerto: es mas estable que usar solo el
                        # hombro/cadera del lado cercano a la camara.
                        points["shoulder_center"] = midpoint(
                            (landmarks[JOINTS["shoulder"][0]].x * w, landmarks[JOINTS["shoulder"][0]].y * h),
                            (landmarks[JOINTS["shoulder"][1]].x * w, landmarks[JOINTS["shoulder"][1]].y * h),
                        )
                        points["hip_center"] = midpoint(
                            (landmarks[JOINTS["hip"][0]].x * w, landmarks[JOINTS["hip"][0]].y * h),
                            (landmarks[JOINTS["hip"][1]].x * w, landmarks[JOINTS["hip"][1]].y * h),
                        )

                        rep_a, rep_b, rep_c = config["rep_angle"]
                        raw_angle = calculate_angle(points[rep_a], points[rep_b], points[rep_c])
                        smoothed_angle = raw_angle if smoothed_angle is None else (
                            SMOOTHING_ALPHA * raw_angle + (1 - SMOOTHING_ALPHA) * smoothed_angle
                        )

                        depth_a, depth_b, depth_c = config["depth_angle"]
                        depth_angle_value = calculate_angle(points[depth_a], points[depth_b], points[depth_c])

                        sign = facing_sign(points["heel"], points["toe"])

                        # --- maquina de estados (con debounce) para contar repeticiones ---
                        if smoothed_angle > config["angle_up"]:
                            down_streak = 0
                            up_streak += 1
                            if up_streak >= DEBOUNCE_FRAMES and stage == "down":
                                counter += 1
                                feedback_lines = form_checks.evaluate_rep(exercise_key, config, metrics)
                                metrics = form_checks.initial_metrics(exercise_key)
                                stage = "up"
                        elif smoothed_angle < config["angle_down"]:
                            up_streak = 0
                            down_streak += 1
                            if down_streak >= DEBOUNCE_FRAMES:
                                stage = "down"
                        else:
                            down_streak = 0
                            up_streak = 0

                        baseline = neutral_backneck if neutral_backneck_frames >= MIN_BASELINE_FRAMES else None

                        if exercise_key == "deadlift":
                            backneck_now = calculate_angle(points["ear"], points["shoulder_center"], points["hip_center"])

                            # Postura neutra de referencia (parado/arriba) para
                            # detectar cuanto se redondea la espalda respecto
                            # de como esta esa persona en particular, no solo
                            # contra un numero fijo que le puede quedar corto
                            # a alguien que ya de por si esta encorvado parado.
                            if stage == "up":
                                neutral_backneck_frames += 1
                                neutral_backneck = backneck_now if neutral_backneck is None else (
                                    NEUTRAL_BACKNECK_ALPHA * backneck_now + (1 - NEUTRAL_BACKNECK_ALPHA) * neutral_backneck
                                )

                            # MediaPipe no tiene puntos de columna: la linea
                            # oreja-hombro-cadera se dibuja siempre recta, no
                            # importa cuanto se arquee la espalda de verdad.
                            # Para que igual se note en el momento (no solo en
                            # el resumen de la repeticion), la pintamos de
                            # rojo apenas se detecta redondeo.
                            if form_checks.is_back_rounded(backneck_now, baseline):
                                ear_px = (int(points["ear"][0]), int(points["ear"][1]))
                                sc_px = (int(points["shoulder_center"][0]), int(points["shoulder_center"][1]))
                                hc_px = (int(points["hip_center"][0]), int(points["hip_center"][1]))
                                cv2.line(frame, hc_px, sc_px, (0, 0, 255), 4)
                                cv2.line(frame, sc_px, ear_px, (0, 0, 255), 4)
                                cv2.putText(frame, "ESPALDA REDONDEADA", (sc_px[0] + 15, sc_px[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        metrics = form_checks.update_metrics(
                            exercise_key, metrics, points, sign, stage, smoothed_angle, depth_angle_value,
                            neutral_backneck=baseline,
                        )

                        p_b = points[rep_b]
                        cv2.putText(frame, f"{int(smoothed_angle)} deg", (int(p_b[0]) + 10, int(p_b[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    feedback_lines = [("No te veo bien, acomodate de costado a la camara", False)]

                last_frame = frame
                draw_hud(frame, config, counter, feedback_lines,
                         layout["bx1"], layout["by1"], layout["bx2"], layout["by2"], side_label)
                cv2.imshow(window_name, frame)
            except Exception as exc:
                # Un frame corrupto o una rareza puntual de un video no
                # tiene que tirar abajo todo el analisis: lo salteamos y
                # seguimos con el siguiente. Igual procesamos eventos de
                # la ventana (tecla/boton) para que no quede colgada si
                # esto se repite en varios frames seguidos.
                print(f"Aviso: no pude procesar el frame {frame_index} ({exc}), sigo con el siguiente.")
                if cv2.waitKey(1) & 0xFF == ord('q') or state["close"]:
                    break
                continue

            wait_ms = 1 if not is_video_file else max(1, int(1000 / fps))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q') or state["close"]:
                break

    cap.release()
    cv2.destroyAllWindows()


def choose_source():
    print("\nFuente de video:")
    print("1 - Camara web")
    print("2 - Archivo de video")
    src_choice = input("Opcion: ").strip()
    if src_choice == "2":
        path = input("Ruta del video: ").strip().strip('"')
        if not os.path.exists(path):
            print(f"No encontre el archivo '{path}', uso la camara.")
            return None
        return path
    return None


if __name__ == "__main__":
    print("Elegi el ejercicio:")
    print("1 - Sentadilla (de costado)")
    print("2 - Peso muerto (de costado)")
    choice = input("Opcion: ").strip()
    exercise_key = {"1": "squat", "2": "deadlift"}.get(choice, "squat")

    video_path = choose_source()
    run(exercise_key, video_path=video_path)
