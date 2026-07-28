"""
Configuracion de cada ejercicio. Los dos se analizan de costado
(perfil), por eso solo se usan las articulaciones del lado de la
camara que este mas visible (son menos puntos que de frente, pero
son los que realmente se ven bien desde ese angulo).

Campos:
- joints: articulaciones genericas que hacen falta pedir (se resuelven
  al lado izq/der segun visibilidad).
- rep_angle: (a, b, c) angulo (vertice en b) que maneja la maquina de
  estados de repeticiones (sube/baja).
- angle_up / angle_down: umbrales de rep_angle para contar y detectar
  el inicio del descenso.
- depth_angle: (a, b, c) angulo cuyo minimo durante la bajada define
  si la profundidad/flexion estuvo bien.
- depth_range: rango bueno para el minimo de depth_angle.
- posture_checks: chequeos biomecanicos extra (ver form_checks.py)
  evaluados ademas de la profundidad.
"""

EXERCISES = {
    "squat": {
        "name": "Sentadilla (de costado)",
        "joints": ("ankle", "knee", "hip", "shoulder", "ear", "heel", "toe"),
        "rep_angle": ("hip", "knee", "ankle"),
        "angle_up": 160,
        "angle_down": 140,
        "depth_angle": ("hip", "knee", "ankle"),
        "depth_range": (70, 100),
        "too_shallow_msg": "Baja mas la cadera",
        "too_deep_msg": "Ojo, estas bajando de mas",
        "good_msg": "Buena profundidad!",
        "setup_msg": "Parate de costado a la camara: se tiene que ver el pie, el tobillo, la rodilla, la cadera, los hombros y la cabeza",
    },
    "deadlift": {
        "name": "Peso muerto (de costado)",
        "joints": ("ankle", "knee", "hip", "shoulder", "ear", "wrist", "heel", "toe"),
        "rep_angle": ("shoulder", "hip", "knee"),
        "angle_up": 160,
        "angle_down": 140,
        "depth_angle": ("hip", "knee", "ankle"),
        # Rango amplio a proposito: el peso muerto convencional dobla
        # bastante mas la rodilla que el rumano/piernas rigidas, y los
        # dos son validos. Medido sobre video, la rodilla abajo da ~110
        # grados en convencional y ~132 en rumano; el rango anterior
        # (150-172) marcaba como error las dos variantes bien hechas.
        # Fuera de este rango si cambia el ejercicio: por debajo es una
        # sentadilla, por arriba es piernas bloqueadas.
        "depth_range": (100, 172),
        "too_shallow_msg": "Flexiona un poco mas las rodillas, no las bloquees",
        "too_deep_msg": "Estas flexionando de mas las rodillas, eso es mas sentadilla que peso muerto",
        "good_msg": "Buena flexion de piernas!",
        "setup_msg": "Parate de costado a la camara: se tiene que ver el pie, el tobillo, la rodilla, la cadera, los hombros, la cabeza y el brazo",
    },
}
