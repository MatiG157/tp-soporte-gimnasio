# tp-soporte-gimnasio

Corrector de postura para gimnasio con visión por computadora. Analiza
**sentadilla** y **peso muerto** filmados de costado (perfil), cuenta las
repeticiones y da feedback de técnica en tiempo real sobre el video.

## Qué hace

- Detecta el cuerpo con MediaPipe Pose (BlazePose, 33 puntos) y dibuja el
  esqueleto sobre la imagen.
- Elige automáticamente el lado del cuerpo más visible para la cámara
  (clave estando de costado, porque el lado de atrás queda tapado).
- Cuenta repeticiones con una máquina de estados suavizada, para no contar
  de más por ruido de detección.
- Al terminar cada repetición evalúa la técnica y muestra avisos:
  - **Sentadilla:** profundidad, rodilla que pasa la punta del pie,
    alineación de hombros con el medio del pie.
  - **Peso muerto:** **espalda redondeada** (en dos niveles: redondeada
    y muy redondeada), flexión de rodillas, extensión de cadera arriba
    (lockout), posición de los hombros respecto de la barra y alineación
    de cabeza y cuello. Si el intento nunca sube, avisa igual al
    terminar, aunque no llegue a contar como repetición.

## Requisitos

- Python 3.9+
- Paquetes:

```bash
pip install opencv-python mediapipe numpy
```

El modelo `pose_landmarker_lite.task` se descarga solo la primera vez si no
está presente (ya viene incluido en el repo).

## Cómo usarlo

```bash
python main.py
```

1. Elegí el ejercicio: `1` sentadilla, `2` peso muerto.
2. Elegí la fuente: `1` cámara web, `2` archivo de video (te pide la ruta).
3. Parate **de costado** a la cámara. Se tiene que ver el pie, tobillo,
   rodilla, cadera, hombros y cabeza (en peso muerto, también el brazo).
4. Para cerrar: botón "Cerrar", tecla `q`, o esperá al fin del video.

## Estructura

| Archivo | Rol |
|---|---|
| `main.py` | Loop principal: captura, detección de pose, dibujo, HUD y conteo de reps. |
| `exercises.py` | Configuración de cada ejercicio (ángulos, umbrales y mensajes). |
| `form_checks.py` | Chequeos de técnica específicos de cada ejercicio. |
| `pose_utils.py` | Cálculo de ángulos/distancias y mapeo de articulaciones izq/der. |
| `pose_landmarker_lite.task` | Modelo de MediaPipe Pose. |

## Limitaciones conocidas

- La espalda redondeada se mide con la **silueta** (máscara de
  segmentación), no con los landmarks: se compara el contorno de la
  espalda contra la recta hombro–cadera y se mide cuánto se arquea.
  Los landmarks no sirven para esto porque MediaPipe no da puntos de
  columna: la línea oreja–hombro–cadera mide si la cabeza sigue al
  torso, no la curvatura. Contrastado contra video real, un peso muerto
  con la espalda claramente redondeada dio un ángulo *más alto* (mediana
  171°) que uno bien hecho (162°), porque al redondear la cabeza
  acompaña la curva y la línea queda recta igual.
- Como la máscara sigue el contorno de la **ropa**, un buzo holgado suma
  volumen en la espalda alta y puede inflar la medición. Conviene filmar
  con ropa ajustada.
- El análisis asume vista de **perfil**; de frente o en ángulos raros la
  detección de técnica pierde precisión.
- El rango de flexión de rodilla del peso muerto es amplio a propósito
  (100°–172°) para aceptar tanto convencional como rumano. Eso lo hace
  tolerante: distingue "esto es una sentadilla" o "tenés las piernas
  bloqueadas", no matices finos entre variantes.
