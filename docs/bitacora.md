# Bitácora

## 26/09/2026 - 1.1 Selección de cámara

- **Fuente:** SkylineWebcams. El stream es HLS (`.m3u8`) a 1920x1080. Su URL se obtiene desde DevTools -> Network y lleva un token de sesión.

- **Cámara descartada:** Napoli (Vesuvio). Pasan muchas personas para contarlas a mano y YOLO no es capaz de detectarlas todas (de noche detecta menos).

- **Hallazgos con YOLO:** Al subir la resolución a la que trabaja YOLO (de 640 a 1920) aumentan las detecciones (ver screenshots).

- **Decisión:** Buscar una escena más tranquila. `https://www.skylinewebcams.com/es/webcam/espana/andalucia/cadiz/grazalema.html` es buena candidata, habría que probarla de día.

## 27/09/2026 - 1.2 Captura de frames

- Función `capture_frame(url)` que abre el stream, lee un frame y lo cierra en cada llamada para evitar frames antiguos del buffer de OpenCV. Devuelve bytes JPEG con calidad configurable.

- Errores mediante `CaptureError`.

- Timeouts de apertura (10s) y de lectura (5s), ya que por defecto FFmpeg espera hasta 30s.

- **Token automático:** El token del stream está en el HTML de la página. La función `resolve_stream_url()` descarga la página, lo extrae y construye la URL del stream. En el .env se guarda la URL de la página, no la del stream.

- **Bucle de captura** (`local/capture_loop.py`): 1 frame cada 5s, guardado en `frames/`. Duerme el tiempo restante, para que el periodo sea de 5s aunque la captura tarde. Si falla, registra el error, renueva el token y sigue.

- **Problema:** Con `python local/capture_loop.py` no se encuentra `common`. Se lanza como módulo desde la raíz: `python -m local.capture_loop`.

## 28/09/2026 - 1.3 Detección local de personas con YOLOv8 + gráfica básica

- Se implementó la interfaz `count_people(img) -> (n, boxes)` en
  `common/count_people.py`. El detector recibe directamente bytes JPEG,
  iguales a los producidos por `capture_frame()`.

- Se utiliza el modelo `yolov8l.pt` tras probar `yolo8n.pt` y no obtener
  buenos resultados. Sigue siendo necesario realizar pruebas en profundidad
  de día con la cámara definitiva que se desee utilizar.
  El modelo se carga una sola vez al arrancar el proceso y se
  reutiliza para todos los frames, evitando repetir el coste de
  inicialización.

- La inferencia se limita a la clase COCO `person` (`class_id = 0`).
  El umbral de confianza y el tamaño de entrada de YOLO permanecen
  configurables mediante `YOLO_CONF` y `YOLO_IMGSZ`. Del mismo modo que
  con el modelo, también es necesario probar estos valores para hallar
  el más adecuado a la escena escogida.

- Se separaron las operaciones comunes de imagen en `common/image.py`.
  `decode_jpeg()` convierte bytes JPEG en una imagen OpenCV y
  `encode_jpeg()` realiza la operación inversa. Esto evita duplicar la
  lógica de conversión entre el detector y el módulo de anotación.

- El tipo `BoundingBox` se ha creado en `common/types.py`, ya que lo utilizan
  tanto el detector como el módulo encargado de dibujar las detecciones.

- Para esta etapa se utiliza un CSV mínimo con las columnas
  `timestamp,n_people`. Las métricas específicas de comparación entre
  arquitecturas se incorporarán en las partes posteriores cuando sean
  necesarias.

### Separación entre las tareas 1.2 y 1.3

- Se creó `local/detection_loop.py` como proceso independiente para la
  tarea 1.3. Este proceso observa `frames/` y procesa únicamente las
  imágenes generadas desde el comienzo de la ejecución actual.

- Las imágenes originales nunca se mueven ni se modifican. Los resultados
  derivados de YOLO se almacenan por separado.

- `local/run_local.py` funciona únicamente como lanzador de ambos procesos:
  inicia `capture_loop.py` y `detection_loop.py` como procesos separados.
  De este modo, la salida de la tarea 1.2 se convierte en la entrada de la
  tarea 1.3.

### Organización de resultados

- Cada ejecución crea un directorio independiente:

      results/local/<fecha-hora>/
          annotated/
          detections.csv
          people_timeline.png

- `frames/` se mantiene como repositorio común de las capturas originales.

- Las imágenes anotadas conservan exactamente el mismo nombre que el frame
  original. Por ejemplo:

      frames/20260929_003501.jpg

  se corresponde con:

      results/local/<ejecución>/annotated/20260929_003501.jpg

- Esto permite relacionar directamente la imagen original, la imagen
  anotada y el resultado almacenado en el CSV sin duplicar los frames
  originales.

### Sincronización entre procesos

- Como `capture_loop.py` escribe directamente cada JPEG en `frames/`,
  `detection_loop.py` espera un breve intervalo (0.25s) antes de leer un archivo
  recién creado. De esta forma se reduce la posibilidad de procesar una
  imagen que todavía se esté escribiendo.

- Para distinguir los frames de la ejecución actual de capturas antiguas,
  `run_local.py` registra el instante de inicio y `detection_loop.py`
  ignora archivos cuya fecha de modificación sea anterior.

- El timestamp almacenado en el CSV se obtiene del nombre generado por
  `capture_loop.py` (`YYYYMMDD_HHMMSS.jpg`), ya que representa el instante
  de captura.

### Ejecución

Para pruebas límitado a cierto número de frames:
`python -m local.run_local --max-frames 3`

De forma continua:
`python -m local.run_local`

Cálculo de la gráfica de la última ejecución:
`python -m local.plot_results`

Cálculo de la gráfica de una ejecución anterior:
`python -m local.plot_results --run-dir results/local/20260929_014010_654321`

## 29/09/2026 - 1.3 Calibración de YOLO

- **Cámara utilizada:** `https://www.skylinewebcams.com/es/webcam/espana/andalucia/cadiz/grazalema.html`.
- **Experimento** (`local/calibrate_yolo.py`): 30 frames de referencia contados a mano. Se comparan los modelos `n`, `s`, `m` y `l`, con tamaños de entrada 640, 960 y 1280 y umbrales de confianza de 0.15 a 0.5. Se mide el error medio de conteo (MAE) y el tiempo de inferencia por frame. La terraza del bar no se ha tenido en cuenta en la evaluación (hay personas tapadas y difíciles de contar a mano).

- **Resultados** (ver `screenshots/pareto.png`):
  - `yolov8s` a 960 con conf 0.25: MAE 1.7 personas/frame, sin sesgo apreciable, 123 ms/frame.
  - `yolov8n` se queda corto. Su mejor opción es 960 con conf 0.20: MAE 2.5, 54 ms/frame.
  - `m` y `l` solo mejoran a `s` con umbrales bajos y a un coste mucho mayor (`l` a 1280: MAE 1.2, 914 ms/frame).

- **Decisión:** `yolov8s`, 960, conf 0.25. En caso de necesitar usar `yolov8n`, se usará 960 con conf 0.20.

## 09/10/2026 - 2.2 Envío de frames a la VM

- **Contrato cliente <-> VM:** `POST http://<VM>:8000/frame` con el JPEG en el cuerpo y las cabeceras `Content-Type: image/jpeg`, `X-Camera-Id`, `X-Captured-At` y `Authorization: Bearer <token>`. Respuesta JSON `{"n_people", "inference_ms", "boxes"}`, 401 si el token no es válido, y 400 si la imagen no es válida. Hay que configurar en el `.env` `VM_URL`, `VM_TOKEN` y `CAMERA_ID`.

- **Captura reutilizable:** La lógica de resolver la URL, capturar y renovar el token con reintento se extrae de `capture_loop.py` a la clase `FrameSource` (`common/capture.py`), para usarla en todos los clientes.

- **Servidor de prueba** (`vm/fake_server.py`): Cumple el contrato sin YOLO y devuelve un resultado fijo. Sirve para probar el cliente y como base para el servidor real (2.3). El endpoint se declara con `def` y no con `async def` para que una inferencia lenta no bloquee el resto de peticiones.

- **Cliente** (`local/vm_client.py`): Envía un frame cada 5 s y registra cada petición en `results/vm_client/<ejecución>/requests.csv` (`timestamp, camera_id, frame_bytes, capture_ms, latency_ms, inference_ms, n_people, status`). La latencia se mide solo alrededor del envío (ida y vuelta), sin la captura, que depende de SkylineWebcams y no de la VM. Se reutiliza la conexión para que la latencia no incluya abrir una conexión nueva en cada frame. Si hay errores se registran y el bucle continúa.

- **Pruebas en local:** Correctas con el servidor de prueba, con token incorrecto y con el servidor parado. A 1080p y con calidad 90, cada frame ocupa unos 400 KB.

- **Problema encontrado:** La conexión a la VM da timeout en el puerto 8000. Queda pendiente abrir el puerto 8000 (TCP) en Azure para repetir la prueba contra la VM (el repo ya está clonado y con todo configurado, solo falta el puerto).

## 11/10/2026 - 2.2 Conexión con la VM a través de Apache

- **Puertos:** Solo se exponen el 22 (SSH) y el 80 (HTTP). No se abre el 8000.

- **Apache como proxy inverso:** El puerto 80 lo ocupa Apache, que reenvía `/frame` al servidor FastAPI, que escucha solo dentro de la VM (`127.0.0.1:8000`) y no es accesible desde fuera.

- **Contrato actualizado:** La URL pasa a ser `POST http://<IP_VM>/frame`, sin el puerto. El resto se mantiene igual.
