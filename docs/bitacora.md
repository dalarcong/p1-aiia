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
```python -m local.run_local --max-frames 3```

De forma continua:
```python -m local.run_local```

Cálculo de la gráfica de la última ejecución:
```python -m local.plot_results```

Cálculo de la gráfica de una ejecución anterior:
```python -m local.plot_results --run-dir results/local/20260929_014010_654321```