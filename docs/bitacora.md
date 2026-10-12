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

## 11/10/2026 - 2.3 Procesamiento YOLOv8 en la VM

- Se sustituyó el servidor de prueba de la tarea 2.2 por un servidor real
  (`vm/server.py`) basado en FastAPI manteniendo el contrato definido.

- Se utiliza YOLOv8 sobre CPU. El modelo se carga una única vez al arrancar
  el servidor y se reutiliza para todas las peticiones, evitando cargar los
  pesos en cada frame.

- FastAPI se ejecuta con un único worker. Esta decisión evita tener varias
  copias de PyTorch/YOLO simultáneamente en memoria, algo especialmente
  importante en la VM Standard B1s debido a su memoria limitada. Para ello en la VM ejecutar:

  ```
  python -m uvicorn vm.server:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
  ```

  Para comprobar que está en marcha, en otra terminal dentro de la VM ejecutar:

  ```
  curl http://127.0.0.1:8000/health
  ```

- Las inferencias se serializan mediante un Lock. De este modo varias
  peticiones pueden llegar al servidor, pero solo una utiliza el modelo
  YOLO cada vez. Esta decisión también permitirá observar posteriormente
  la formación de colas al realizar las pruebas multicámara.

- El servidor devuelve un JSON con:

      {
          "n_people": ...,
          "inference_ms": ...,
          "boxes": [...]
      }

- `inference_ms` mide únicamente el procesamiento de YOLO dentro de la VM.
  La latencia total (`latency_ms`) se sigue midiendo en el cliente, alrededor
  de la petición HTTP completa, por lo que incluye transferencia, Apache,
  FastAPI, posibles esperas y procesamiento.

- Los resultados procesados por la VM se almacenan en:

      results/vm/<fecha-hora>/detections.csv

  con los campos:

      captured_at,received_at,camera_id,n_people,inference_ms,frame_bytes

- Se utiliza un entorno virtual de Python en la VM para instalar PyTorch,
  Ultralytics y FastAPI sin modificar el Python administrado por el sistema.
  PyTorch se instaló en su versión CPU-only. Para acceder:

  ```
  source .venv/bin/activate
  ```

### Validación

Se realizó una prueba extremo a extremo:

    webcam -> cliente local -> HTTP -> Apache -> FastAPI -> YOLOv8 -> respuesta

El cliente obtuvo, entre otros, los siguientes resultados:

    people=25, latency=5618 ms, inference=4836 ms
    people=28, latency=3779 ms, inference=3375 ms
    people=26, latency=9065 ms, inference=8771 ms
    people=32, latency=4651 ms, inference=4428 ms
    people=30, latency=4360 ms, inference=4006 ms
    people=31, latency=4031 ms, inference=3726 ms
    people=24, latency=3966 ms, inference=3643 ms

La prueba confirma que la mayor parte de la latencia se debe al tiempo de
inferencia en la CPU de la VM, mientras que la diferencia entre
`latency_ms` e `inference_ms` corresponde al resto de costes del sistema
(red, Apache/FastAPI, serialización, etc.).

### Problemas observados

- Durante algunas capturas OpenCV/FFmpeg mostró avisos H.264 del tipo:

      non-existing PPS 0 referenced
      decode_slice_header error
      no frame!

  Estos errores proceden de la decodificación del stream de la webcam en el
  cliente y no del procesamiento en Azure. El cliente consiguió recuperarse
  y seguir capturando frames posteriores.

- La VM Standard B1s presenta tiempos de inferencia elevados y variables.
  En esta primera prueba se observaron aproximadamente entre 3,4 s y 8,8 s
  por frame.

## 11/10/2026 - 2.3 Tiempos de inferencia en la VM y créditos de CPU

- **Problema:** En la primera prueba la inferencia tardaba entre 3,4 y 8,8 s por frame. La VM es "burstable": rinde al 100% mientras tiene créditos de CPU y, si se agotan, queda limitada a su rendimiento base (20%). La instalación de Pytorch y Ultralytics, justo antes de la prueba, agotó los créditos.

- **Repetición con créditos:** Unos 460 ms por frame estables. Los primeros frames tardan 1-1.4 s por el calentamiento del modelo.

- **Comparación:** La VM es unas 3,7 veces más lenta que el portátil con la misma configuración (460 ms frente a 123 ms).

- **Conclusión:** Con una cámara (un frame cada 5 s), la inferencia ocupa alrededor del 9% de la CPU, por debajo del 20% base, así que la VM puede mantener ese ritmo indefinidamente.

## 11/10/2026 - 2.4 Publicación de resultados para la web

- **Datos para la web:** Si está definida `VM_WEB_DATA_DIR` (en la VM, `/var/www/html/data`, servida por Apache), el servidor publica tras cada frame la última imagen con las cajas y una copia del CS de la ejecución. El CSV de la web se vacía al arrancar el servidor: la web muestra la ejecución actual.

- **Errores de la web:** Si falla la publicación, se registra pero la petición responde igualmente 200. La publicación se hace fuera del lock de inferencia, así que suma algo a la latencia del cliente pero no a `inference_ms`.

- **Calentamiento de YOLO:** Al arrancar se hace una inferencia sobre una imagen en blanco, para que el primer frame real no pague el coste inicial.

- **FastAPI:** El arranque pasa de `@app.on_event("startup")` (obsoleto) a `lifespan`.

## 12/10/2026 - 2.4 Web de resultados

- **Servida por Apache:** Desde `/var/www/html` (`vm/web/index.html` y `vm/web/script.js` del repo), accesible en `http://<IP_VM>/`. La página lee los datos que publica el servidor (`data/detections.csv` y `data/latest.jpg`) y se actualiza cada 5 s.

- **Contenido:** Personas por frame a lo largo del tiempo, histograma de personas por frame, tiempo de inferencia por frame y el último frame con las detecciones. Indicadores de personas actuales, media, máximo, frames procesados, inferencia media, tamaño medio del frame y hora de la última actualización. Un indicador de estado avisa si no llegan frames nuevos en 30 s.

- **Despliegue:** Tras cada cambio, `git pull` y `cp vm/web/index.html vm/web/script.js /var/www/html`.



## 11/10/2026 - 2.4 Falta de memoria en la VM

- **Síntoma:** Con la web activa, la inferencia pasó a más de 10 s por frame.

- **Diagnóstico:** RAM llena (15 MB disponibles de 892) y la VM sin swamp. `vmstat` mostraba lecturas continuas del disco y la CPU esperando al disco. Sin swap, el sistema expulsaba de la RAM el código de las librerías de Pytorch y lo volvía a leer en cada inferencia.

- **Causa:** La publicación para la web añadió memoria al servidor, que ya estaba casi al límite.

- **Solución:** swap de 2 GB. La inferencia vuelve a los valores normales sin cambiar de modelo.

- **Después del swap:** 160 MB de RAM disponibles (antes 15), 236 MB de caché (antes 53), 184 MB en el swap y sin lecturas continuas de disco.

- **Conclusión:** En esta VM el recurso limitante es la RAM, no la CPU.



## 12/10/2026 - 2.5 Métricas de la VM (en curso)

### Eficacia

- **Método** (`vm/eval_effectiveness.py`): Los 30 frames de referencia se envían a la VM con el mismo contrato del cliente. Las cajas devueltas se filtran con la misma zona excluida usada en el conteo manual y se comparan con `calibration/ground_truth.csv`.

- **Resultado:** MAE 1.70 personas/frame, sesgo +0.30, 5/30 conteos exactos. Inferencia media en la VM: 497 ms (frente a 123 ms en el portátil); los dos primeros frames, más lentos por estar el servidor recién arrancado (1048 y 1423 ms), suben la media. En la prueba de 10 minutos la mediana es de 445 ms. El conteo es idéntico al del modelo en local: la eficacia no depende de dónde se ejecute la IA.

### Prueba de rendimiento

- **Condiciones:** 12/10/2026, de 11:51:02 a 12:01:02 (CEST). Cámara de Grazalema, de día. Una cámara, un frame cada 5 s. `yolov8s`, imgsz 960, conf 0.25. VM B2ats_v2 con swap de 2 GB y la web activa. Cliente en el portátil.

- **Datos:** `docs/results/2_5_vm`. El monitor de la VM se lanzó un minuto antes y se paró un minuto después. Sus columnas de red (`net_recv_kbps`, `net_sent_kbps`) no son válidas en esta prueba: contaban dos veces cada frame al sumar la interfaz interna (`lo`), por la que Apache reenvía los frames a FastAPI. Corregido después en el monitor (`_external_traffic`).

- **Frames:** 117 de 120, todos correctos. Los 3 que faltan se deben a una captura de SkylineWebcams que tardó 21,7 s.

- **Tiempo de respuesta** (mediana / p95): latencia 920 / 1178 ms; inferencia 445 / 449 ms; resto (red, Apache y publicación web) 474 / 728 ms.

- **CPU:** Media 4%, máximo 32%. Steal 0% durante toda la prueba: sin limitación por créditos. Con una cámara la carga está muy por debajo del 20% base.

- **Memoria:** Unos 190 MB disponibles y 265 MB en el swap, estables. Iowait ~0 %. El servidor ocupa 407-443 MB.

- **Ancho de banda:** 456 KiB por frame (1080p, calidad 90), unos 89 KiB/s de subida durante la prueba.

- **Análisis** (`vm/analyze_run.py`): Tabla resumen y gráficas en `docs/results/2_5_vm/analysis` (`summary.md`, `latency_timeline.png`, `latency_distribution.png`, `vm_resources.png`). La inferencia es casi constante. La variación de latencia viene de la red y del resto de la petición.

### Coste

- **Tarifas** (pago por uso, Linux, Poland Central; API pública de precios de Azure, consultada el 12/10/2026): VM B2ats v2 0,0095 €/h; disco Standard SSD E4 2,11 €/mes (~0,0029 €/h); IP pública estática 0,0044 €/h. **Total ~0,0168 €/h (~12,3 €/mes).**

- **Coste por frame:** 0,0000239 € con los 704 frames/hora procesados en la prueba (0,0000233 € con los 720 teóricos de un frame cada 5 s). Solo la VM: ~0,0000132 €/frame.

- El coste de la VM es fijo por hora, se procesen frames o no: con más cámaras el coste por frame baja, hasta el límite que aguante la VM. El tráfico de entrada (los frames) es gratuito.

### Pendiente (Laura, en el portal de Azure)

- **Azure Monitor**, ventana de **11:46 a 12:06 (CEST)**, granularidad de 1 minuto:
  - _Percentage CPU_ (Avg y Max).
  - _CPU Credits Remaining_ y _CPU Credits Consumed_.
  - _Available Memory Bytes_ (Min).
  - _Network In Total_ y _Network Out Total_ (Sum): ancho de banda externo real.
  - _Disk Read Bytes_ (Sum).
- **Cost Management** (_Cost analysis_), a partir del día siguiente, filtrado por el grupo de recursos.
- Capturas en `docs/screenshots/` con el formato `2_5_azure_<métrica>.png`, y, si el portal lo permite, exportación a CSV.



## 12/10/2026 - 3.2 Subida de imágenes al contenedor propio
Subida de imágenes al contenedor compleatada. Ejecutar con:
```
python -m serverless.upload_loop
```
Se añadieron 4 nuevos valores en el ``.env``

```
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
AZURE_RAW_CONTAINER=
AZURE_RAW_PREFIX=

CAPTURE_INTERVAL_SERVERLESS=
```

Se configuró un periodo objetivo de subida de 0,5 s, siguiendo el ejemplo del enunciado. Sin embargo, la frecuencia efectiva resultó inferior debido a que el proceso de captura abre y decodifica el stream H.264 en cada iteración. En las pruebas realizadas, la captura y subida requirieron aproximadamente entre 1,5 y 4 s por frame. Al superar este tiempo el intervalo configurado, el cliente inicia inmediatamente el siguiente ciclo sin espera adicional. Finalmente se ha dejado fijado el intervalo de subida en 2.5s.

En Windows durante la captura se observaron avisos de FFmpeg relativos a SPS/PPS del flujo H.264. Al tratarse de avisos asociados al stream de la webcam y continuar obteniéndose imágenes válidas, no se introdujo tratamiento adicional.

Asimismo, se ha detectado una posible optimización a evaluar más adelante de ``source.capture()`` manteniendo el stream abierto.