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
