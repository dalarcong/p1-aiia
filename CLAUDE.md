# Práctica 1 – Cloud Computing Services (Smart Lighting System)

Asignatura: Arquitecturas e Infraestructuras para Inteligencia Artificial.
Práctica en pareja (Persona A y Persona B). Idioma de trabajo: español.

## Objetivo

Construir un **Smart Lighting System (SLS)**: webcams públicas actúan como sensores de presencia,
se cuenta el número de **personas** por frame y se compara **dónde se ejecuta la IA**:
local vs. VM en Azure (IaaS) vs. Azure Functions (serverless). El entregable final es una
memoria en PDF con pasos, problemas, decisiones de diseño, capturas y conclusiones comparativas.

El enunciado completo está en `docs/Practice1.pdf`. Los ejemplos del profesorado están en
`docs/ejemplo/`: `grabber.py` (captura MJPEG + YOLOv8 + CSV) y la web `index.html` / `script.js`
(gráfica Chart.js a partir de `database.csv` y `processed_frame.jpg`).
La documentación de Azure del profesorado está en `docs/azure/`:

- `Azure documentation 1.pdf`: Function App (plan de consumo, límite 1,5 GB RAM) y web estática
  en Storage Account (`$web`) → tareas 3.1 y 3.3.
- `Azure documentation 2.pdf`: recurso Azure AI services (S0) y snippet con
  `azure-ai-vision-imageanalysis` (`VisualFeatures.PEOPLE`) → tareas 3.4 y 3.5. Su último snippet
  (`VisionSource`) es del SDK antiguo: con el nuevo, `client.analyze(image_data=bytes)` no necesita `/tmp`.

**Partir siempre de ellos** cuando el enunciado lo indique.

## Partes y tareas

- **Parte 1 – Local**
  - 1.1 Elegir webcam pública fija con personas.
  - 1.2 Capturar 1 frame cada 5 s.
  - 1.3 Contar personas con YOLOv8 (usar `yolov8n`), guardar CSV (fecha, nº personas) y gráfica temporal.
- **Parte 2 – VM en Azure**
  - 2.1 VM Standard B1s (1 GB RAM) con python3-opencv y python3-pip (`--no-cache-dir` si sale "killed").
  - 2.2 El cliente local envía frames a la VM.
  - 2.3 La VM ejecuta YOLOv8 y guarda CSV.
  - 2.4 Web con histograma accesible por IP pública.
  - 2.5 Métricas: eficacia, CPU, RAM, ancho de banda, coste, tiempo de respuesta.
- **Parte 3 – Serverless**
  - 3.1 Web estática en Storage Account.
  - 3.2 El cliente sube frames a un contenedor "raw" (distinto del de la web) cada 0,5 s.
  - 3.3 Azure Function con blob trigger.
  - 3.4 Recurso Azure AI Vision (Image Analysis 4.0).
  - 3.5 La Function cuenta personas y dibuja cajas. Ojo: las cajas vienen de _objects/people_, no de _tags_.
  - 3.6 Métricas: eficacia, coste, tiempo de respuesta.
- **Parte 4 – Multicámara**
  - 4.1 Lanzador de N cámaras a 1 frame / 5 s por cámara.
  - 4.2 Límite de cámaras de la VM.
  - 4.3 Límite de cámaras en serverless.
- **Avanzado (+2)**
  - Contar personas únicas mediante tracking (`model.track()`).
  - Contar también vehículos.
  - Optimizaciones: resolución mínima, compresión, detección de movimiento en local, etc.

## Interfaces comunes (no cambiar sin acordarlo)

- `capture_frame(url) -> bytes (JPEG)` en `common/capture.py`.
- `count_people(img) -> (n, boxes)` en `common/count_people.py`.
- CSV de resultados (`common/results.py`): de momento `timestamp, n_people` (Parte 1). Las
  columnas para comparar arquitecturas (latencia, tamaño del frame, etc.) se acordarán en la Parte 2.

## Estructura del repo

```
common/  local/  vm/  serverless/  multicam/  results/  docs/ (bitacora.md, screenshots/, Practice1.pdf, ejemplo/, azure/)
.env.example  .gitignore  README.md
```

## Reglas y convenciones

- **Nunca** subir secretos: `.env`, `local.settings.json`, connection strings y claves.
  Tampoco pesos del modelo (`*.pt`), frames en bruto ni `venv/`.
- Toda la configuración (URLs de cámaras, intervalos, IP de la VM, nombres de contenedores) va en `.env` o en config.
- Una rama por parte (`feat/local`, `feat/vm`, `feat/serverless`, `feat/multicam`, `feat/avanzado`)
  y commits con número de tarea (p. ej. `2.3: YOLO en la VM guarda CSV`).
- Un `requirements.txt` por componente.
- `vm/setup_vm.sh` reproduce la VM completa: swap de 2 GB, torch **CPU-only**
  (`--index-url https://download.pytorch.org/whl/cpu`) y dependencias.
- Desplegar en la VM con `git pull`, no copiando archivos a mano.
- Anotar en `docs/bitacora.md` los problemas y decisiones, con fecha, porque la memoria los pide.

## Avisos críticos

- **Azure Function:** el trigger debe dispararse SOLO en el contenedor `raw/{name}`, y la salida
  debe escribirse en otro contenedor. Si no, entra en ejecución recursiva y consume los créditos.
- Parar la VM cuando no se use. Configurar alertas de presupuesto.
- Coste: Cost Management tarda 8–24 h en reflejar el gasto; calcular también el **coste por frame**
  a partir de las tarifas.

## Metodología de medición

- Latencia medida en el cliente (ida y vuelta), para evitar desincronización de relojes.
- Pruebas de duración fija (p. ej. 10 min por configuración), siempre con las mismas cámaras.
- Eficacia: verdad manual en 30–50 frames, comparando YOLO con Azure AI Vision.
- CPU/RAM de la VM con `psutil`, o con las métricas de Azure Monitor.

## Cámaras

- Preferencia por escenas tranquilas: 2–10 personas por frame, bien separadas, con las personas
  ocupando al menos un 5–10 % del alto de la imagen.
- Fuentes sugeridas por el enunciado: EarthCam, SkylineWebcams y WebcamTaxi.
