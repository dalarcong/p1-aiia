# Practice 1 - Cloud Computing Services

Architectures and Infrastructures for Artificial Intelligence
Authors: David Alarcón Garrido and Laura Saval Cillero.

## Structure

| Folder         | Contents                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `common/`      | Frame capture, YOLOv8 detection and shared utilities                        |
| `local/`       | Part 1: local capture and counting, YOLO calibration                        |
| `vm/`          | Part 2: client, FastAPI server, web page and VM metrics                     |
| `serverless/`  | Part 3: Blob Storage upload, Azure Function and metrics                     |
| `multicam/`    | Part 4: multi-camera launcher and scalability sweeps                        |
| `advanced/`    | Advanced: compression, and unique people and vehicle counting with tracking |
| `calibration/` | Manual ground truth and calibration results                                 |
| `docs/`        | Work log, test results (`results/`) and screenshots                         |

## Usage

Each component has its own `requirements.txt`. Configuration goes in a `.env` file created from `.env.example`. Scripts run as modules from the repository root, for example:

```bash
python -m local.run_local
python -m vm.vm_client
python -m serverless.upload_loop
python -m multicam.launcher
python -m advanced.track_count --duration 600
```
