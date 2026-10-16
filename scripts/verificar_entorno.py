"""Verifica el entorno "inventariovision".

Uso (Anaconda Prompt):
    conda activate inventariovision
    python scripts/verificar_entorno.py

Imprime el intérprete, las versiones, si hay GPU, prueba que TensorFlow y
PyTorch convivan en el mismo proceso y corre yolo26n.pt sobre una imagen de prueba.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

LIBRERIAS = ["ultralytics", "torch", "tensorflow", "keras", "tensorboard", "cv2", "numpy",
             "pandas", "matplotlib", "PIL", "rembg", "imagehash", "roboflow", "fastapi",
             "uvicorn", "sqlalchemy", "pydantic", "dotenv", "anthropic", "pytest"]


def main() -> int:
    print("Python:", sys.executable)
    if "inventariovision" not in sys.executable:
        print("  ATENCIÓN: no es el entorno 'inventariovision'.")
    print("Versión:", sys.version.split()[0])

    errores = []
    for nombre in LIBRERIAS:
        try:
            modulo = importlib.import_module(nombre)
            print(f"  {nombre:12s} {getattr(modulo, '__version__', '(sin versión)')}")
        except Exception as e:  # noqa: BLE001 - queremos ver cualquier fallo de import
            errores.append(nombre)
            print(f"  {nombre:12s} ERROR: {e}")

    import tensorflow as tf
    import torch
    print("\nGPU PyTorch (CUDA):", torch.cuda.is_available())
    print("GPU TensorFlow:", tf.config.list_physical_devices("GPU") or "ninguna")
    print("Hilos CPU de PyTorch:", torch.get_num_threads())

    # TF y torch en el mismo proceso: una operación de cada uno.
    print("Prueba TF + torch juntos:", float(tf.reduce_sum(tf.ones((2, 2)))),
          float(torch.ones(2, 2).sum()))

    from contratos.vision_types import ResultadoConteo  # noqa: F401
    from contratos import api_schemas, modelos_bd  # noqa: F401
    print("Contratos: importan OK")

    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    pesos = RAIZ / "modelos" / "yolo26n.pt"
    pesos.parent.mkdir(exist_ok=True)
    modelo = YOLO(str(pesos) if pesos.exists() else "yolo26n.pt")
    if not pesos.exists() and Path("yolo26n.pt").exists():
        Path("yolo26n.pt").replace(pesos)   # la descarga cae en el cwd; la movemos a modelos/
    imagen = ASSETS / "bus.jpg"
    modelo.predict(imagen, verbose=False)   # calentamiento
    inicio = time.perf_counter()
    resultado = modelo.predict(imagen, verbose=False)[0]
    ms = (time.perf_counter() - inicio) * 1000
    nombres = [resultado.names[int(c)] for c in resultado.boxes.cls]
    print(f"\nYOLO26n sobre {imagen.name}: {len(nombres)} objetos en {ms:.0f} ms -> {sorted(set(nombres))}")

    if errores:
        print("\nFALLARON:", ", ".join(errores))
        return 1
    print("\nEntorno OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
