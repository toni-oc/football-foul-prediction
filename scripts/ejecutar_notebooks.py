"""Ejecuta los cuatro notebooks en kernels nuevos, usando el Python que invoca el script.

No necesita nbconvert ni nbclient; usa el protocolo de Jupyter instalado con ipykernel.
Las salidas se guardan únicamente tras completar cada notebook sin errores.
"""
import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from jupyter_client import KernelManager


RAIZ = Path(__file__).resolve().parents[1]


def ejecutar(ruta, timeout=1800):
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    for celda in nb["cells"]:
        if celda["cell_type"] == "code":
            celda["execution_count"], celda["outputs"] = None, []
    runtime = RAIZ / "artifacts/runtime"
    runtime.mkdir(parents=True,exist_ok=True)
    if not runtime.resolve().is_relative_to(RAIZ.resolve()):
        raise ValueError("La carpeta temporal debe permanecer dentro del proyecto.")
    # OneDrive/Windows puede retener brevemente history.sqlite tras cerrar el kernel.
    # Ese bloqueo de una caché no convierte en fallida una ejecución correcta.
    with tempfile.TemporaryDirectory(dir=runtime,ignore_cleanup_errors=True) as temporal:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE":"1", "LOKY_MAX_CPU_COUNT":"2", "OMP_NUM_THREADS":"1",
            "OPENBLAS_NUM_THREADS":"1", "IPYTHONDIR":str(Path(temporal)/"ipython"),
            "MPLCONFIGDIR":str(Path(temporal)/"matplotlib"), "JUPYTER_PLATFORM_DIRS":"1"}
        km = KernelManager(connection_file=str(Path(temporal)/"kernel.json"),kernel_name="python3")
        km.kernel_spec.argv = [sys.executable,"-B","-m","ipykernel_launcher","-f","{connection_file}"]
        try:
            km.start_kernel(cwd=str(ruta.parent),env=env)
            cliente = km.client()
            cliente.start_channels()
            cliente.wait_for_ready(timeout=90)
            for celda in nb["cells"]:
                if celda["cell_type"] != "code":
                    continue
                codigo = "".join(celda["source"])
                mid = cliente.execute(codigo,stop_on_error=True)
                limite = time.monotonic()+timeout
                while True:
                    restante = limite-time.monotonic()
                    if restante <= 0:
                        raise TimeoutError(f"Timeout de una celda de {ruta.name}")
                    m = cliente.get_iopub_msg(timeout=restante)
                    if m.get("parent_header",{}).get("msg_id") != mid:
                        continue
                    tipo,c = m["msg_type"],m["content"]
                    if tipo == "status" and c["execution_state"] == "idle":
                        break
                    if tipo == "execute_input":
                        celda["execution_count"] = c["execution_count"]
                    elif tipo == "stream":
                        celda["outputs"].append({"output_type":"stream","name":c["name"],"text":c["text"]})
                    elif tipo in ("display_data","execute_result"):
                        salida = {"output_type":tipo,"data":c["data"],"metadata":c.get("metadata",{})}
                        if tipo == "execute_result":
                            salida["execution_count"] = c["execution_count"]
                        celda["outputs"].append(salida)
                    elif tipo == "clear_output":
                        celda["outputs"] = []
                    elif tipo == "error":
                        raise RuntimeError(f"{ruta.name}: {c['ename']}: {c['evalue']}\n"+"\n".join(c["traceback"]))
            ruta.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
        finally:
            with suppress(Exception):
                cliente.stop_channels()
            with suppress(Exception):
                km.shutdown_kernel(now=True)
            with suppress(Exception):
                km.cleanup_resources()
    print(f"Ejecutado sin errores: {ruta.name}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks",nargs="*")
    args = parser.parse_args()
    rutas = [RAIZ / "notebooks" / n for n in args.notebooks] if args.notebooks else sorted((RAIZ / "notebooks").glob("*.ipynb"))
    for ruta in rutas:
        ejecutar(ruta)
