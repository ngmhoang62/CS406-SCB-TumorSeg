"""Run TV4 week 1 notebook from a fresh kernel and save outputs in place."""

import os
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks/TV4_Week1_Training_SmokeTest.ipynb"


def main():
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    # Private kernel spec: no global/user Jupyter configuration is changed.
    kernel_root = PROJECT_ROOT / "outputs/tv4_week1/kernels"
    kernel_dir = kernel_root / "tv4-local"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "TV4 local", "language": "python",
    }), encoding="utf-8")
    os.environ.setdefault("MPLBACKEND", "Agg")
    runtime_dir = PROJECT_ROOT / "outputs/tv4_week1/runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    manager = KernelSpecManager(kernel_dirs=[str(kernel_root)])
    kernel_manager = AsyncKernelManager(kernel_name="tv4-local", kernel_spec_manager=manager)
    client = NotebookClient(nb, timeout=900, startup_timeout=60, kernel_name="tv4-local",
                            resources={"metadata": {"path": str(PROJECT_ROOT)}},
                            km=kernel_manager)
    client.on_cell_start = lambda cell, cell_index: print(f"Cell {cell_index + 1}/{len(nb.cells)}", flush=True)
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(nb, NOTEBOOK)
    print(f"PASS: executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells: {NOTEBOOK}")
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text.rstrip())


if __name__ == "__main__":
    main()
