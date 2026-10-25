"""Import isolated upstream functions without importing veRL/Ray/torch."""
import importlib.util
from pathlib import Path


def module(name):
    if name not in {"scoring", "yesno"}:
        raise ValueError(name)
    path = Path(__file__).parents[2] / "verl/third_party/official_vlmeval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"official_{name}", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
