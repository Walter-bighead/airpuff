import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

try:
    import requests
except Exception:
    requests = None


MODULES = [
    ("flask", "flask"),
    ("requests", "requests"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python-headless"),
    ("faster_whisper", "faster-whisper"),
]


def module_status(module_name, package_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"installed": False, "package": package_name}
    try:
        module = importlib.import_module(module_name)
        try:
            version = importlib.metadata.version(package_name)
        except Exception:
            version = getattr(module, "__version__", None)
    except Exception as exc:
        return {"installed": True, "package": package_name, "import_ok": False, "error": str(exc)}
    return {
        "installed": True,
        "package": package_name,
        "import_ok": True,
        "version": version,
    }


def run_command(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return {"ok": True, "output": out.strip()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_ollama():
    result = {"cli": shutil.which("ollama") is not None, "api_ok": False, "models": []}
    if requests is None:
        result["error"] = "requests_missing"
        return result
    try:
        res = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        res.raise_for_status()
        payload = res.json()
        result["api_ok"] = True
        result["models"] = [item.get("name") for item in payload.get("models", [])]
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main():
    modules = {name: module_status(name, pkg) for name, pkg in MODULES}
    have_base = all(modules[name].get("installed") and modules[name].get("import_ok", True) for name in ("flask", "requests"))
    have_vision = have_base and all(
        modules[name].get("installed") and modules[name].get("import_ok", True) for name in ("numpy", "cv2")
    )
    have_full_python = have_vision and modules["faster_whisper"].get("installed") and modules["faster_whisper"].get("import_ok", True)

    report = {
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "modules": modules,
        "commands": {
            "ffmpeg": run_command(["ffmpeg", "-version"]),
            "ollama_version": run_command(["ollama", "--version"]) if shutil.which("ollama") else {"ok": False, "error": "not_found"},
        },
        "ollama": check_ollama(),
    }

    report["profiles"] = {
        "minimal_ready": have_base,
        "vision_lite_ready": have_vision,
        "vision_flow_ready": have_vision,
        "full_ready": have_full_python and report["ollama"].get("api_ok", False),
    }

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
