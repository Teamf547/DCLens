"""Small JSON and NPY operations shared by maps and simulation outputs."""
import json
from pathlib import Path


def _read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def _npy_path(path):
    path = Path(path).expanduser().resolve()
    return path if path.suffix == '.npy' else Path(str(path) + '.npy')
