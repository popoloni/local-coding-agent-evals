import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_old_config_import_path_still_works():
    from project.config import DEFAULT_TIMEOUT

    assert DEFAULT_TIMEOUT == 30
