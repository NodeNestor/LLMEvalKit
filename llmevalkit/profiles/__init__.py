from pathlib import Path
import yaml

PROFILE_DIR = Path(__file__).parent


def load_profile(name: str) -> dict:
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        # Allow absolute or relative path to arbitrary YAML
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"No profile '{name}'. Built-ins: {[p.stem for p in PROFILE_DIR.glob('*.yaml')]}"
            )
    data = yaml.safe_load(path.read_text())
    # Expand `extends`
    if "extends" in data:
        merged: list = []
        for parent_name in data["extends"]:
            parent = load_profile(parent_name)
            merged.extend(parent.get("runners", []))
        merged.extend(data.get("runners", []))
        data["runners"] = merged
    return data


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
