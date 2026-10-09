"""Model recipes and their catalogue metadata."""

import ast
from importlib.resources import files


def list_models(category: str | None = None) -> list[dict]:
    """Read recipe metadata without importing models or constructing matrices.

    Names come from filenames. Models may define a literal ``METADATA`` dict
    containing category, description and reference. Missing fields are None;
    references are the source attributions declared by the model authors.
    Actual dimensions belong to built instances, not catalogue entries.
    """
    entries = []
    for source in sorted(files(__package__).iterdir(), key=lambda item: item.name):
        if not source.is_file() or not source.name.endswith(".py"):
            continue
        if source.name.startswith("_"):
            continue
        metadata = {}
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "METADATA"
                for target in node.targets
            ):
                metadata = ast.literal_eval(node.value)
                if not isinstance(metadata, dict):
                    raise TypeError(f"{source.name}: METADATA must be a literal dict")
                break
        entry = {
            "name": source.name[:-3],
            "category": metadata.get("category"),
            "description": metadata.get("description"),
            "reference": metadata.get("reference"),
        }
        if category is None or entry["category"] == category:
            entries.append(entry)
    return entries
