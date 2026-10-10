"""Model recipes and their catalogue metadata."""

import ast
from collections.abc import Iterator
from importlib import import_module
from importlib.resources import files

from mdpforge.core.model import MDPProtocol


def list_models(
    category: str | None = None, *, size: str | None = None, type: str | None = None
) -> list[dict]:
    """Read recipe metadata without importing models or constructing matrices.

    Names come from filenames. Models may define a literal ``METADATA`` dict
    containing category, description, reference, tags and size presets.
    Missing tags/sizes are empty; other missing fields are None;
    references are the source attributions declared by the model authors.
    Preset state_dim is the expected actual state count, while parameters are
    constructor inputs. Inferred sizes are runtime estimates, not guarantees.
    A type matches either a category or a tag; size selects available presets.
    """
    if size is not None and size not in ("small", "medium", "large"):
        raise ValueError("size must be 'small', 'medium', 'large' or None")
    if type is not None and (not isinstance(type, str) or not type):
        raise ValueError("type must be a nonempty category/tag string or None")
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
            "tags": metadata.get("tags", []),
            "sizes": metadata.get("sizes", {}),
        }
        entries.append(entry)
    if type is not None and not any(
        entry["category"] == type or type in entry["tags"] for entry in entries
    ):
        raise ValueError(f"Unknown model type {type!r}; use list_models() to browse tags")
    return [
        entry
        for entry in entries
        if (category is None or entry["category"] == category)
        and (type is None or entry["category"] == type or type in entry["tags"])
        and (
            size is None
            or entry["sizes"].get(size, {}).get("parameters") is not None
        )
    ]


def load_model(
    *, size: str | None = None, type: str | None = None, save: bool = True
) -> Iterator[MDPProtocol]:
    """Yield built catalogue models selected by size and category/tag.

    Selection is validated immediately. Models are imported and constructed one
    at a time, in filename order, using create_model() and its existing cache.
    With size=None, constructor defaults are used. Unavailable size presets are
    excluded; import/construction failures propagate when advancing the iterator.
    save=False prevents writing newly built matrices, but still reads caches.
    This catalogue API is separate from utils.persistence.load_model(model),
    which restores the cached matrices of a single instance.
    """
    entries = list_models(size=size, type=type)

    def generate():
        for entry in entries:
            preset = entry["sizes"][size] if size is not None else None
            options = dict(preset["parameters"]) if preset is not None else {}
            if "custom_track" in options:
                import numpy as np

                options["custom_track"] = np.asarray(options["custom_track"])
            model = import_module(f"{__package__}.{entry['name']}").Model(**options)
            model.create_model(save=save)
            if preset is not None and model.state_dim != preset["state_dim"]:
                raise ValueError(
                    f"{entry['name']}/{size}: expected {preset['state_dim']} states, "
                    f"got {model.state_dim}; check the preset and cached matrices"
                )
            yield model

    return generate()
