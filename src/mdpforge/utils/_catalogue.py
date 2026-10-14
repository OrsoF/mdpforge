"""Read literal catalogue metadata without importing model or solver modules."""

import ast


def _read_metadata(directory):
    """Yield public module names and independently parsed metadata dictionaries."""
    for source in sorted(directory.iterdir(), key=lambda item: item.name):
        if (
            not source.is_file()
            or not source.name.endswith(".py")
            or source.name.startswith("_")
        ):
            continue
        metadata = {}
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "METADATA"
                for target in node.targets
            ):
                try:
                    metadata = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise TypeError(
                        f"{source.name}: METADATA must be a literal dict"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise TypeError(f"{source.name}: METADATA must be a literal dict")
                break
        yield source.name[:-3], metadata
