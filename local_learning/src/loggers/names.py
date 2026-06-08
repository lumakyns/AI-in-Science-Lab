import re


_LAYER_INDEX_PATTERN = re.compile(r"(?<![A-Za-z0-9_])layer(\d+)(?!\d)")


def wandb_safe_layer_name(layer_name: str, *, layer_index_width: int = 2) -> str:
    """Convert a layer path to a sortable WandB metric component."""

    def pad_layer_index(match: re.Match[str]) -> str:
        return f"layer{match.group(1).zfill(layer_index_width)}"

    padded_name = _LAYER_INDEX_PATTERN.sub(pad_layer_index, str(layer_name))
    return padded_name.replace(".", "__").replace("/", "__")
