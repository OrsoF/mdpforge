"""Database repairman model: continuous-time source, awaiting an explicit finite MDP adaptation."""

METADATA = {
    "category": "service_operations",
    "description": "Database repairman model",
    "reference": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/Database",
    "model_type": "sta",
    "source_collection": "jani-models",
    "source_url": "https://github.com/ahartmanns/jani-models/tree/5e2fd8fa31778ad7e06b298b2d3e333ba450b042/Database",
    "variants": 4,
    "available": False,
    "reason": "Source type sta requires a finite discrete-time MDP adaptation before benchmarking.",
    "sizes": {
        "small": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "Source type sta requires a finite discrete-time MDP adaptation before benchmarking.",
        },
        "medium": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "Source type sta requires a finite discrete-time MDP adaptation before benchmarking.",
        },
        "large": {
            "requested_state_dim": None,
            "parameters": None,
            "reason": "Source type sta requires a finite discrete-time MDP adaptation before benchmarking.",
        },
    },
}
