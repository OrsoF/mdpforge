"""Build reproducible size sweeps using the existing benchmark engine."""

from mdpforge.core.benchmark import Benchmark


class ScalingBenchmark(Benchmark):
    """Benchmark independent instances of one parameterized MDP family."""

    def __init__(self, *, default_solvers=True):
        super().__init__(default_solvers=default_solvers)
        self._scaling_study = None

    def add_family(self, factory, sizes, *, instances=3, seed=0, family=None):
        """Register one size sweep constructed by ``factory(size, model_seed)``.

        factory: callable returning a built MDP or an MDP with create_model().
        sizes: distinct, positive integer targets passed to the factory.
        instances: positive number of independent models per size.
        seed: first model-generation seed, independent of Benchmark.run(seed=...).
        family: optional human-readable label for the manifest.
        """
        if self._scaling_study is not None or self._mdps:
            raise ValueError("Register one family per ScalingBenchmark study")
        if not callable(factory):
            raise TypeError("factory must be callable as factory(size, seed)")
        sizes = list(sizes)
        if not sizes or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("sizes must contain positive integers")
        if len(set(sizes)) != len(sizes):
            raise ValueError("sizes must be distinct")
        if type(instances) is not int or instances < 1:
            raise ValueError("instances must be a positive integer")
        if type(seed) is not int or seed < 0 or seed + len(sizes) * instances > 2**32:
            raise ValueError("seed must allow a distinct 32-bit seed for each model")
        if family is not None and (not isinstance(family, str) or not family.strip()):
            raise ValueError("family must be a nonempty string or None")

        built = []
        metadata = {}
        action_dim = None
        for index, size in enumerate(sizes):
            for instance in range(instances):
                model_seed = seed + index * instances + instance
                model = factory(size, model_seed)
                if not hasattr(model, "create_model"):
                    raise TypeError("factory must return a model with create_model()")
                if type(model.state_dim) is not int or model.state_dim < 1:
                    raise ValueError(
                        "Factory must provide a positive integer state_dim"
                    )
                if action_dim is None:
                    action_dim = model.action_dim
                elif model.action_dim != action_dim:
                    raise ValueError("All instances must have the same action_dim")
                name = getattr(model, "name", None)
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Factory must provide a nonempty model name")
                if name in metadata:
                    raise ValueError(
                        f"Duplicate model name {name!r}; include the generation "
                        "seed in the model name to avoid cache collisions"
                    )
                # Avoid saving an entire sweep to the shared model cache. The
                # factory must use seed in its generation and name for provenance.
                expected_states, expected_actions = model.state_dim, model.action_dim
                model.create_model(save=False)
                if (model.state_dim, model.action_dim) != (
                    expected_states,
                    expected_actions,
                ):
                    raise ValueError(
                        "Model dimensions changed during matrix construction"
                    )
                model.requested_state_dim = size
                metadata[name] = {
                    "state_dim": model.state_dim,
                    "requested_state_dim": size,
                    "instance": instance,
                    "model_seed": model_seed,
                }
                built.append(model)

        for model in built:
            self.add_mdp(model)
        self._scaling_study = {
            "type": "scaling",
            "variable": "state_dim",
            "family": family,
            "sizes": sorted({entry["state_dim"] for entry in metadata.values()}),
            "requested_state_dims": sizes,
            "instances": instances,
            "seed": seed,
            "models": metadata,
        }
        return self

    def reset(self, *, models=True, solvers=True):
        """Reset registrations and clear scaling metadata when models are removed."""
        super().reset(models=models, solvers=solvers)
        if models:
            self._scaling_study = None
        return self

    def run(self, *args, **kwargs):
        """Execute the inherited benchmark and attach the scaling study metadata."""
        if self._scaling_study is None:
            raise ValueError("Call add_family() before running a scaling benchmark")
        if {model.name for model in self._mdps} != set(self._scaling_study["models"]):
            raise ValueError("A scaling benchmark can contain only its family models")
        result = super().run(*args, **kwargs)
        result.experiment["study"] = self._scaling_study.copy()
        return result
