__all__ = ["BinnedCosmicIntegrator", "CosmicIntegration"]


def __getattr__(name: str):
    if name == "CosmicIntegration":
        from .ratesSampler import CosmicIntegration

        return CosmicIntegration
    if name == "BinnedCosmicIntegrator":
        from .binned_cosmic_integrator import BinnedCosmicIntegrator

        return BinnedCosmicIntegrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
