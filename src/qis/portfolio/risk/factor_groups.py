"""Provider-neutral factor-family membership and bump directions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorGroupSpec:
    """Named factor family with an explicit total-bump allocation.

    Weights allocate one unit of a family bump across members. They do not
    collapse the fitted covariance or replace historical factor returns.

    Attributes:
        group_id: Distinct scenario key; must not collide with a fitted factor.
        members: Ordered fitted factor identifiers.
        weights: Nonnegative weights summing to one; None means an equal split.
        label: Optional economic display label supplied by the model definition.
    """

    group_id: str
    members: tuple[str, ...]
    weights: tuple[float, ...] | None = None
    label: str | None = None

    def __post_init__(self):
        """Freeze and validate membership without silently normalizing weights."""
        members = tuple(self.members)
        if (
            not self.group_id
            or not members
            or len(set(members)) != len(members)
            or any(not isinstance(member, str) or not member for member in members)
        ):
            raise ValueError("factor group needs a name and distinct nonempty members")
        weights = (
            (1.0 / len(members),) * len(members) if self.weights is None else tuple(self.weights)
        )
        if (
            len(weights) != len(members)
            or not np.isfinite(weights).all()
            or min(weights) < 0
            or not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("factor-group weights must be nonnegative and sum to one")
        if self.label is not None and not self.label:
            raise ValueError("group label must be nonempty when supplied")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "weights", weights)
