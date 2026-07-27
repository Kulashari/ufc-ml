"""Feature discovery, ordering, and semantic grouping."""

from ufc_ml_core.features.groups import (
    REQUIRED_ABLATION_GROUPS,
    FeatureGroup,
    ablation_feature_groups,
    classify_feature,
    group_features,
)
from ufc_ml_core.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    discover_feature_columns,
)

__all__ = [
    "REQUIRED_ABLATION_GROUPS",
    "FeatureGroup",
    "FeatureRegistry",
    "FeatureSpec",
    "ablation_feature_groups",
    "classify_feature",
    "discover_feature_columns",
    "group_features",
]
