from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsam.algorithms.representations import representations

if TYPE_CHECKING:
    from tsam.config import Distribution, MinMaxMean

# Default representation method per clustering method. Mean-based methods are
# represented by the cluster mean, medoid-based methods by the medoid, and
# kmaxoids by the maxoid. Overridable via ``representation_method``.
_DEFAULT_REPRESENTATION = {
    "averaging": "mean",
    "kmeans": "mean",
    "kmedoids": "medoid",
    "kmaxoids": "maxoid",
    "hierarchical": "medoid",
    "contiguous": "medoid",
}


def _ward_labels(candidates: np.ndarray, n_clusters: int) -> np.ndarray:
    """Ward clustering into ``n_clusters`` groups, sklearn-compatible labels.

    Builds the Ward dendrogram with :func:`scipy.cluster.hierarchy.linkage`,
    which shares scikit-learn's node numbering (leaves ``0..n-1``, the ``i``-th
    merge becoming node ``n + i``). Cutting and labelling the tree with the same
    heap traversal scikit-learn's ``AgglomerativeClustering`` uses reproduces its
    exact cluster ids without importing scikit-learn — so the cluster ordering,
    and every result keyed on it, is unchanged from the previous backend.
    """
    from heapq import heappush, heappushpop

    from scipy.cluster.hierarchy import linkage

    children = linkage(candidates, method="ward")[:, :2].astype(np.intp)
    n_leaves = len(candidates)

    # Descend from the single root, splitting the highest node each time, until
    # ``n_clusters`` subtrees remain (scikit-learn's ``_hc_cut``).
    nodes = [-(int(children[-1].max()) + 1)]
    for _ in range(n_clusters - 1):
        merge = children[-nodes[0] - n_leaves]
        heappush(nodes, -int(merge[0]))
        heappushpop(nodes, -int(merge[1]))

    labels = np.zeros(n_leaves, dtype=np.intp)
    for label, node in enumerate(nodes):
        labels[_leaves_of(-node, children, n_leaves)] = label
    return labels


def _leaves_of(node: int, children: np.ndarray, n_leaves: int) -> list[int]:
    """Return the original-period indices under a dendrogram ``node``."""
    if node < n_leaves:
        return [node]
    leaves: list[int] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current < n_leaves:
            leaves.append(current)
        else:
            stack.extend(int(child) for child in children[current - n_leaves])
    return leaves


def assign_clusters(
    candidates: np.ndarray,
    n_clusters: int,
    cluster_method: str,
    n_iter: int = 100,
    solver: str = "highs",
) -> np.ndarray:
    """Assign each period to a cluster.

    Pure clustering step: maps each period (row of ``candidates``) to a cluster
    id, without computing the cluster representatives. See
    :func:`cluster_and_represent` for the combined clustering + representation
    step.

    Valid ``cluster_method`` values: 'averaging', 'kmeans', 'kmedoids',
    'kmaxoids', 'hierarchical', 'contiguous'.
    """
    if cluster_method == "averaging":
        n_sets = len(candidates)
        cluster_size = n_sets // n_clusters
        order_lists = [[c] * cluster_size for c in range(n_clusters)]
        remainder = n_sets - cluster_size * n_clusters
        if remainder > 0:
            order_lists.append([n_clusters - 1] * remainder)
        order = np.hstack(np.array(order_lists, dtype=object))  # type: ignore[call-overload]
        return np.asarray(order)

    if cluster_method == "kmeans":
        from sklearn.cluster import KMeans

        k_means = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=n_iter, tol=1e-4)
        return np.asarray(k_means.fit_predict(candidates))

    if cluster_method == "kmedoids":
        from tsam.algorithms.k_medoids_exact import KMedoids

        kmedoids = KMedoids(n_clusters=n_clusters, solver=solver)
        return np.asarray(kmedoids.fit_predict(candidates))

    if cluster_method == "kmaxoids":
        from tsam.algorithms.k_maxoids import KMaxoids

        return np.asarray(KMaxoids(n_clusters=n_clusters).fit_predict(candidates))

    if cluster_method in ("hierarchical", "contiguous"):
        if n_clusters == 1:
            return np.asarray([0] * len(candidates))

        if cluster_method == "hierarchical":
            # Unconstrained Ward via scipy keeps the default path off sklearn
            # (~600 ms import) while reproducing its exact cluster ids.
            return _ward_labels(candidates, n_clusters)

        # contiguous: only adjacent periods may be merged. scipy's hierarchy has
        # no connectivity constraint, so this keeps sklearn's constrained Ward.
        from sklearn.cluster import AgglomerativeClustering

        adjacency_matrix = np.eye(len(candidates), k=1) + np.eye(len(candidates), k=-1)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", connectivity=adjacency_matrix
        )
        return np.asarray(clustering.fit_predict(candidates))

    raise ValueError(
        f"Unknown cluster_method '{cluster_method}'. "
        f"Valid options: 'averaging', 'kmeans', 'kmedoids', 'kmaxoids', "
        f"'hierarchical', 'contiguous'."
    )


def cluster_and_represent(
    candidates: np.ndarray,
    n_clusters: int = 8,
    n_iter: int = 100,
    cluster_method: str = "kmeans",
    solver: str = "highs",
    representation_method: str | Distribution | MinMaxMean | None = None,
    representation_dict: dict[str, str] | None = None,
    distribution_period_wise: bool = True,
    n_timesteps_per_period: int | None = None,
    representation_candidates: np.ndarray | None = None,
    reference_attribute_idx: int | None = None,
) -> tuple[list[np.ndarray], list[int] | None, np.ndarray]:
    """Cluster ``candidates`` and compute the representative profile per cluster.

    Two steps: :func:`assign_clusters` assigns each period to a cluster, then
    :func:`~tsam.algorithms.representations.representations` derives the cluster
    representatives (using the per-method default unless ``representation_method``
    overrides it).
    """
    cluster_order = assign_clusters(
        candidates, n_clusters, cluster_method, n_iter=n_iter, solver=solver
    )

    # Representatives may be drawn from a separate candidate set (e.g. unweighted).
    rep_candidates = (
        representation_candidates
        if representation_candidates is not None
        else candidates
    )
    cluster_centers, cluster_center_indices = representations(
        rep_candidates,
        cluster_order,
        default=_DEFAULT_REPRESENTATION[cluster_method],
        representation_method=representation_method,
        representation_dict=representation_dict,
        distribution_period_wise=distribution_period_wise,
        n_timesteps_per_period=n_timesteps_per_period,
        reference_attribute_idx=reference_attribute_idx,
    )
    return cluster_centers, cluster_center_indices, cluster_order
