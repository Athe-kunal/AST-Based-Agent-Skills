import networkx as nx
from ast_skills.common.api import FunctionCallInfo, ClassInfo, ModuleIndex


def build_call_only_graph(infos: list[FunctionCallInfo]) -> nx.DiGraph:
    """
    Graph with only call relationships:
      caller -> callee
    """
    G = nx.DiGraph()

    for info in infos:
        G.add_node(
            info.full_name,
            node_type=info.kind,
            module=info.module_name,
            qualname=info.qualname,
            class_name="" if info.class_name is None else info.class_name,
            lineno=-1 if info.lineno is None else info.lineno,
            end_lineno=-1 if info.end_lineno is None else info.end_lineno,
        )

    for info in infos:
        for callee in info.calls:
            if not G.has_node(callee):
                G.add_node(callee, node_type="external")
            G.add_edge(info.full_name, callee)

    return G


def compute_pagerank(
    G: nx.DiGraph,
    alpha: float = 0.85,
) -> dict[str, float]:
    return nx.pagerank(G, alpha=alpha)


def top_ranked_methods(
    G: nx.DiGraph,
    pagerank_scores: dict[str, float],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    method_nodes = [
        node
        for node, attrs in G.nodes(data=True)
        if attrs.get("node_type") in {"method", "async_method"}
    ]

    ranked = sorted(
        ((node, pagerank_scores[node]) for node in method_nodes),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


def top_ranked_classes(
    G: nx.DiGraph,
    pagerank_scores: dict[str, float],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    class_nodes = [
        node for node, attrs in G.nodes(data=True) if attrs.get("node_type") == "class"
    ]

    ranked = sorted(
        ((node, pagerank_scores[node]) for node in class_nodes),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


def aggregate_method_pagerank_by_class(
    infos: list[FunctionCallInfo],
    pagerank_scores: dict[str, float],
) -> dict[str, float]:
    class_scores: dict[str, float] = {}

    for info in infos:
        if info.class_name is None:
            continue

        class_full = f"{info.module_name}.{info.class_name}"
        class_scores.setdefault(class_full, 0.0)
        class_scores[class_full] += pagerank_scores.get(info.full_name, 0.0)

    return class_scores


def top_classes_by_method_pagerank(
    infos: list[FunctionCallInfo],
    pagerank_scores: dict[str, float],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    class_scores = aggregate_method_pagerank_by_class(infos, pagerank_scores)
    return sorted(class_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
