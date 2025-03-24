import math
import networkx as nx

from . import basename, console, get_transformations

SUPPORTED_EXTENSIONS = (".mid", ".midi")


def find_path(
    G: nx.Graph,
    source: str,
    destination: str,
    played_files: list[str],
    max_nodes: int = 5,
    max_updates: int = 20,
    max_visits: int = 1,
    allow_transpose: bool = True,
    allow_shift: bool = True,
    verbose: bool = False,
) -> tuple[list[str], float] | None:
    """
    Find the path from source to destination with the smallest average edge cost.

    This function uses a recursive backtracking approach to explore paths from source
    to destination with at most `max_nodes` nodes, where each node can be visited
    up to `max_visits` times. It selects the path with the lowest average edge cost.

    Parameters
    ----------
    G : networkx.Graph
        A weighted graph with a 'weight' attribute on each edge.
    source : str
        The starting node.
    destination : str
        The target node.
    played_files : list[str]
        List of files/nodes that should be avoided in the path.
    max_nodes : int
        The maximum number of nodes allowed in the path. Defaults to 5.
    max_updates : int
        The maximum number of times to update the best path. Defaults to 20.
    max_visits : int
        Maximum number of times a node can be visited. Defaults to 1 for simple paths.
        Set to 0 to run simple djikstra, 2 or higher to allow revisiting nodes.

    Returns
    -------
    tuple or None
        A tuple (path, total_cost) where path is a list of nodes representing the
        path with the smallest average edge cost, and total_cost is the sum of
        weights along that path; or None if no such path exists.
    """
    # handle case where max_visits is 0
    if max_visits < 1:
        path = nx.shortest_path(G, source=source, target=destination, weight="weight")
        path = [str(f) for f in path]  # make the linter shut up
        return path, 0

    # Convert played_files to set for faster lookups and strip file extensions
    played_nodes = {basename(f) for f in played_files if f != source}

    # Check if destination is in played_files
    if destination in played_nodes:
        if verbose:
            console.log(f"destination is in played files, no valid path exists")
        return None

    best_path = {"path": None, "total_cost": math.inf, "avg_cost": math.inf}
    # counter for number of best path updates
    update_count = {"value": 0}

    def backtrack(current_path, total_weight, visit_counts):
        """
        Recursive function to explore paths from source to destination.

        Parameters
        ----------
        current_path : list
            The current path being explored.
        total_weight : float
            The total weight of edges in the current path.
        visit_counts : dict
            Dictionary tracking number of visits to each node.

        Returns
        -------
        bool
            True if we should continue searching, False if we've reached max updates.
        """
        # if we've reached the update limit, stop searching
        if update_count["value"] >= max_updates:
            return False

        current_node = current_path[-1]

        # if we've reached the destination, check if this path has a better average cost
        if current_node == destination:
            # calculate average cost (total weight / number of edges)
            num_edges = len(current_path) - 1
            if num_edges > 0:
                avg_cost = total_weight / num_edges
                if avg_cost < best_path["avg_cost"]:
                    if verbose:
                        console.log(
                            f"\t\t[grey70]found new best path with avg cost {avg_cost:.4f} (update {update_count['value'] + 1}/{max_updates})[/grey70]"
                        )
                    best_path["path"] = current_path.copy()
                    best_path["total_cost"] = total_weight
                    best_path["avg_cost"] = avg_cost
                    update_count["value"] += 1
                    if update_count["value"] >= max_updates:
                        if verbose:
                            console.log(
                                f"\t\t[grey70]reached maximum of {max_updates} path updates, returning best path found[/grey70]"
                            )
                        return False
            return True

        # if we've exceeded max_nodes, stop exploring this path
        if len(current_path) >= max_nodes:
            return True

        # explore neighbors
        for neighbor, data in G[current_node].items():
            if (
                not allow_transpose
                and get_transformations(neighbor)[1]["transpose"] != 0
            ):
                continue
            if not allow_shift and get_transformations(neighbor)[1]["shift"] != 0:
                continue
            # skip if in played_nodes or if we've visited this node max times
            if (
                neighbor not in played_nodes
                and visit_counts.get(neighbor, 0) < max_visits
            ):
                weight = data.get("weight", 1)

                # pruning: if adding this edge would already make the average cost worse than the best found,
                # don't explore this path further (only if we have found at least one path to destination)
                if best_path["path"] is not None:
                    potential_edges = (
                        len(current_path) - 1 + 1
                    )  # Current edges + 1 for this new edge
                    potential_avg = (total_weight + weight) / potential_edges
                    if potential_avg >= best_path["avg_cost"]:
                        continue

                # increment visit count for this neighbor
                visit_counts[neighbor] = visit_counts.get(neighbor, 0) + 1

                # add neighbor to path and continue exploration
                current_path.append(neighbor)
                should_continue = backtrack(
                    current_path, total_weight + weight, visit_counts
                )
                current_path.pop()  # backtrack

                # decrement visit count when backtracking
                visit_counts[neighbor] -= 1
                if visit_counts[neighbor] == 0:
                    del visit_counts[neighbor]

                # stop exploring if we've reached max updates
                if not should_continue:
                    return False

        return True

    # start backtracking from source with initial visit count for source node
    initial_visits = {source: 1}
    backtrack([source], 0, initial_visits)

    if best_path["path"] is None:
        console.log(
            f"no path found from '{source}' to '{destination}' with at most {max_nodes} nodes "
            f"and max {max_visits} visits per node"
        )
        return None

    if verbose:
        console.log(
            f"found path with {len(best_path['path'])} nodes, total cost {best_path['total_cost']:.4f}, "
            f"and average edge cost {best_path['avg_cost']:.4f} after {update_count['value']} updates"
        )

    return best_path["path"], best_path["total_cost"]
