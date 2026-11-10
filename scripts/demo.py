"""Print W, D and L for the four-node example supplied by member B."""

from spectral_clustering.graphs import graph_matrices_from_edges


def main() -> None:
    edges = [(0, 1, 5), (1, 3, 2), (2, 3, 9), (0, 3, 3)]
    adjacency, degree, laplacian = graph_matrices_from_edges(4, edges)

    print("W =\n", adjacency)
    print("D =\n", degree)
    print("L =\n", laplacian)


if __name__ == "__main__":
    main()
