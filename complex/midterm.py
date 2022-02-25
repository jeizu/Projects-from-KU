import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange


def generate_network_a(N):
    # Generate left and right neighbours
    network = np.eye(N, k=1, dtype=int) + np.eye(N, k=-1, dtype=int)
    # And wrap around
    network[[0, N-1], [N-1, 0]] = 1
    return network


def generate_network_b(N):
    # Generate upper triangle then lower triangle
    diags = [1, 2, N-2, N-1, -1, -2, -N+2, -N+1]
    network = np.zeros((N, N), dtype=int)
    for n in diags:
        network = network + np.eye(N, k=n, dtype=int)
    return network


def switch_bonds(network, node1, node2, r1=None, r2=None):
    N = network.shape[0]
    diags = [1, -1, N-1, -N+1]
    old = network.copy()

    # Get nearest neighbours of nodes
    node1nearest = [(node1+1) % N, (node1-1) % N]
    node2nearest = [(node2+1) % N, (node2-1) % N]

    # Generate part of adjacency matrix, without nearest neighbours
    non_neighbour = network - sum([np.eye(N, k=n, dtype=int) for n in diags])
    # print(f"Non-nearest neighbours:\n{non_neighbour}")

    i1 = node1
    i2 = node2

    # # choose which bonds to swap
    r1 = np.random.randint(0, 2)
    r2 = np.random.randint(0, 2)

    # Grab row from adjacency and find bond indices
    row1 = non_neighbour[i1]
    bonds1 = np.where(row1)[0]
    row2 = non_neighbour[i2]
    bonds2 = np.where(row2)[0]

    # print(f"Row1 is row {node1}: {row1}. has bonds {bonds1}")
    # print(f"Row2 is row {node2}: {row2}. has bonds {bonds2}")

    # choose random bond
    j1 = bonds1[r1]
    j2 = bonds2[r2]

    if (j1 in node2nearest) or (j2 in node1nearest):
        # j1 is a nearest neighbour of node2, or j2 is a nearest neighbour of
        # j1. ABORT
        return network


    # print(network.sum())
    # delete current bonds
    i_to_delete = [i1, j1, i2, j2]
    j_to_delete = [j1, i1, j2, i2]
    network[i_to_delete, j_to_delete] = 0
    sum_delete = network.sum()

    # Create new bonds
    i_to_create = [i1, j1, i2, j2]
    j_to_create = [j2, i2, j1, i1]
    # print(i_to_create, j_to_create)
    network[i_to_create, j_to_create] = 1
    sum_create = network.sum()
    # print(network.sum())

    if sum_delete + 4 != sum_create:
        # We don goofed!
        # print("Bonds were lost!")
        # print(f"i1={i1}, j1={j1}, i2={i2}, j2={j2}")
        # print(f"Original bonds1: {bonds1} and {node1nearest}")
        # print(f"original bonds2: {bonds2} and {node2nearest}")
        # print(f"Tried to delete: {list(zip(i_to_delete, j_to_delete))}")
        # print(f"Tried to create: {list(zip(i_to_create, j_to_create))}")
        return old

    return network


def find_neighbours(network, i):
    return np.where(network[i])[0].tolist()


def breadth_first_search(occupied, network, start_i):
    """
    Perform Breadth first search on board to find clusters
    """

    # Create list of cells in current cluster
    cluster = [start_i]

    # Running list of neighbours to check (the queue)
    queue = find_neighbours(network, start_i)

    while queue:
        # Take first cell in neighbours
        i = queue.pop(0)

        # Find its neighbours
        neighbours = find_neighbours(network, i)

        # Add cell to cluster, if it is occupied, and not already in it.
        # Also add neighbours to queue
        if occupied[i] and (i not in cluster):
            cluster.append(i)
            queue.extend(neighbours)

    return cluster


def find_largest_cluster(occupied, network):
    occupied_indices = np.where(occupied)[0]
    found = []
    sizes = []
    for i in occupied_indices:
        # If i is already part of a previous cluster, we skip it
        if i in found:
            continue

        # If not, find largest cluster, add to found nodes, and record size
        cluster = breadth_first_search(occupied, network, i)
        found.extend(cluster)
        sizes.append(len(cluster))
    return max(sizes)


def percolation1():
    network_size = [10**i for i in [2, 3, 4, 5]]
    p = 0.9
    number_trials = 1000

    
    # probs = np.linspace(0.1, 0.9, number_probabilities)

    average_largest = []
    average_std = []
    for m in trange(len(network_size)):
        size = network_size[m]
        sizes = []
        network = generate_network_a(size)
        for n in trange(number_trials, leave=False):
            occupied = np.random.uniform(size=size)
            mask = occupied < p
            occupied[mask] = 1
            occupied[~mask] = 0
            largest = find_largest_cluster(occupied, network)
            sizes.append(largest)

        average_largest.append(np.mean(sizes))
        average_std.append(np.std(sizes))

    average_largest = np.array(average_largest)/np.array(network_size)
    average_std = np.array(average_std)/np.array(network_size)
    np.save("size.npy", average_largest)
    np.save("std.npy", average_std)
    fig, ax = plt.subplots()
    ax.errorbar(network_size, average_largest, average_std)
    ax.set_xscale('log')
    plt.show()


if __name__ == "__main__":
    # net = generate_network_b(100)
    # print(net.sum())
    # N = 100
    # for n in range(1000):
    #     node1, node2 = np.random.choice(N, size=2, replace=False)
    #     net = switch_bonds(net, node1, node2)
    #     print(net.sum())

    # rows_that_sum_to_4 = net.sum(axis=0) == 4
    # print(f"we should have {N} rows. We have: {rows_that_sum_to_4.sum()}")

    # diags = [1, -1, N-1, -N+1]
    # non_neighbour = net - sum([np.eye(N, k=n, dtype=int) for n in diags])
    # rows_that_sum_to_2 = non_neighbour.sum(axis=0) == 2
    # print(f"we should have {N} rows. We have: {rows_that_sum_to_2.sum()}")
    # N = 10
    # net = generate_network_a(N)
    # occupied = np.ones(N)
    # clust = breadth_first_search(occupied, net, 1)
    # print(net)
    # print(clust)
    # print(len(clust))
    percolation1()
