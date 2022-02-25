#include <iostream>
#include <armadillo>
#include <list>

using namespace std;
using namespace arma;

imat create_network_a(int N);
imat create_network_b(int N);
ivec create_occupancy(int N, float p);
int find_largest_cluster(imat network, ivec occupancy);
int bfs(imat network, ivec occupancy, int start_i);
list<int> get_neighbours(imat network, int i);
void print_list(list<int> l);
bool in_list(list<int> l, int i);


void print_list(list<int> l)
{
    for (list<int>::iterator it = l.begin(); it != l.end(); it++)
        cout << *it << ' ';
    cout << '\n';
}

imat create_network_a(int N)
{
    // Create integer matrix
    imat A(N, N, fill::zeros);

    int n_diags = 4;
    int diags[4] = {1, -1, N-1, -N+1};
    for (int i=0; i<n_diags; i++)
    {
        A.diag(diags[i]).ones();
    }
    return A;
};

imat create_network_b(int N)
{
    // Creates the adjacency matrix for nearest and next-nearest neighbour
    // coupling, with size N.

    // Create integer matrix
    imat A(N, N, fill::zeros);

    // Fill 8 diff diagonals with ones
    int n_diags = 8;
    int diags[8] = {1, -1, 2, -2, N-1, -N+1, N-2, -N+2};
    for (int i = 0; i < n_diags; i++)
    {
        A.diag(diags[i]).ones();
    }
    return A;
};

ivec create_occupancy(int N, float p)
{
    vec b(N, fill::randu);
    uvec mask1 = find(b < 0.2);
    uvec mask2 = find(b >= 0.2);
    ivec c(N, fill::zeros);
    c.elem(mask1).ones();
    return c;
}

list<int> get_neighbours(imat network, int i)
{
    // Finds neighbours of node i
    int N = network.n_cols;
    ivec row = network.col(i);
    list<int> neighbours;
    for (int j=0; j<N; j++)
    {
        if (row[j] == 1)
        {
            neighbours.push_back(j);
        }
    }

    return neighbours;
};

bool in_list(list<int> l, int i)
{
    bool in = false;
    for (list<int>::iterator it = l.begin(); it != l.end(); it++)
    {
        if (*it == i)
        {
            in = true;
            break;
        }
    }
    return in;
}

void extend_list(list<int> l1, list<int> l2)
{
    for (list<int>::iterator it = l2.begin(); it != l2.end(); it++)
    {
        l1.push_back(*it);
    }
}

int bfs(imat network, ivec occupancy, int start_i)
{
    list<int> cluster = {start_i};
    list<int> queue = get_neighbours(network, start_i);

    while (queue.size() > 0)
    {
        // cout << "Queue has size " << queue.size() << endl;
        // Pop front of queue
        int i = queue.front();
        queue.pop_front();

        // Grab the neighbours
        list<int> neigh = get_neighbours(network, i);
        
        // cout << "Current element is " << i << " and has neighbours:" << endl;
        // print_list(neigh);

        // If the node is occupied, and not already in the cluster

        // cout << "occupancy[i]: " << occupancy[i] << " and in_list: "
        //      << in_list(cluster, i) << endl;
        // cout << "Together: " << occupancy[i] << " and " 
        //      << not in_list(cluster, i) << " gives " 
        //      << (occupancy[i] and (not in_list(cluster, i))) << endl;

        if (occupancy[i] and (not in_list(cluster, i)))
        {
            // add to cluster
            cluster.push_back(i);

            // update queue
            // extend_list(queue, neigh);
            for (list<int>::iterator it = neigh.begin(); it != neigh.end(); it++)
            {
                queue.push_back(*it);
            }
        }
    }
    return cluster.size();
};

int main()
{
    // Network size
    int N=100;

    // Create network
    
    imat A = create_network_a(N);
    // cout << A << endl;

    list<int> neigh = get_neighbours(A, 1);
    // print_list(neigh);

    list<int> a = {1, 2, 0, 4, 2};
    // cout << in_list(a, 2) << " " << in_list(a, 3) << endl;

    ivec occu(N, fill::ones);
    int size = bfs(A, occu, 1);
    cout << size << endl;
    // N = 5;
    // for (int i=0; i<N; i++)
    // {
    //     int c = a.front();
    //     a.pop_front();
    //     cout << c << " " << a.size() << endl;
    // }

    ivec c = create_occupancy(N, 0.2);
    cout << c << endl << sum(c);

    return 0;
}
