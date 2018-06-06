import networkx as nx

def make_ancestor_closure(G, name):
    G_BFS = nx.bfs_tree(G, 0)
    G_A   = nx.Graph()
    f = open(name + ".edges", 'w')

    for node in G_BFS.nodes():
        curr = node
        while len(list(G_BFS.predecessors(curr))):
            curr = list(G_BFS.predecessors(curr))[0]
            G_A.add_edge(node, curr)
            f.write(str(node) + "\t" + str(curr) + "\n")
    f.close()
    return G_A

def save_edges(G, name, data=False):
    f = open(name + ".edges", 'w')

    for edge in G.edges(data=data):
        if data:
            f.write(str(edge[0]) + "\t" + str(edge[1]) + "\t" + str(edge[2]['weight']) + "\n")
        else:
            f.write(str(edge[0]) + "\t" + str(edge[1]) + "\n")

    f.close()

def make_tree_weights(G):
    G_BFS = nx.bfs_tree(G, 0)
    G_W   = nx.Graph()

    curr_nodes = [0]
    next_nodes = []
    depth = 0

    while 1:
        if len(curr_nodes) == 0:
            if len(next_nodes) == 0:
                break
            depth += 1
            curr_nodes = next_nodes.copy()
            next_nodes.clear()

        node = curr_nodes[0]
        parent = list(G_BFS.predecessors(node))
        if len(parent) > 0:
            G_W.add_edge(node, parent[0], weight=3**(depth-1))

        curr_nodes.remove(node)
        next_nodes += list(G_BFS.successors(node))

    save_edges(G_W, "weighted_testanc", data=True)

if __name__ == '__main__':
    G = nx.balanced_tree(2,3)
    make_ancestor_closure(G, 'testclosure')
    make_tree_weights(G)
    nx.write_edgelist(G, 'test.edges', data=False)
