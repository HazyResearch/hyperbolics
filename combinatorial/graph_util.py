import networkx as nx

# wrapper for nx.bfs_tree that keeps weights
def get_BFS_tree(G, src):
    G_BFS = nx.bfs_tree(G, src)

    for edge in G_BFS.edges():
        if G[edge[0]][edge[1]]:
            G_BFS.add_edge(edge[0], edge[1], weight=G[edge[0]][edge[1]]['weight'])
    
    return G_BFS

def max_degree(G):
    max_d = 0;
    max_node = -1;
    
    for deg in G.degree(G.nodes(), 'weight'):
        if deg[1] > max_d:
            max_d = deg[1]
            max_node = deg[0]
            
    return [max_node, max_d]

    
    
    