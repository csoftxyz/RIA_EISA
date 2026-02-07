import numpy as np
import networkx as nx
import pymatching
import multiprocessing
import time
from scipy.sparse import csc_matrix

print("=== Z3 Toric Code: Low-P Threshold Scan ===\n")

# Use 'fork' to prevent Windows-style process spawning recursion loop printing
# Note: On Windows, this print repetition is normal but annoying.
# The logic below is wrapped in __main__ to be safe.

def build_toric_graph(L):
    G = nx.Graph()
    nodes = [(x, y) for x in range(L) for y in range(L)]
    node_map = {n: i for i, n in enumerate(nodes)}
    G.add_nodes_from(range(len(nodes)))
    
    for x in range(L):
        for y in range(L):
            u = node_map[(x, y)]
            neighbors = [((x+1)%L, y), ((x, (y+1)%L), ((x+1)%L, (y+1)%L))] # Triangular
            # Fixed syntax error in neighbor list construction above
            neighbors = [
                ((x+1)%L, y), 
                (x, (y+1)%L), 
                ((x+1)%L, (y+1)%L) # Diagonal for Triangular
            ]
            for nx_node in neighbors:
                v = node_map[nx_node]
                if not G.has_edge(u, v): G.add_edge(u, v)
    return G

def simulate_shot(args):
    L, p, seed = args
    G = build_toric_graph(L) # Cached in worker
    num_edges = G.number_of_edges()
    
    rng = np.random.default_rng(seed)
    noise = (rng.random(num_edges) < p).astype(int)
    
    # H matrix construction (sparse)
    row_ind, col_ind = [], []
    for idx, (u, v) in enumerate(G.edges()):
        row_ind.extend([u, v])
        col_ind.extend([idx, idx])
    H = csc_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(L*L, num_edges))
    
    syndrome = H @ noise % 2
    matching = pymatching.Matching(H)
    prediction = matching.decode(syndrome)
    
    residual = (noise + prediction) % 2
    if np.sum(residual) > 0:
        # Check homology: if path spans > L/2
        # Simple heuristic: weight of residual > L/2
        if np.sum(residual) >= L/2: 
            return 1
    return 0

def run_scan():
    L_LIST = [8, 12, 16]
    # Scan low p region to find crossing
    P_LIST = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] 
    TRIALS = 2000
    
    print(f"{'p':<8} | {'L=8':<10} | {'L=12':<10} | {'L=16':<10}")
    print("-" * 50)
    
    with multiprocessing.Pool() as pool:
        for p in P_LIST:
            row_str = f"{p:.4f}   "
            vals = []
            for L in L_LIST:
                seeds = [int(time.time()*1000)+i for i in range(TRIALS)]
                args = [(L, p, s) for s in seeds]
                fails = sum(pool.map(simulate_shot, args))
                rate = fails / TRIALS
                vals.append(rate)
                row_str += f"| {rate:.4f}     "
            
            # Check for crossing: if Rate(L=16) < Rate(L=8), we are below threshold
            if vals[2] < vals[0]:
                row_str += " [BELOW THRESHOLD]"
            print(row_str)

if __name__ == "__main__":
    run_scan()