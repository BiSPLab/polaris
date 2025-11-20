# %%
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression

# %%

def plot_causal_graph_fever2():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    G.add_edges_from([
        ('F', 'T'), ('F', 'H'),
    ])

    # Define layout for better visualization
    # pos = nx.spring_layout(G)
    pos = {'T': np.array([-1, 0]), 'H': np.array([1, 0]), 'F': np.array([0, 1])}

    # Draw nodes, edges, and labels
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='darkblue')


def create_fever2_dataset(N=100):
    F = (np.random.rand(N, 1) > 0.5) + 0.0
    T = 36.5 + F + 0.5*np.random.randn(N, 1)
    H = 7 + F + 0.5*np.random.randn(N, 1)

    #X = create_fever2_dataset(100)
    #np.savetxt('datasets/fever2.csv', X, delimiter=',')

    return np.concatenate([T, H, F], axis=1)

def logistic_regression_f():
    model = LogisticRegression(penalty=None, fit_intercept=True)

    return model

def calcola_accuratezza(Y: np.ndarray, Yp: np.ndarray) -> float:
    # Ensure both arrays are numpy arrays.
    Y = np.asarray(Y)
    Yp = np.asarray(Yp)
    
    # Convert both to boolean if they are floats.
    if np.issubdtype(Y.dtype, np.floating):
        Y = Y >= 0.5
    if np.issubdtype(Yp.dtype, np.floating):
        Yp = Yp >= 0.5
    
    # Now both are boolean arrays, safe to compare.
    return np.mean(Y == Yp)
