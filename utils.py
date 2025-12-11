# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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

    return np.concatenate([T, H, F], axis=1)

def logreg_f(n_features, n_classes=2):
    model = LogisticRegression(penalty=None, fit_intercept=True)

    # Manually set the fitted attributes
    model.classes_ = np.arange(n_classes)
    model.coef_ = np.random.randn(1, n_features)
    model.intercept_ = np.random.randn(1)
    model.n_features_in_ = n_features

    return model

def adjust_arrays(X):
    if isinstance(X, list):
        list_of_column_vectors = [np.expand_dims(x, axis=1) if x.ndim==1 else x for x in X]
        Xp = np.concatenate(list_of_column_vectors, axis=1)
    else:
        if X.ndim == 1:
            Xp = np.expand_dims(X, axis=1)
        else:
            Xp = X
    return Xp

def logreg_addestra(X, Y, model:LogisticRegression):
    Xp = adjust_arrays(X)
    model.fit(Xp, Y)

def logreg_predici(X, model:LogisticRegression):
    Xp = adjust_arrays(X)
    Yp = model.predict(Xp)
    return Yp

def calcola_accuratezza(Y: np.ndarray, Yp: np.ndarray) -> float:
    # Ensure both arrays are numpy arrays
    Y = np.asarray(Y)
    Yp = np.asarray(Yp)
    
    # Convert both to boolean if they are floats
    if np.issubdtype(Y.dtype, np.floating):
        Y = Y >= 0.5
    if np.issubdtype(Yp.dtype, np.floating):
        Yp = Yp >= 0.5
    
    # Now both are boolean arrays, safe to compare
    return np.mean(Y == Yp)

def disegna_decision_boundary(X, model:LogisticRegression, fun=lambda x1, x2: [x1, x2]):
    Xp = adjust_arrays(X)
    X1 = np.arange(np.min(Xp[:, 0]), np.max(Xp[:, 0]), 0.001)
    X2 = np.arange(np.min(Xp[:, 1]), np.max(Xp[:, 1]), 0.001)

    xx1, xx2 = np.meshgrid(X1, X2)

    x1_x2_features = adjust_arrays([xx1.flatten(order='F'), xx2.flatten(order='F')])
    features = adjust_arrays(fun(x1_x2_features[:, 0], x1_x2_features[:, 1]))
    y = model.predict(features)

    YY = np.reshape(y, shape=xx1.shape, order='F')

    plt.contour(X1, X2, YY, [0.0], origin='lower')

def disegna_torta(lista, etichette):
    _, ax = plt.subplots()
    ax.pie(lista, labels=etichette, autopct='%1.1f%%')

def calcola_tabella_infarto_sesso(D, S):
    cm = np.zeros((2, 2))
    cm[0, 0] = np.sum((D==1) & (S==0))
    cm[1, 0] = np.sum((D==1) & (S==1))
    cm[0, 1] = np.sum((D==0) & (S==0))
    cm[1, 1] = np.sum((D==0) & (S==1))
    return cm

def stampa_tabella_infarto_sesso(D, S):
    cm = calcola_tabella_infarto_sesso(D, S)

    data = [
        ["F", cm[0, 0], cm[0, 1]],
        ["M", cm[1, 0], cm[1, 1]]
    ]

    print(tabulate(data, headers=["M/F", "Infarto", "Sano"], tablefmt="grid"))

def disegna_ecg(E, D, index, fs=1000.0):
    t = np.arange(0, E.shape[1], 1)/fs
    plt.plot(t, E[index, :])
    plt.title(f'Diagnosi: {'Infarto' if D[index] == 1 else 'Sano'}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ampiezza (mV)')

def disegna_due_ecg(E, D, index1, index2, fs=1000.0):
    t = np.arange(0, E.shape[1], 1)/fs
    
    _, ax = plt.subplots(1, 2)

    m = 1.1*np.min(np.array([np.min(E[index1, :]), np.min(E[index2, :])]))
    M = 1.1*np.max(np.array([np.max(E[index1, :]), np.max(E[index2, :])]))

    ax[0].plot(t, E[index1, :])
    ax[0].set_title(f'Diagnosi: {'Infarto' if D[index1] == 1 else 'Sano'}')
    ax[0].set_xlabel('Tempo (s)')
    ax[0].set_ylabel('Ampiezza (mV)')
    ax[0].set_ylim([m, M])

    ax[1].plot(t, E[index2, :])
    ax[1].set_title(f'Diagnosi: {'Infarto' if D[index2] == 1 else 'Sano'}')
    ax[1].set_xlabel('Tempo (s)')
    ax[1].set_ylabel('Ampiezza (mV)')
    ax[1].set_ylim([m, M])

def ann_f(n_hidden=5, n_classes=2):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(n_hidden, n_classes), random_state=1)
    
    return model

def ann_addestra(X, Y, model:MLPClassifier):
    Xp = adjust_arrays(X)
    model.fit(Xp, Y)

def ann_predici(X, model:MLPClassifier):
    Xp = adjust_arrays(X)
    Yp = model.predict(Xp)
    return Yp

def dividi_dataset(X, Y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def stampa_matrice_confusione_infarto(D, Yp):
    cm = np.zeros((2, 2))
    cm[0, 0] = np.sum((D==1) & (Yp==1))
    cm[1, 0] = np.sum((D==1) & (Yp==0))
    cm[0, 1] = np.sum((D==0) & (Yp==1))
    cm[1, 1] = np.sum((D==0) & (Yp==0))

    data = [
        ["ANN+", cm[0, 0], cm[0, 1]],
        ["ANN-", cm[1, 0], cm[1, 1]]
    ]

    print(tabulate(data, headers=["", "Infarto", "Sano"], tablefmt="grid"))