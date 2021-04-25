import pdb
import os
import pickle
import networkx as nx


def createFolder(path):
    filename = os.path.join(os.getcwd(), f"{path}.graph")
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def saveResultToFile(
    folderName, circuitDescription, testability, hardest10PercentS0, hardest10PercentS1
):
    fileNames = lambda name: f"{folderName}.{name}"
    graph_file, x, y = map(fileNames, ["graph", "x", "y"])
    createFolder(graph_file)
    graph = {}
    circuit_hash = circuitDescription[2]
    for k, v in circuit_hash.items():
        graph[k] = graph.get(k, set())
        graph[k].update(v["outputs"])
        for input_node in v["inputs"]:
            graph[input_node] = graph.get(input_node, set())
            graph[input_node].add(k)
    pickle.dump(graph, open(graph_file, "wb"))

    graph_nodes = nx.from_dict_of_lists(graph).nodes
    X = []
    Y = []
    hard_to_observe = set(hardest10PercentS0)
    hard_to_observe.update(set(hardest10PercentS1))
    for k in graph_nodes:
        node_elements = []
        if testability.get(k):
            for attr in ["level", "control0", "control1", "obs"]:
                node_elements.append(testability[k][attr])
        if node_elements:
            X.append(node_elements)

        if k in hard_to_observe:
            Y.append("ho")
        else:
            Y.append("eo")
    pickle.dump(X, open(x, "wb"))
    pickle.dump(Y, open(y, "wb"))
