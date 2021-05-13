from spektral.data import Dataset, DisjointLoader, Graph
from data_pre_processing import load_data


class CircuitDataset(Dataset):
    def read(self, path="../data/output", circs=[]):
        circuits = []
        # circs = [
        #     "c6288",
        #     "c5315",
        #     "c432",
        #     "c499",
        #     "c880",
        #     "c1355",
        #     "c1908",
        #     "c3540",
        #     "adder.bench",
        #     "arbiter.bench",
        #     "cavlc.bench",
        #     "dec.bench",
        #     "voter.bench",
        #     "sin.bench",
        #     "priority.bench",
        # ]
        for circ in circs:
            A, X, labels = load_data(circ, path, normalize="")
            circuits.append(Graph(x=X.toarray(), a=A, y=labels))
            print(f"{circ}: {sum(labels)}, {len(labels)}")
        return circuits