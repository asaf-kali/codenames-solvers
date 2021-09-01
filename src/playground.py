import networkx as nx
from gensim import models
from pyvis.network import Network

# %%

w = models.KeyedVectors.load_word2vec_format("DataFiles/GoogleNews-vectors-negative300.bin", binary=True)

# %%
a = w.most_similar(w.get_vector("king") + w.get_vector("woman") - w.get_vector("man"))
print(a)

# %%
board_words = ["king", "queen", "teenage", "tomato", "parrot", "london", "spiderman"]
n = len(board_words)
board_vecs = w[board_words]

# %%
nx_graph = nx.Graph()
nx_graph.add_nodes_from(board_words)
for i in range(n):
    for j in range(i + 1, n):
        nx_graph.add_edge(
            board_words[i],
            board_words[j],
            weight=100 * (1 - w.similarity(board_words[i], board_words[j])),
        )

# %%
# nx_graph = nx.cycle_graph(10)
# nx_graph.nodes[1]['title'] = 'Number 1'
# nx_graph.nodes[1]['group'] = 1
# nx_graph.nodes[3]['title'] = 'I belong to a different group!'
# nx_graph.nodes[3]['group'] = 10
# nx_graph.add_node(20, size=20, title='couple', group=2)
# nx_graph.add_node(21, size=15, title='couple', group=2)
# nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)

nt = Network("500px", "500px")

# populates the nodes and edges data structures
nt.from_nx(nx_graph)
nt.show_buttons(filter_=["physics"])
nt.show("visualizer/nx.html")
