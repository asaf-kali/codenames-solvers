
from gensim import models
from pyvis.network import Network
import networkx as nx
import community


# %%

w = models.KeyedVectors.load_word2vec_format(
    'DataFiles/GoogleNews-vectors-negative300.bin', binary=True)

# %%
a = w.most_similar(w.get_vector('king') + w.get_vector('woman') -w.get_vector('man'))
print(a)

# %%
board_words = ['cloak', 'kiss', 'flood', 'mail', 'skates', 'paper', 'frog', 'skyscraper', 'moon', 'egypt', 'teacher',
                 'avalanche', 'newton', 'violet', 'drill', 'fever', 'ninja', 'jupiter', 'ski', 'attic', 'beach', 'lock',
                 'earth', 'park', 'gymnast']#['king', 'queen', 'teenage', 'tomato', 'parrot', 'london', 'spiderman']
n = len(board_words)
board_vecs = w[board_words]


# %%
nx_graph = nx.Graph()
nx_graph.add_nodes_from(board_words)
for i in range(n):
    for j in range(i+1, n):
        nx_graph.add_edge(board_words[i], board_words[j], weight=w.similarity(board_words[i], board_words[j])+1)
        print(board_words[i],  board_words[j], (w.similarity(board_words[i], board_words[j])+1)**3)

louvain_g = nx.Graph()
louvain_g.add_nodes_from(board_words)
for i in range(n):
    for j in range(i+1, n):
        louvain_g.add_edge(board_words[i], board_words[j], weight=(w.similarity(board_words[i], board_words[j])+1)**10)

partition_object = community.best_partition(louvain_g)
values = [partition_object.get(node) for node in louvain_g.nodes()]

colors = dict([(0, '#0157a1'),
             (1, '#77f431'),
             (2, '#000cb3'),
             (3, '#e4ff3f'),
             (4, '#6213c6'),
             (5, '#1abd00'),
             (6, '#ab39eb'),
             (7, '#00c932'),
             (8, '#e232e8'),
             (9, '#2a9b00')])
for key, value in partition_object.items():
    partition_object[key] = colors[value]

nx.set_node_attributes(nx_graph, partition_object, 'group')



# %%
# nx_graph = nx.cycle_graph(10)
# nx_graph.nodes[1]['title'] = 'Number 1'
# nx_graph.nodes[1]['group'] = 1
# nx_graph.nodes[3]['title'] = 'I belong to a different group!'
# nx_graph.nodes[3]['group'] = 10
# nx_graph.add_node(20, size=20, title='couple', group=2)
# nx_graph.add_node(21, size=15, title='couple', group=2)
# this.body.data.edges._data[i[a]].weight
# nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
nt = Network('900px', '1500px')
# populates the nodes and edges data structures
nt.from_nx(nx_graph)
nt.show_buttons(filter_='physics')
nt.show('nx.html')


