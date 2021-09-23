# %% Originaly from sna_hinter
# board_size = state.board_size#
# vis_graph = nx.Graph()
# vis_graph.add_nodes_from(state.all_words)
# louvain = nx.Graph(vis_graph)
# for i in range(board_size):
#     v = state.all_words[i]
#     for j in range(i + 1, board_size):
#         u = state.all_words[j]
#         distance = self.model.similarity(v, u) + 1
#         if distance > 1.1:
#             vis_graph.add_edge(v, u, weight=distance)
#         louvain_weight = distance ** 10
#         louvain.add_edge(v, u, weight=louvain_weight)
#
# word_to_group: Dict[str, int] = community.best_partition(louvain)
# # self.board_data.cluster = self.board_data.index.map(word_to_group)
#
# group_to_words = _invert_dict(word_to_group)
# # group_grades = {}
# for group, words in group_to_words.items():
#     vectors = self.model[words]
#     average = sum(vectors)
#     similarities = self.model.most_similar(average)
#     filtered_similarities = filter_similarities(similarities=similarities, words_to_filter_out=words)
#     print(f"\n\nFor the word group: {words}, got:")
#     pretty_print_similarities(filtered_similarities)
# # values = [partition_object.get(node) for node in louvain.nodes()]
# # print(f"Values are: {values}")
#
# nx.set_node_attributes(vis_graph, word_to_group, "group")
# render(vis_graph)
# return Hint("hi", 2)