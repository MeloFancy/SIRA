""" Doc Cluster """
import spacy
import numpy as np

import networkx as nx
from chinese_whispers import chinese_whispers
from tqdm import tqdm


def create_graph(doc_embeddings, threshold=0.75):
    nodes = []
    edges = []

    docs, domains = [], []
    for domain in doc_embeddings:
        docs.extend(doc_embeddings[domain])
        domains += [domain]*len(doc_embeddings[domain])

    if len(docs) <= 1:
        print("No enough docs to cluster!")
        return []

    for idx, embedding_to_check in tqdm(enumerate(docs)):
        # Adding node of doc embedding
        node_id = idx + 1

        node = (node_id, {'text': embedding_to_check, 'embedding': embedding_to_check.vector, 'domain': domains[idx]})
        nodes.append(node)

        # doc embeddings to compare
        if (idx + 1) >= len(docs):
            # Node is last element, don't create edge
            break

        compare_embeddings = docs[idx + 1:]
        distances = doc_distance(compare_embeddings, embedding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


def compute_embeddings(domain_docs):
    nlp = spacy.load('en_use_md')

    doc_embeddings = {}
    for app in domain_docs:
        doc_embeddings[app] = [nlp(doc) for doc in domain_docs[app]]
        # print(doc_embeddings[app][0], doc_embeddings[app][0].vector, doc_embeddings[app][0].vector.shape)
    return doc_embeddings


def doc_distance(doc_embeddings, doc_to_compare):
    if len(doc_embeddings) == 0:
        return np.empty((0))

    return np.array([doc_to_compare.similarity(doc) for doc in doc_embeddings])


def main():
    domain_docs = {'communication': ['xxx'], 'social': ['xxx'], 'finance': ['xxx']}
    doc_embeddings = compute_embeddings(domain_docs)
    G = create_graph(doc_embeddings, threshold=0.5)
    # Perform clustering of G, parameters weighting and seed can be omitted
    chinese_whispers(G, weighting='top', iterations=20)
    for node in G.nodes():
        print(str(G.nodes[node]['text']) + ', ' + str(G.nodes[node]['label']))


if __name__ == '__main__':
    """ Entry point """
    main()
