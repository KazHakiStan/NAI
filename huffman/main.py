import heapq
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx


class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        return heapq.heappop(self.heap)[1]

    def __len__(self):
        return len(self.heap)


class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq):
    pq = PriorityQueue()
    for char, f in freq.items():
        pq.push(HuffmanNode(char, f), f)

    while len(pq) > 1:
        left = pq.pop()
        right = pq.pop()
        merged = HuffmanNode(left=left, right=right, freq=left.freq + right.freq)
        pq.push(merged, merged.freq)

    return pq.pop()


def generate_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}

    if node.char is not None:
        codebook[node.char] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)

    return codebook


def encode(sequence, codebook):
    return ''.join(codebook[char] for char in sequence)


def decode(encoded_sequence, root):
    decoded_output = []
    node = root
    for bit in encoded_sequence:
        node = node.left if bit == '0' else node.right
        if node.char is not None:
            decoded_output.append(node.char)
            node = root

    return ''.join(decoded_output)


def plot_huffman_tree(root):
    def add_edges(graph, node, label='', pos=None, x=0, y=0, layer=1):
        if node:
            if pos is None:
                pos = {node: (x, y)}
            else:
                pos[node] = (x, y)

            if node.left:
                graph.add_edge(node, node.left, label='0')
                l = layer + 1
                add_edges(graph, node.left, label, pos=pos, x=x - 1 / l, y=y - 1, layer=l)

            if node.right:
                graph.add_edge(node, node.right, label='1')
                l = layer + 1
                add_edges(graph, node.right, label, pos=pos, x=x + 1 / l, y=y - 1, layer=l)

            return pos

    graph = nx.DiGraph()
    pos = add_edges(graph, root)
    labels = nx.get_edge_attributes(graph, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, labels={node: f'{node.char}' for node in graph.nodes if node.char},
            node_size=5000, node_color="lightblue", font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


if __name__ == "__main__":
    sequence = "this is an example for huffman encoding"

    freq = Counter(sequence)

    root = build_huffman_tree(freq)

    huffman_codes = generate_huffman_codes(root)

    print("Huffman Codes:")
    for char, code in huffman_codes.items():
        print(f"{char}: {code}")

    encoded_sequence = encode(sequence, huffman_codes)
    print("\nEncoded Sequence:")
    print(encoded_sequence)

    decoded_sequence = decode(encoded_sequence, root)
    print("\nDecoded Sequence:")
    print(decoded_sequence)

    plot_huffman_tree(root)
