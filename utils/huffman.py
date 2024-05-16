import torch
import heapq
from collections import defaultdict


class HuffmanCoding:
    """
    Huffman Coding class for compressing and decompressing PyTorch tensors.
    """

    class Node:
        """
        Node class to represent elements in the Huffman tree.
        """

        def __init__(self, value, freq):
            """
            Initialize a Node.

            Args:
                value: The value associated with the node.
                freq: The frequency (count) of the value.
            """
            self.value = value
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            """
            Comparison method for sorting nodes in the heap.

            Args:
                other: Another node to compare with.

            Returns:
                True if this node has a lower frequency than the other.
            """
            return self.freq < other.freq

    def __init__(self):
        """
        Initialize a HuffmanCoding instance.
        """
        self.heap = []  # Priority queue for building the Huffman tree
        self.codes = {}  # Dictionary to store Huffman codes
        self.reverse_mapping = {}  # Dictionary to map codes to values

    def build_heap(self, tensor):
        """
        Build a heap of nodes based on the frequency of values in the tensor.

        Args:
            tensor: PyTorch tensor to be compressed.
        """
        frequency = defaultdict(int)
        for value in tensor.view(-1):
            value = int(value)
            frequency[value] += 1

        for value, freq in frequency.items():
            node = self.Node(value, freq)
            heapq.heappush(self.heap, node)

    def build_tree(self):
        """
        Build the Huffman tree by merging nodes in the heap.
        """
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged_node = self.Node(None, node1.freq + node2.freq)
            merged_node.left = node1
            merged_node.right = node2

            heapq.heappush(self.heap, merged_node)

    def build_codes_helper(self, root, current_code):
        """
        Recursively build Huffman codes and the reverse mapping.

        Args:
            root: The current node in the tree.
            current_code: The current Huffman code being constructed.
        """
        if root is None:
            return

        if root.value is not None:
            self.codes[root.value] = current_code
            self.reverse_mapping[current_code] = root.value

        self.build_codes_helper(root.left, current_code + "0")
        self.build_codes_helper(root.right, current_code + "1")

    def build_codes(self):
        """
        Build Huffman codes and the reverse mapping after the tree is built.
        """
        root = heapq.heappop(self.heap)
        current_code = ""
        self.build_codes_helper(root, current_code)

    def encode(self, tensor):
        """
        Encode the input tensor using Huffman codes.

        Args:
            tensor: PyTorch tensor to be encoded.

        Returns:
            The encoded text as a binary string.
        """
        encoded_text = ""
        for value in tensor.view(-1):
            value = int(value)
            encoded_text += self.codes[value]
        return encoded_text

    def decode(self, encoded_text):
        """
        Decode the encoded text into a PyTorch tensor.

        Args:
            encoded_text: The encoded text as a binary string.

        Returns:
            The decoded PyTorch tensor.
        """
        current_code = ""
        decoded_tensor = []
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                value = self.reverse_mapping[current_code]
                decoded_tensor.append(value)
                current_code = ""
        return torch.tensor(decoded_tensor)

    def compress(self, tensor):
        """
        Compress a PyTorch tensor using Huffman coding.

        Args:
            tensor: PyTorch tensor to be compressed.

        Returns:
            The compressed text as a binary string.
        """
        self.build_heap(tensor)
        self.build_tree()
        self.build_codes()
        encoded_text = self.encode(tensor)
        ori_shape = tensor.shape
        return encoded_text, ori_shape, tensor.device

    def decompress(self, encoded_text, ori_shape, device):
        """
        Decompress the encoded text into a PyTorch tensor.

        Args:
            encoded_text: The encoded text as a binary string.
            ori_shape: The original shape of text.
            device: The torch device (cuda).

        Returns:
            The decompressed PyTorch tensor.
        """
        decoded_tensor = self.decode(encoded_text).to(device)
        return decoded_tensor.view(ori_shape)
