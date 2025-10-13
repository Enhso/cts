from __future__ import annotations
from typing import Optional

class Node:
    """
    Represents a single node in the Context Tree.

    Each node corresponds to a specific context (a binary suffix) and stores
    the necessary statistics and weights for the CTS algorithm.
    """
    def __init__(self):
        # --- Statistics for the Krichevsky-Trofimov (KT) estimator ---
        # a_c: Count of '0's seen in this context.
        self.count_0: int = 0
        # b_c: Count of '1's seen in this context.
        self.count_1: int = 0

        # --- Weights for the Context Tree Switching (CTS) mechanism ---
        # k_c: The weight associated with the KT estimator (the "leaf" model).
        self.weight_k: float = 0.5
        # s_c: The weight associated with the switched/mixed children nodes.
        self.weight_s: float = 0.5

        # --- Stored Probabilities of the sequence seen in this context ---
        # ξ_KT(x_c,1:n_c): The total KT probability for the subsequence seen so far.
        # Initialized to 1.0, as the probability of an empty sequence is 1.
        self.prob_kt: float = 1.0
        
        # CTS_D^c(x_c,1:n_c): The total CTS probability for the subsequence.
        # Initialized to 1.0.
        self.prob_cts: float = 1.0

        # --- Tree Structure ---
        # Pointers to children nodes. A '0' bit traverses to child_0, '1' to child_1.
        self.child_0: Optional[Node] = None
        self.child_1: Optional[Node] = None

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation for debugging."""
        return (
            f"Node(counts=({self.count_0}, {self.count_1}), "
            f"weights=({self.weight_k:.3f}, {self.weight_s:.3f}), "
            f"probs=(kt={self.prob_kt:.3f}, cts={self.prob_cts:.3f}))"
        )
    
class ContextTree:
    """
    Manages the overall Context Tree data structure.

    This class holds the root of the tree and provides the main interface for
    traversing it. It dynamically creates nodes as new contexts are seen.
    """
    def __init__(self, depth: int):
        """
        Initializes the Context Tree.

        Args:
            depth (int): The maximum depth D of the contexts to track.
        """
        if depth < 0:
            raise ValueError("Tree depth cannot be negative.")
        self.depth = depth
        self.root = Node()

    def get_nodes_for_context(self, context: List[int]) -> List[Node]:
        """
        Traverses the tree to find or create all nodes for a given context.

        This method follows the path defined by the context, creating nodes
        if they do not exist. It returns the list of nodes along this path,
        ordered from the deepest node (the leaf) back to the root. This is
        the exact order required for the CTS update algorithm.

        Args:
            context (List[int]): The sequence of bits representing the context,
                                 e.g., [1, 0, 1] for context '101'.

        Returns:
            List[Node]: A list of Node objects from leaf to root.
        """
        # The context is defined as x_{n-1}x_{n-2}...x_{n-D}.
        # We only care about the last `self.depth` bits.
        effective_context = context[-self.depth:]

        current_node = self.root
        path_nodes = [current_node]

        # Traverse the tree from the root, following the path of the context.
        for bit in effective_context:
            if bit == 0:
                # If the child node for a '0' doesn't exist, create it.
                if current_node.child_0 is None:
                    current_node.child_0 = Node()
                current_node = current_node.child_0
            elif bit == 1:
                # If the child node for a '1' doesn't exist, create it.
                if current_node.child_1 is None:
                    current_node.child_1 = Node()
                current_node = current_node.child_1
            else:
                raise ValueError("Context must contain only 0s and 1s.")
            
            path_nodes.append(current_node)

        # The paper's update rule processes nodes from the specific context
        # back to the empty context (root). Reversing the list achieves this.
        return path_nodes[::-1]