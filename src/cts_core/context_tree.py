"""The context tree shared by CTW and CTS (paper sections 2.4.3 and 3.1)."""
from __future__ import annotations

from math import log
from typing import Optional, Sequence

from .kt_estimator import kt_prob

LOG_HALF = log(0.5)


class Node:
    """One node of the context tree, i.e. one context c.

    Everything is kept in natural-log space so long sequences cannot underflow.
    The node sees only the subsequence x^c of bits that occurred in its context.
    """

    __slots__ = (
        "count_0", "count_1", "log_kt", "log_k", "log_s", "log_prob",
        "child_0", "child_1",
    )

    def __init__(self):
        # a_c and b_c: the number of 0s and 1s seen in this context.
        self.count_0: int = 0
        self.count_1: int = 0
        # ln of the KT probability of x^c. The empty sequence has probability 1.
        self.log_kt: float = 0.0
        # CTS only: ln k_c and ln s_c, the weights on "use KT here" and "use the
        # children". A new node starts at k_c = s_c = 1/2.
        self.log_k: float = LOG_HALF
        self.log_s: float = LOG_HALF
        # ln of the node's mixed probability of x^c: CTW^c_D or CTS^c_D.
        self.log_prob: float = 0.0
        # The child reached by a 0 (resp. 1) in the next-older position.
        self.child_0: Optional[Node] = None
        self.child_1: Optional[Node] = None

    def observe(self, bit: int) -> float:
        """Count `bit` in this context. Returns ln of its KT conditional probability."""
        log_cond = log(kt_prob(bit, self.count_0, self.count_1))
        if bit:
            self.count_1 += 1
        else:
            self.count_0 += 1
        self.log_kt += log_cond
        return log_cond

    def save(self) -> tuple:
        """Snapshot of every field that an update can change."""
        return (self.count_0, self.count_1, self.log_kt,
                self.log_k, self.log_s, self.log_prob)

    def restore(self, state: tuple) -> None:
        (self.count_0, self.count_1, self.log_kt,
         self.log_k, self.log_s, self.log_prob) = state

    def __repr__(self) -> str:
        return (
            f"Node(counts=({self.count_0}, {self.count_1}), "
            f"log_kt={self.log_kt:.3f}, log_prob={self.log_prob:.3f})"
        )


class ContextTree:
    """A binary tree of Nodes, created on demand as contexts are seen.

    A context is the preceding bits, most recent first: x_{n-1}, x_{n-2}, ...
    The most recent bit therefore selects the first child below the root and
    each further bit looks one step deeper into the past.
    """

    def __init__(self, depth: int):
        if depth < 0:
            raise ValueError("Tree depth cannot be negative.")
        self.depth = depth
        self.root = Node()

    def get_nodes_for_context(self, context: Sequence[int]) -> list[Node]:
        """Return the nodes on the path for `context`, leaf first and root last.

        `context` holds exactly `depth` bits, most recent first. Missing nodes
        are created; a new node is in its initial state, which does not change
        any probability.
        """
        if len(context) != self.depth:
            raise ValueError(f"Context must have exactly {self.depth} bits.")
        node = self.root
        path = [node]
        for bit in context:
            if bit == 0:
                if node.child_0 is None:
                    node.child_0 = Node()
                node = node.child_0
            elif bit == 1:
                if node.child_1 is None:
                    node.child_1 = Node()
                node = node.child_1
            else:
                raise ValueError("Context must contain only 0s and 1s.")
            path.append(node)
        path.reverse()
        return path
