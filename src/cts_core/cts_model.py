"""Context Tree Weighting (CTW) and Context Tree Switching (CTS) as sequential predictors.

Both models read a bit sequence one bit at a time:

    p = model.predict()      # P(next bit = 1 | history)
    model.update(bit)        # observe the bit

and expose `log_joint`, the natural log of the model's probability of the whole
sequence so far, so `code_length` (-log2 of that probability) is what an
arithmetic coder would need. All node statistics are kept as logarithms.

The first `depth` bits have no full context. Instead of coding them separately,
as the paper does, the context is padded with zeros as if the sequence had been
preceded by `depth` zero bits.
"""
from __future__ import annotations

from collections import deque
from math import exp, log, log1p

from .context_tree import LOG_HALF, ContextTree, Node
from .kt_estimator import LN2

NEG_INF = float("-inf")


def log_add(x: float, y: float) -> float:
    """ln(e^x + e^y), safe when either argument is -inf."""
    if x < y:
        x, y = y, x
    if y == NEG_INF:
        return x
    return x + log1p(exp(y - x))


def safe_log(x: float) -> float:
    return log(x) if x > 0 else NEG_INF


def switching_rate(t: int) -> float:
    """The switching rate alpha_t = 1/t.

    The paper uses this for every context, with t the position in the whole
    sequence, not the number of times the context has been visited (section 3.1).
    """
    return 1.0 / t


class _TreeModel:
    """What CTW and CTS share: the tree, the context and the predict/update protocol."""

    def __init__(self, depth: int):
        self.depth = depth
        self.tree = ContextTree(depth)
        self.n = 0  # bits seen so far
        # x_{n-1}, x_{n-2}, ..., x_{n-depth}, most recent first.
        self._context: deque[int] = deque([0] * depth, maxlen=depth)
        self._path: list[Node] | None = None

    @property
    def log_joint(self) -> float:
        """ln CTW_D(x_1:n) or ln CTS_D(x_1:n): the root's mixed probability."""
        return self.tree.root.log_prob

    @property
    def code_length(self) -> float:
        """-log2 of the joint probability, in bits."""
        return -self.log_joint / LN2

    def _nodes(self) -> list[Node]:
        """Path for the current context, leaf first, root last."""
        if self._path is None:
            self._path = self.tree.get_nodes_for_context(self._context)
        return self._path

    def _apply(self, nodes: list[Node], bit: int, n: int) -> None:
        """Update every node on the path for `bit`, the n-th bit of the sequence."""
        raise NotImplementedError

    def update(self, bit: int) -> float:
        """Observe `bit`. Returns ln P(bit | history), as it was before the update."""
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        before = self.log_joint
        self.n += 1
        self._apply(self._nodes(), bit, self.n)
        self._context.appendleft(bit)
        self._path = None
        return self.log_joint - before

    def log_prob(self, bit: int) -> float:
        """ln P(next bit = `bit` | history), from ln [P(x_1:n bit) / P(x_1:n)].

        The update is applied to the current path and then rolled back, so this
        cannot disagree with `update`. Nodes created for the context stay in
        the tree, but a fresh node is a no-op.
        """
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        nodes = self._nodes()
        saved = [node.save() for node in nodes]
        before = self.log_joint
        self._apply(nodes, bit, self.n + 1)
        result = self.log_joint - before
        for node, state in zip(nodes, saved):
            node.restore(state)
        return result

    def predict(self) -> float:
        """P(next bit = 1 | history)."""
        return exp(self.log_prob(1))


class CTWModel(_TreeModel):
    """Context Tree Weighting, with the recursion of Eq. 14:

        CTW^c_D = 1/2 xi_KT(x^c) + 1/2 CTW^{0c}_{D-1} CTW^{1c}_{D-1}     (D > 0)
        CTW^c_0 = xi_KT(x^c)
    """

    def _apply(self, nodes, bit, n):
        # Leaf first, so a node reads children that have already been updated.
        for height, node in enumerate(nodes):
            node.observe(bit)
            if height == 0:
                node.log_prob = node.log_kt
            else:
                log_children = (
                    (node.child_0.log_prob if node.child_0 else 0.0)
                    + (node.child_1.log_prob if node.child_1 else 0.0)
                )
                node.log_prob = LOG_HALF + log_add(node.log_kt, log_children)


class CTSModel(_TreeModel):
    """Context Tree Switching, with the update equations of section 3.1:

        CTS^c_D(x_1:n) <- k_c xi_KT(x_n | x^c_<n) + s_c z^c_D(x_n | x_<n)
        k_c <- alpha CTS^c_D(x_1:n) + (1 - 2 alpha) k_c xi_KT(x_n | x^c_<n)
        s_c <- alpha CTS^c_D(x_1:n) + (1 - 2 alpha) s_c z^c_D(x_n | x_<n)

    with alpha = alpha_{n+1} and z the ratio by which the child on the path
    changed its probability. At depth D the node is a leaf: CTS^c_0 = xi_KT.
    """

    def _apply(self, nodes, bit, n):
        alpha = switching_rate(n + 1)
        log_alpha = safe_log(alpha)
        log_keep = safe_log(1 - 2 * alpha)
        log_z = 0.0
        for height, node in enumerate(nodes):
            log_kt_cond = node.observe(bit)
            if height == 0:
                new = node.log_kt
            else:
                kt_term = node.log_k + log_kt_cond
                switch_term = node.log_s + log_z
                new = log_add(kt_term, switch_term)
                node.log_k = log_add(log_alpha + new, log_keep + kt_term)
                node.log_s = log_add(log_alpha + new, log_keep + switch_term)
            log_z = new - node.log_prob
            node.log_prob = new
