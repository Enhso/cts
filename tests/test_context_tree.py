import pytest

from src.cts_core.context_tree import ContextTree, Node


def test_new_node_is_in_its_initial_state():
    node = Node()
    assert (node.count_0, node.count_1) == (0, 0)
    assert node.log_kt == 0.0
    assert node.log_prob == 0.0
    assert node.log_k == node.log_s  # k_c = s_c = 1/2
    assert node.child_0 is None and node.child_1 is None


def test_path_has_depth_plus_one_nodes_leaf_first_root_last():
    tree = ContextTree(3)
    nodes = tree.get_nodes_for_context([1, 0, 1])
    assert len(nodes) == 4
    assert nodes[-1] is tree.root


def test_most_recent_bit_selects_the_first_child_below_the_root():
    tree = ContextTree(3)
    # The context is given most recent first: x_{n-1} = 1, x_{n-2} = 0, x_{n-3} = 0.
    leaf, second, first, root = tree.get_nodes_for_context([1, 0, 0])
    assert first is tree.root.child_1
    assert second is tree.root.child_1.child_0
    assert leaf is tree.root.child_1.child_0.child_0
    assert tree.root.child_0 is None


def test_paths_with_a_shared_recent_bit_share_the_node_below_the_root():
    tree = ContextTree(2)
    a = tree.get_nodes_for_context([1, 0])
    b = tree.get_nodes_for_context([1, 1])
    assert a[1] is b[1]
    assert a[0] is not b[0]


def test_same_context_returns_the_same_nodes():
    tree = ContextTree(4)
    first = tree.get_nodes_for_context([0, 1, 1, 0])
    second = tree.get_nodes_for_context([0, 1, 1, 0])
    assert all(x is y for x, y in zip(first, second))


def test_depth_zero_tree_is_just_the_root():
    tree = ContextTree(0)
    assert tree.get_nodes_for_context([]) == [tree.root]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        ContextTree(-1)
    tree = ContextTree(2)
    with pytest.raises(ValueError):
        tree.get_nodes_for_context([0, 1, 0])  # too long
    with pytest.raises(ValueError):
        tree.get_nodes_for_context([0])  # too short
    with pytest.raises(ValueError):
        tree.get_nodes_for_context([0, 2])


def test_save_and_restore_round_trip():
    node = Node()
    state = node.save()
    node.observe(1)
    node.log_prob = -3.0
    assert node.save() != state
    node.restore(state)
    assert node.save() == state
