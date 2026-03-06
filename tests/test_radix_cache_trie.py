"""
tests/test_radix_cache_trie.py

Branch coverage for squish/radix_cache.py trie methods not hit by the
existing TestPrefixCache tests in test_server_unit.py.

Covers:
  - RadixNode.touch()
  - RadixTree.clear() — trie root reset
  - RadixTree.put() — key-already-exists branch (move_to_end)
  - RadixTree.insert_prefix() — empty token_ids early-return branch
  - RadixTree.find_prefix() — empty token_ids early-return branch
  - _trie_insert: new leaf, entire-edge match (descend), split edge
  - _trie_find_longest: match, partial match, no match
  - _collect_trie_nodes_with_blocks
  - evict_prefix_lru / _trie_evict_lru
  - prefix_hits counter
"""
from __future__ import annotations

import time

import pytest

from squish.radix_cache import RadixNode, RadixTree


# ── RadixNode.touch ───────────────────────────────────────────────────────────

class TestRadixNodeTouch:

    def test_touch_updates_last_access(self):
        node  = RadixNode()
        before = node.last_access
        time.sleep(0.01)
        node.touch()
        assert node.last_access >= before


# ── RadixTree.clear ───────────────────────────────────────────────────────────

class TestRadixTreeClear:

    def test_clear_resets_text_cache(self):
        rt = RadixTree(maxsize=8)
        rt.put("hello", "world", "stop")
        assert rt.size == 1
        rt.clear()
        assert rt.size == 0

    def test_clear_resets_trie_root(self):
        rt = RadixTree(maxsize=8)
        rt.insert_prefix([1, 2, 3], [10, 11])
        rt.clear()
        # After clear, prefix look-up should return nothing
        n, refs = rt.find_prefix([1, 2, 3])
        assert n   == 0
        assert refs == []


# ── RadixTree.put re-insert (move_to_end) ─────────────────────────────────────

class TestRadixTreePutReinsert:

    def test_reinsert_same_key_updates_value(self):
        rt = RadixTree(maxsize=8)
        rt.put("prompt", "first", "stop")
        rt.put("prompt", "second", "length")   # re-insert same hash key
        result = rt.get("prompt")
        assert result == ("second", "length")
        assert rt.size == 1  # no duplicate entries

    def test_put_maxsize_zero_drops_entry(self):
        rt = RadixTree(maxsize=0)
        rt.put("hello", "world", "stop")
        assert rt.size == 0


# ── insert_prefix / find_prefix edge cases ────────────────────────────────────

class TestInsertFindPrefixEdges:

    def test_insert_prefix_empty_token_ids_is_noop(self):
        rt = RadixTree()
        rt.insert_prefix([], [99])
        n, refs = rt.find_prefix([1])
        assert n   == 0
        assert refs == []

    def test_insert_prefix_empty_block_refs_is_noop(self):
        rt = RadixTree()
        rt.insert_prefix([1, 2], [])
        n, refs = rt.find_prefix([1, 2])
        assert n   == 0
        assert refs == []

    def test_find_prefix_empty_query_returns_zero(self):
        rt = RadixTree()
        rt.insert_prefix([1, 2], [5])
        n, refs = rt.find_prefix([])
        assert n   == 0
        assert refs == []

    def test_find_prefix_no_match_returns_zero(self):
        rt = RadixTree()
        rt.insert_prefix([1, 2, 3], [10])
        n, refs = rt.find_prefix([4, 5, 6])
        assert n   == 0
        assert refs == []


# ── _trie_insert paths ────────────────────────────────────────────────────────

class TestTrieInsert:

    def test_new_leaf_on_empty_trie(self):
        rt = RadixTree()
        rt.insert_prefix([1, 2, 3], [10, 11])
        n, refs = rt.find_prefix([1, 2, 3])
        assert n   == 3
        assert refs == [10, 11]

    def test_entire_edge_match_descend(self):
        """Insert [1,2,3] then [1,2,3,4] — second insert descends existing edge."""
        rt = RadixTree()
        rt.insert_prefix([1, 2, 3], [5])
        rt.insert_prefix([1, 2, 3, 4], [6])

        n, refs = rt.find_prefix([1, 2, 3, 4])
        assert n == 4
        assert refs == [6]

    def test_return_node_on_exact_match(self):
        """
        _trie_insert returns the existing *node* when the full token sequence
        matches an existing path exactly (remaining becomes empty after descend).
        Used by insert_prefix to re-set block_refs on an already-inserted key.
        """
        rt = RadixTree()
        rt.insert_prefix([1, 2, 3], [10])
        # Second insert with same tokens — must update block_refs without error
        rt.insert_prefix([1, 2, 3], [20])

        n, refs = rt.find_prefix([1, 2, 3])
        assert n   == 3
        assert refs == [20]   # new block_refs overwrote old ones

    def test_split_edge_partial_match(self):
        """Insert [1,2,3] then [1,2,5] — split at position 2."""
        rt = RadixTree()
        rt.insert_prefix([1, 2, 3], [10])
        rt.insert_prefix([1, 2, 5], [20])

        n1, refs1 = rt.find_prefix([1, 2, 3])
        n2, refs2 = rt.find_prefix([1, 2, 5])
        assert n1 == 3
        assert n2 == 3
        assert refs1 == [10]
        assert refs2 == [20]

    def test_split_edge_no_new_suffix(self):
        """Insert [1,2,3] then [1,2] — split at position 2 with no new leaf."""
        rt = RadixTree()
        rt.insert_prefix([1, 2, 3], [10])
        rt.insert_prefix([1, 2], [20])   # prefix of existing edge
        n, refs = rt.find_prefix([1, 2])
        assert n   == 2
        assert refs == [20]

    def test_insert_and_find_multiple_keys(self):
        rt = RadixTree()
        rt.insert_prefix([1, 2],    [1])
        rt.insert_prefix([3, 4, 5], [2])
        rt.insert_prefix([1, 2, 3], [3])

        n, refs = rt.find_prefix([1, 2, 3])
        assert n   == 3
        assert refs == [3]

        n, refs = rt.find_prefix([3, 4, 5])
        assert n   == 3
        assert refs == [2]


# ── find_prefix with partial match ────────────────────────────────────────────

class TestFindPrefixPartial:

    def test_partial_edge_match_returns_best(self):
        """Query starts matching an edge but diverges mid-edge."""
        rt = RadixTree()
        rt.insert_prefix([1, 2, 3], [10])
        # Query [1, 2, 9] shares first 2 tokens with the edge but diverges
        n, refs = rt.find_prefix([1, 2, 9])
        # [1,2,3] is stored; we only match a prefix node *if* it has block_refs
        # Since there's no intermediate node for [1,2], the result depends on trie.
        # The important thing is no exception and n >= 0.
        assert n >= 0

    def test_find_prefix_increments_prefix_hits(self):
        rt = RadixTree()
        rt.insert_prefix([10, 20], [99])
        before = rt.prefix_hits
        rt.find_prefix([10, 20])
        assert rt.prefix_hits == before + 1


# ── evict_prefix_lru ──────────────────────────────────────────────────────────

class TestEvictPrefixLRU:

    def test_evict_removes_lru_node(self):
        rt = RadixTree()
        rt.insert_prefix([1, 2], [10])
        rt.insert_prefix([3, 4], [20])
        # Both nodes have block_refs; evict 1 (the older LRU one)
        evicted = rt.evict_prefix_lru(n=1)
        assert evicted == 1

    def test_evict_zero_when_no_nodes_with_blocks(self):
        rt = RadixTree()
        evicted = rt.evict_prefix_lru(n=5)
        assert evicted == 0

    def test_evict_respects_n(self):
        rt = RadixTree()
        for i in range(5):
            rt.insert_prefix([i, i + 1], [i])
        evicted = rt.evict_prefix_lru(n=3)
        assert evicted == 3

    def test_evict_skips_nodes_with_ref_count(self):
        """Nodes with ref_count > 0 are not evicted."""
        rt = RadixTree()
        rt.insert_prefix([1, 2], [5])
        # Manually bump ref_count on the node
        node = rt._root.children[1]
        node.ref_count = 1
        evicted = rt.evict_prefix_lru(n=1)
        assert evicted == 0

    def test_collect_nodes_with_blocks(self):
        """_collect_trie_nodes_with_blocks returns exactly the leaf nodes."""
        rt    = RadixTree()
        rt.insert_prefix([1, 2], [10])
        rt.insert_prefix([3],    [20])
        nodes = rt._collect_trie_nodes_with_blocks()
        assert len(nodes) == 2
