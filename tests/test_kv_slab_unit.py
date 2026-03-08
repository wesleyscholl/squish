"""tests/test_kv_slab_unit.py — 100% coverage for squish/kv_slab.py"""
import threading
import numpy as np
import pytest

from squish.kv_slab import KVPage, KVSlabAllocator


# ---------------------------------------------------------------------------
# KVPage
# ---------------------------------------------------------------------------

class TestKVPage:
    def _make_page(self, page_id=0, page_size=4, n_layers=2, n_heads=2, head_dim=8):
        keys   = np.zeros((n_layers, page_size, n_heads, head_dim), dtype=np.float16)
        values = np.zeros_like(keys)
        return KVPage(page_id=page_id, page_size=page_size, keys=keys, values=values)

    def test_initial_state(self):
        page = self._make_page(page_id=3, page_size=8)
        assert page.page_id   == 3
        assert page.page_size == 8
        assert page.n_filled  == 0

    def test_is_full_false_when_empty(self):
        page = self._make_page(page_size=4)
        assert page.is_full() is False

    def test_is_full_true_when_full(self):
        page = self._make_page(page_size=4)
        page.n_filled = 4
        assert page.is_full() is True

    def test_remaining_decreases_as_filled(self):
        page = self._make_page(page_size=4)
        assert page.remaining() == 4
        page.n_filled = 3
        assert page.remaining() == 1

    def test_reset_clears_n_filled(self):
        page = self._make_page(page_size=4)
        page.n_filled = 4
        page.reset()
        assert page.n_filled == 0

    def test_keys_values_are_numpy_arrays(self):
        page = self._make_page()
        assert isinstance(page.keys,   np.ndarray)
        assert isinstance(page.values, np.ndarray)


# ---------------------------------------------------------------------------
# KVSlabAllocator
# ---------------------------------------------------------------------------

class TestKVSlabAllocator:
    def _make_slab(self, n_pages=4, page_size=4, n_layers=2, n_heads=2, head_dim=8,
                   dtype=np.float16):
        return KVSlabAllocator(
            n_pages=n_pages, page_size=page_size,
            n_layers=n_layers, n_heads=n_heads,
            head_dim=head_dim, dtype=dtype,
        )

    def test_initial_all_free(self):
        slab = self._make_slab(n_pages=8)
        assert slab.n_free() == 8
        assert slab.n_used() == 0

    def test_alloc_returns_page(self):
        slab = self._make_slab()
        page = slab.alloc()
        assert page is not None
        assert isinstance(page, KVPage)

    def test_alloc_decrements_free(self):
        slab = self._make_slab(n_pages=4)
        slab.alloc()
        assert slab.n_free() == 3
        assert slab.n_used() == 1

    def test_alloc_returns_none_when_exhausted(self):
        slab = self._make_slab(n_pages=2)
        slab.alloc()
        slab.alloc()
        page = slab.alloc()   # OOM
        assert page is None
        assert slab.n_used() == 2

    def test_free_returns_page_to_pool(self):
        slab = self._make_slab(n_pages=2)
        p1 = slab.alloc()
        p2 = slab.alloc()
        slab.free(p1)
        assert slab.n_free() == 1
        assert slab.n_used() == 1

    def test_free_many_returns_batch(self):
        slab = self._make_slab(n_pages=4)
        pages = [slab.alloc() for _ in range(4)]
        assert slab.n_free() == 0
        slab.free_many(pages)
        assert slab.n_free() == 4

    def test_alloc_resets_page(self):
        slab = self._make_slab(n_pages=1)
        page = slab.alloc()
        page.n_filled = 3
        slab.free(page)
        page2 = slab.alloc()
        assert page2.n_filled == 0   # reset() was called

    def test_alloc_count_tracked(self):
        slab = self._make_slab(n_pages=4)
        slab.alloc()
        slab.alloc()
        assert slab._alloc_count == 2

    def test_free_count_tracked(self):
        slab = self._make_slab(n_pages=4)
        p = slab.alloc()
        slab.free(p)
        assert slab._free_count == 1

    def test_free_many_count_tracked(self):
        slab = self._make_slab(n_pages=4)
        pages = [slab.alloc() for _ in range(3)]
        slab.free_many(pages)
        assert slab._free_count == 3

    def test_memory_bytes_nonzero(self):
        slab = self._make_slab(n_pages=4, page_size=8, n_layers=2, n_heads=2,
                               head_dim=8, dtype=np.float16)
        mb = slab.memory_bytes()
        # 4 pages × 8 tokens × 2 layers × 2 heads × 8 dim × 2 bytes × 2 (K+V)
        expected = 4 * 2 * 8 * 2 * 8 * 2 * 2
        assert mb == expected

    def test_stats_returns_dict(self):
        slab  = self._make_slab(n_pages=4)
        _page = slab.alloc()
        stats = slab.stats()
        assert stats["n_pages"]     == 4
        assert stats["n_used"]      == 1
        assert stats["n_free"]      == 3
        assert stats["alloc_total"] == 1
        assert stats["memory_mb"]   > 0.0

    def test_repr_contains_key_info(self):
        slab = self._make_slab(n_pages=4)
        r = repr(slab)
        assert "KVSlabAllocator" in r
        assert "n_pages=4"       in r

    def test_pages_are_views_into_slab(self):
        """Pages should be numpy views, not copies."""
        slab = self._make_slab(n_pages=2)
        p = slab.alloc()
        # Mutating the page's keys should also mutate the slab
        p.keys[:] = 1.0
        assert slab._slab_k[p.page_id].sum() != 0

    def test_thread_safety_concurrent_alloc(self):
        """Thread-safe: 4 threads each alloc 1 page from a 4-page slab."""
        slab   = self._make_slab(n_pages=4)
        pages  = []
        errors = []
        lock   = threading.Lock()

        def worker():
            p = slab.alloc()
            with lock:
                if p is not None:
                    pages.append(p)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(pages) == 4
        assert slab.n_free() == 0

    def test_n_used_correct_after_alloc_free_cycle(self):
        slab  = self._make_slab(n_pages=3)
        pages = [slab.alloc() for _ in range(3)]
        assert slab.n_used() == 3
        slab.free(pages[0])
        assert slab.n_used() == 2
        slab.free_many([pages[1], pages[2]])
        assert slab.n_used() == 0
