"""tests/test_seq_packing_unit.py — unit tests for squish.seq_packing"""

import numpy as np
import pytest

from squish.seq_packing import (
    PackingConfig,
    SequencePacker,
    PackedBatch,
    PackingStats,
)


# ---------------------------------------------------------------------------
# PackingConfig
# ---------------------------------------------------------------------------

class TestPackingConfig:
    def test_defaults(self):
        cfg = PackingConfig()
        assert cfg.max_packed_length == 8192
        assert cfg.pad_to_multiple == 8
        assert cfg.allow_partial is True

    def test_custom(self):
        cfg = PackingConfig(max_packed_length=512, pad_to_multiple=16)
        assert cfg.max_packed_length == 512

    @pytest.mark.parametrize("field,val", [
        ("max_packed_length", 0),
        ("pad_to_multiple", 0),
    ])
    def test_invalid(self, field, val):
        with pytest.raises(ValueError):
            PackingConfig(**{field: val})


# ---------------------------------------------------------------------------
# PackedBatch properties
# ---------------------------------------------------------------------------

class TestPackedBatch:
    def _batch(self, sequences=None):
        packer = SequencePacker(PackingConfig(max_packed_length=64, pad_to_multiple=1))
        seqs = sequences or [[1, 2, 3], [4, 5], [6]]
        batches = packer.pack(seqs)
        return batches[0]

    def test_is_valid(self):
        b = self._batch()
        assert b.is_valid()

    def test_n_sequences(self):
        b = self._batch([[1, 2], [3]])
        assert b.n_sequences == 2

    def test_content_length(self):
        b = self._batch([[1, 2, 3], [4, 5]])
        assert b.content_length == 5

    def test_padding_ratio_perfect_pack(self):
        cfg = PackingConfig(max_packed_length=6, pad_to_multiple=1)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3], [4, 5, 6]])
        assert batches[0].padding_ratio == 0.0

    def test_padding_ratio_with_pad(self):
        cfg = PackingConfig(max_packed_length=8, pad_to_multiple=4)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3]])  # content=3, padded to 4
        assert batches[0].padding_ratio > 0.0

    def test_attention_mask_no_cross_sequence(self):
        cfg = PackingConfig(max_packed_length=8, pad_to_multiple=1)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3], [4, 5]])
        b = batches[0]
        # Token at position 4 (seq2, idx 0) should NOT attend to position 0 (seq1)
        assert not b.attention_mask[3, 0]

    def test_attention_mask_causal_within_sequence(self):
        cfg = PackingConfig(max_packed_length=8, pad_to_multiple=1)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3]])
        b = batches[0]
        # Within seq1: token 2 (pos 2) attends to pos 0,1,2 but NOT 3
        assert b.attention_mask[2, 0]
        assert b.attention_mask[2, 1]
        assert b.attention_mask[2, 2]

    def test_sequence_offsets_sorted(self):
        b = self._batch([[1], [2], [3]])
        for i in range(len(b.sequence_offsets) - 1):
            assert b.sequence_offsets[i] < b.sequence_offsets[i + 1]


# ---------------------------------------------------------------------------
# SequencePacker
# ---------------------------------------------------------------------------

class TestSequencePacker:
    def test_single_sequence(self):
        packer = SequencePacker(PackingConfig(max_packed_length=16, pad_to_multiple=1))
        batches = packer.pack([[1, 2, 3, 4]])
        assert len(batches) == 1
        assert batches[0].n_sequences == 1

    def test_multiple_sequences_packed(self):
        cfg = PackingConfig(max_packed_length=10, pad_to_multiple=1)
        packer = SequencePacker(cfg)
        # Three 3-token seqs: two fit in first batch, one in second
        batches = packer.pack([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        total_seqs = sum(b.n_sequences for b in batches)
        assert total_seqs == 3

    def test_overlong_sequence_skipped(self):
        cfg = PackingConfig(max_packed_length=4, pad_to_multiple=1, allow_partial=False)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3, 4, 5]])  # length 5 > max 4
        assert len(batches) == 0

    def test_overlong_truncated_when_partial(self):
        cfg = PackingConfig(max_packed_length=4, pad_to_multiple=1, allow_partial=True)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3, 4, 5]])
        assert len(batches) == 1
        assert batches[0].content_length <= 4

    def test_empty_sequences(self):
        packer = SequencePacker(PackingConfig())
        batches = packer.pack([])
        assert len(batches) == 0

    def test_token_ids_type(self):
        packer = SequencePacker(PackingConfig(max_packed_length=32, pad_to_multiple=1))
        batches = packer.pack([[1, 2], [3]])
        assert batches[0].token_ids.dtype == np.int64

    def test_all_seqs_covered(self):
        cfg = PackingConfig(max_packed_length=20, pad_to_multiple=1)
        packer = SequencePacker(cfg)
        seqs = [[1] * 5, [2] * 7, [3] * 3, [4] * 4, [5] * 6]
        batches = packer.pack(seqs)
        total = sum(b.n_sequences for b in batches)
        assert total == 5


# ---------------------------------------------------------------------------
# PackingStats
# ---------------------------------------------------------------------------

class TestPackingStats:
    def test_defaults(self):
        s = PackingStats()
        assert s.packing_efficiency == 0.0
        assert s.mean_sequences_per_batch == 0.0

    def test_record_batches(self):
        cfg = PackingConfig(max_packed_length=10, pad_to_multiple=1)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3], [4, 5], [6]])
        s = PackingStats()
        s.record_batches(batches)
        assert s.total_sequences == 3
        assert s.total_tokens == 6

    def test_packing_efficiency_upper_bound(self):
        cfg = PackingConfig(max_packed_length=6, pad_to_multiple=1)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3], [4, 5, 6]])
        s = PackingStats()
        s.record_batches(batches)
        assert s.packing_efficiency <= 1.0

    def test_padding_ratio_complement(self):
        cfg = PackingConfig(max_packed_length=8, pad_to_multiple=4)
        packer = SequencePacker(cfg)
        batches = packer.pack([[1, 2, 3]])
        s = PackingStats()
        s.record_batches(batches)
        assert abs(s.packing_efficiency + s.padding_ratio - 1.0) < 1e-6

    def test_mean_sequences_per_batch(self):
        s = PackingStats(total_sequences=9, total_batches=3)
        assert s.mean_sequences_per_batch == 3.0
