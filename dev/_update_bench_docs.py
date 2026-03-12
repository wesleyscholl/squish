#!/usr/bin/env python3
"""Update benchmark docs with fresh measurement numbers from JSON results."""
import json
import pathlib


def update_wave13_14():
    d = json.loads(pathlib.Path("dev/results/wave13_14_bench.json").read_text())
    p = pathlib.Path("docs/benchmark_wave13_14.md")
    t = p.read_text()

    subs = {
        "| DuoAttention | `store_kv()` latency | 0.89 µs | per token |":
            f"| DuoAttention | `store_kv()` latency | {d['duo_attention']['store_kv_mean_us']:.2f} µs | per token |",
        "| ShadowKV | `store()` 128 tokens | 2440.8 µs | 16/64 rank |":
            f"| ShadowKV | `store()` 128 tokens | {d['shadow_kv']['store_mean_us']:.1f} µs | 16/64 rank |",
        "| PQCache | retrieve top-32 (256 tok) | 348.0 µs | ADC search |":
            f"| PQCache | retrieve top-32 (256 tok) | {d['pq_cache']['retrieve_top32_us']:.1f} µs | ADC search |",
        "| TokenMerging | `merge()` seq=512 | 748.0 µs | r=16 pairs |":
            f"| TokenMerging | `merge()` seq=512 | {d['token_merging']['merge_512_us']:.1f} µs | r=16 pairs |",
        "| DFloat11 | compress 65K BF16 wts | 48.6 ms | 1.000× (16.00 bits/wt) |":
            f"| DFloat11 | compress 65K BF16 wts | {d['dfloat11']['compress_64k_ms']:.1f} ms | 1.000× (16.00 bits/wt) |",
        "| RANSCodec | encode/decode n=4096 | 2887.6/4223.9 µs | 0.485 bytes/sym (entropy=1.846 bits) |":
            (f"| RANSCodec | encode/decode n=4096 | "
             f"{d['rans_codec']['encode_4096_us']:.1f}/{d['rans_codec']['decode_4096_us']:.1f} µs | "
             f"{d['rans_codec']['bytes_per_symbol']:.3f} bytes/sym "
             f"(entropy={d['rans_codec']['entropy_bits']:.3f} bits) |"),
        "| SqueezeLLM | INT3 128×128 quant | 372.5 ms | 0.312× · SNR=15.8 dB |":
            f"| SqueezeLLM | INT3 128×128 quant | {d['squeeze_llm']['quantize_128x128_ms']:.1f} ms | 0.312× · SNR={d['squeeze_llm']['snr_db']:.1f} dB |",
        "| NF4 | quantize_nf4() 128×128 | 788.8 µs | MSE=0.00835 · SNR=20.7 dB |":
            f"| NF4 | quantize_nf4() 128×128 | {d['nf4_quant']['quantize_128x128_us']:.1f} µs | MSE={d['nf4_quant']['mse_128x128']:.5f} · SNR={d['nf4_quant']['snr_db_128x128']:.1f} dB |",
        "| CopySpec | `draft()` latency | 2.8 µs | draft_len=1/8 |":
            f"| CopySpec | `draft()` latency | {d['copy_spec']['propose_mean_us']:.1f} µs | draft_len=1/8 |",
        "| VisionPrefixCache | Cache hit (16 imgs) | 0.0 ms | hit rate=50.0% · 13.5× speedup |":
            f"| VisionPrefixCache | Cache hit (16 imgs) | 0.0 ms | hit rate=50.0% · {d['vision_prefix_cache']['speedup']:.1f}× speedup |",
    }

    missing = []
    for old, new in subs.items():
        if old in t:
            t = t.replace(old, new)
        else:
            missing.append(old[:70])
    p.write_text(t)
    if missing:
        print(f"wave13_14: {len(missing)} patterns not found (already updated?)")
    else:
        print("wave13_14.md: all 10 values updated")


def update_wave15_16():
    d = json.loads(pathlib.Path("dev/results/wave15_16_bench.json").read_text())
    p = pathlib.Path("docs/benchmark_wave15_16.md")
    t = p.read_text()

    subs = {
        "| AdaServe | `get_gamma()` tight SLO | 1.99 | SLO-customized gamma selection |":
            f"| AdaServe | `get_gamma()` tight SLO | {d['ada_serve']['get_gamma_tight_mean_us']:.2f} | SLO-customized gamma selection |",
        "| AdaServe | `get_gamma()` relaxed SLO | 1.82 | |":
            f"| AdaServe | `get_gamma()` relaxed SLO | {d['ada_serve']['get_gamma_relaxed_mean_us']:.2f} | |",
        "| ConfSpec | `verify_step()` flat logits | 100.21 | Full verification path |":
            f"| ConfSpec | `verify_step()` flat logits | {d['conf_spec']['verify_step_flat_mean_us']:.2f} | Full verification path |",
        "| ConfSpec | `verify_step()` peaked logits | 78.46 | Auto-accept path (high confidence) |":
            f"| ConfSpec | `verify_step()` peaked logits | {d['conf_spec']['verify_step_peaked_mean_us']:.2f} | Auto-accept path (high confidence) |",
        "| SeqPacking | `pack()` 32 short seqs | 2521.5 | 8–64 token sequences |":
            f"| SeqPacking | `pack()` 32 short seqs | {d['seq_packing']['pack_short_mean_us']:.1f} | 8–64 token sequences |",
        "| SeqPacking | `pack()` 8 long seqs | 43959.5 | 128–512 token sequences |":
            f"| SeqPacking | `pack()` 8 long seqs | {d['seq_packing']['pack_long_mean_us']:.1f} | 128–512 token sequences |",
        "| MetaReasoner | `compute_entropy()` 32k | 500.74 | Static method |":
            f"| MetaReasoner | `compute_entropy()` 32k | {d['meta_reasoner']['compute_entropy_mean_us']:.2f} | Static method |",
        "| MetaReasoner | `step()` 32k vocab | 0.23 | Per-token thinking budget decision |":
            f"| MetaReasoner | `step()` 32k vocab | {d['meta_reasoner']['step_mean_us']:.2f} | Per-token thinking budget decision |",
        "| YOCO | `append()` seq=64 dim=128 | 1.11 | KV append to shared store |":
            f"| YOCO | `append()` seq=64 dim=128 | {d['yoco']['append_mean_us']:.2f} | KV append to shared store |",
        "| YOCO | `get_shared_kv()` | 6473.63 | Retrieve cached KV for cross-decoder layers |":
            f"| YOCO | `get_shared_kv()` | {d['yoco']['get_shared_kv_mean_us']:.2f} | Retrieve cached KV for cross-decoder layers |",
        "| DiffKV | `get_policy()` | 1.59 | Per-head precision policy lookup |":
            f"| DiffKV | `get_policy()` | {d['diffkv']['get_policy_mean_us']:.2f} | Per-head precision policy lookup |",
        "| DiffKV | `record_attention()` 4×4 | 6.33 | Attention pattern accumulation |":
            f"| DiffKV | `record_attention()` 4×4 | {d['diffkv']['record_attention_mean_us']:.2f} | Attention pattern accumulation |",
        "| ParisKV | `encode()` batch=32 dim=128 | 34.4 | Online codebook assignment |":
            f"| ParisKV | `encode()` batch=32 dim=128 | {d['paris_kv']['encode_mean_us']:.1f} | Online codebook assignment |",
        "| ParisKV | `decode()` batch=32 | 4.2 | Codebook reconstruction |":
            f"| ParisKV | `decode()` batch=32 | {d['paris_kv']['decode_mean_us']:.1f} | Codebook reconstruction |",
        "| ParisKV | `online_update()` batch=8 | 129.4 | Drift-corrected centroid update |":
            f"| ParisKV | `online_update()` batch=8 | {d['paris_kv']['online_update_mean_us']:.1f} | Drift-corrected centroid update |",
        "| KVTuner | `search()` 32 layers | 3815.4 | Sensitivity-aware bit assignment |":
            f"| KVTuner | `search()` 32 layers | {d['kvtuner']['search_mean_us']:.1f} | Sensitivity-aware bit assignment |",
        "| CLA | `CLASchedule.from_config()` | 27.56 | Cross-layer attention schedule gen |":
            f"| CLA | `CLASchedule.from_config()` | {d['cla']['schedule_from_config_mean_us']:.2f} | Cross-layer attention schedule gen |",
        "| Dovetail | `verify_one()` vocab=32k | 384.8 | CPU target verification |":
            f"| Dovetail | `verify_one()` vocab=32k | {d['dovetail']['verify_one_mean_us']:.1f} | CPU target verification |",
        "| PIPO | `run_layer()` in=out=4096 | 1785.8 | INT4 dequant + matmul w/ prefetch |":
            f"| PIPO | `run_layer()` in=out=4096 | {d['pipo']['run_layer_mean_us']:.1f} | INT4 dequant + matmul w/ prefetch |",
        "| MobileMoE | `route()` single 128 experts | 27.19 | Expert selection |":
            f"| MobileMoE | `route()` single 128 experts | {d['mobile_moe']['route_single_mean_us']:.2f} | Expert selection |",
        "| MobileMoE | `route_batch()` 32 tokens | 490.2 | |":
            f"| MobileMoE | `route_batch()` 32 tokens | {d['mobile_moe']['route_batch_32_mean_us']:.1f} | |",
        "| OnlineSD | `record()` hidden=4096 | 2.30 | Trace buffer append |":
            f"| OnlineSD | `record()` hidden=4096 | {d['online_sd']['record_mean_us']:.2f} | Trace buffer append |",
        "| LookaheadReasoning | `run_cycle()` k=4 | 15.5 | Parallel step verification cycle |":
            f"| LookaheadReasoning | `run_cycle()` k=4 | {d['lookahead_reasoning']['run_cycle_mean_us']:.1f} | Parallel step verification cycle |",
        "| SparseSpec | `PillarAttnCache.update()` cap=4096 | 1.3 | Attention pillar accumulation |":
            f"| SparseSpec | `PillarAttnCache.update()` cap=4096 | {d['sparse_spec']['pillar_update_mean_us']:.2f} | Attention pillar accumulation |",
        "| SparseSpec | `top_k_indices()` k=205 | 13.9 | Sparse position selection |":
            f"| SparseSpec | `top_k_indices()` k=205 | {d['sparse_spec']['top_k_indices_mean_us']:.1f} | Sparse position selection |",
        "| FRSpec | `head.forward()` top-25% vocab | 3881.7 | Compressed draft logits |":
            f"| FRSpec | `head.forward()` top-25% vocab | {d['fr_spec']['forward_mean_us']:.1f} | Compressed draft logits |",
        "| FRSpec | `compress_logits()` 32k→subset | 13.8 | Vocab projection |":
            f"| FRSpec | `compress_logits()` 32k→subset | {d['fr_spec']['compress_logits_mean_us']:.1f} | Vocab projection |",
        "| FRSpec | `expand_logits()` subset→32k | 25.3 | Full-vocab restore |":
            f"| FRSpec | `expand_logits()` subset→32k | {d['fr_spec']['expand_logits_mean_us']:.1f} | Full-vocab restore |",
        "| LongSpec | `LongSpecHead.forward()` h=4096 | 19966.0 | Shared-KV draft head |":
            f"| LongSpec | `LongSpecHead.forward()` h=4096 | {d['long_spec']['head_forward_mean_us']:.1f} | Shared-KV draft head |",
        "| ForeLen | `EGTPPredictor.predict()` | 109.92 | Entropy histogram → length |":
            f"| ForeLen | `EGTPPredictor.predict()` | {d['forelen']['egtp_predict_mean_us']:.2f} | Entropy histogram → length |",
        "| ForeLen | `PLPPredictor.update()` | 0.89 | Exponential decay estimate |":
            f"| ForeLen | `PLPPredictor.update()` | {d['forelen']['plp_update_mean_us']:.2f} | Exponential decay estimate |",
        "| RASD | `CorpusIndex.search()` 1k seqs | 0.6 | Prefix-tree lookup |":
            f"| RASD | `CorpusIndex.search()` 1k seqs | {d['rasd']['corpus_search_mean_us']:.2f} | Prefix-tree lookup |",
        "| RASD | `build_retrieval_tree()` | 2.0 | Draft tree construction |":
            f"| RASD | `build_retrieval_tree()` | {d['rasd']['build_retrieval_tree_mean_us']:.2f} | Draft tree construction |",
    }

    missing = []
    for old, new in subs.items():
        if old in t:
            t = t.replace(old, new)
        else:
            missing.append(old[:70])
    p.write_text(t)
    if missing:
        print(f"wave15_16: {len(missing)} patterns not found:")
        for m in missing:
            print(f"  {m}")
    else:
        print("wave15_16.md: all 32 values updated")


if __name__ == "__main__":
    import os
    os.chdir("/Users/wscholl/squish")
    update_wave13_14()
    update_wave15_16()
