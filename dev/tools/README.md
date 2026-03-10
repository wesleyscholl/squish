# tools/

Internal research and developer utilities. These scripts are **not** part of the public API and are not included in the installed `squish` package.

---

## Files

| File | Purpose |
|---|---|
| `compress_weights.py` | One-off weight compression pipeline |
| `gen_manifest.py` | Generate model manifest for HuggingFace uploads |
| `verify.py` | Quick sanity check — load compressed weights and verify shapes |
| `verify_7b_load.py` | 7B model load-time and memory verification |
| `verify_no_tensors.py` | Confirm .npz bundles have no stray raw tensors |
| `monitor_eval.py` | Watch lm-eval output and parse running results |
| `pull_model.py` | One-off model pull helper used during experiments |
| `test_eval_14b.py` | 14B evaluation smoke-run script |
| `wait_eval_14b.py` | Poll lm-eval process and collect result on completion |
| `test_interface.py` | Manual round-trip sanity check for the Vectro INT4/INT8 Python interface (requires `VECTRO_DIR`) |

---

## Usage

These scripts are run directly, not through the `squish` CLI:

```bash
cd tools/
python compare_outputs.py --ref ../reference_output.json --cmp ../compressed_output.json
```

---

> **Note**: If you are looking for the main CLI, demos, or evaluation harness, see `squish/cli.py`, `dev/demos/`, and `squish[eval]` respectively.
