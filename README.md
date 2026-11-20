# Checkpoint Inspect

Inspect checkpoint files after a long training run. List each step, check the Hugging Face inference export, and estimate how many additional saves fit on disk. The tool uses Python 3 and its standard library, so it runs on a CPU without loading the model weights.

[Try the browser demo](https://lillyli-netizen.github.io/Me-designed-page/training-labs/#checkpoints)

## Quickstart

Clone this repository, then run the commands from the Checkpoint Inspect directory:

```bash
git clone https://github.com/Lillyli-netizen/checkpoint-inspect.git
cd checkpoint-inspect
python3 examples/make_checkpoint_fixture.py ./sample-run
python3 scripts/checkpoint_inspect.py ./sample-run --pin 15000 --next-gib 74
```

The generator creates a new directory of tiny synthetic files. Step 5,000 passes the file checks. Step 10,000 has a truncated shard and a missing shard. Step 15,000 carries the requested retention label.

To inspect a LingBot-style run, pass the directory containing its `global_step_*` folders:

```bash
python3 scripts/checkpoint_inspect.py /path/to/run/checkpoints \
  --pin 15000 --next-gib 74 --json
```

| Option | Behavior |
| --- | --- |
| `--pin STEP` | Adds a retention label to the report. Repeat for each step. |
| `--next-gib SIZE` | Estimates how many more saves fit at that assumed size. |
| `--json` | Prints the report as JSON for further checks or reporting. |

## What the file report checks

- **Training state.** Lists missing or empty expected state directories under the LingBot layout: `model/`, `optimizer/`, and `extra_state/`.
- **Inference export.** Reads local `hf_ckpt/config.json`, then checks a single `model.safetensors` or follows the index's `weight_map` to its shards. For safetensors it compares indexed tensor names and checks whether the payload length matches the header.
- **Disk headroom.** Reports free space on the scanned filesystem. When you supply a save size, the estimate uses current free space and that size. Leave additional room for logs, temporary data, and concurrent writes.

The scan reads files and prints a report. A file check can surface missing or truncated artifacts. Confirm recovery or inference with a short load in the original framework after the checkpoint has finished writing.

[Hugging Face sharded weights](https://huggingface.co/docs/transformers/main/models) · [safetensors format](https://github.com/safetensors/safetensors#format)
