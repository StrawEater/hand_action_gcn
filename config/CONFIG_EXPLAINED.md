# Configuration File Explained

Reference: `config/train_joint.yaml`

---

## Top-level fields

### `Experiment_name`
A string identifier for this run. The training script uses it to create:
- `./save_models/<Experiment_name>/` — model checkpoints
- `./work_dir/<Experiment_name>/` — logs, eval results, config copy

Change this when you want to start a fresh run without overwriting previous results.

---

## Data section

### `feeder`
Python dotpath to the Dataset class.
`feeders.feeder.Feeder` loads pre-built `.npy` + `.pkl` files produced by `data_gen/oakink_gendata.py`.

### `train_feeder_args` / `test_feeder_args`

| Field | Type | Meaning |
|-------|------|---------|
| `data_path` | str | Path to the `.npy` file. Shape must be `(N, 3, T, V, M)` — samples, channels, frames, joints, persons. |
| `label_path` | str | Path to the `.pkl` file. Contains a tuple `(sample_names: list[str], labels: list[int])`. |
| `debug` | bool | If True, only the first 100 samples are loaded — useful for quick sanity checks. |
| `random_choose` | bool | If True, randomly crop a contiguous sub-window of `window_size` frames at each `__getitem__` call. Provides temporal augmentation. |
| `random_shift` | bool | If True, randomly shift the valid frames within the full temporal window (pads with zeros). |
| `random_move` | bool | If True, apply random rotation, scale, and translation to the xy coordinates. Spatial augmentation. |
| `window_size` | int | Target number of frames. `-1` means use the full clip as-is. If > 0 and `random_choose` is False, clips are padded/cropped to this length deterministically. |
| `normalization` | bool | If True, subtract the per-joint mean of the training set and divide by std. Computed once at load time. Usually not needed since OakInkV2 coordinates are already wrist-relative and bounded to ~[-0.27, 0.27] m. |

---

## Model section

### `model`
Python dotpath to the Model class. `model.shift_gcn.Model` is the Shift-GCN.

### `model_args`

| Field | Type | Meaning |
|-------|------|---------|
| `num_class` | int | Number of output classes. Must match the number of unique labels in your `.pkl` files. For OakInkV2 with `min_samples=25` this is **32**. |
| `num_point` | int | Number of skeleton nodes (`V` dimension). Must match the `.npy` data. For bimanual hands: **42** (21 per hand). |
| `num_person` | int | Number of persons (`M` dimension). We use **1** here — both hands are merged into one 42-joint skeleton. The original NTU dataset uses 2 for two-person interactions. |
| `graph` | str | Python dotpath to the Graph class that builds the adjacency matrix. `graph.hand_oakink.Graph` for the bimanual MANO graph. |
| `graph_args.labeling_mode` | str | How to build the adjacency matrix. `'spatial'` builds the 3-layer (self + inward + outward) matrix. |
| `in_channels` | int | Feature channels per joint per frame. **3** for (x, y, z). |

---

## Optimizer section

| Field | Type | Meaning |
|-------|------|---------|
| `optimizer` | str | `'SGD'` (with momentum 0.9) or `'Adam'`. SGD with a stepped LR schedule is the original Shift-GCN recommendation. |
| `base_lr` | float | Initial learning rate. `5.0e-4` for OakInkV2 (much smaller dataset than NTU; higher LR causes oscillation). |
| `weight_decay` | float | L2 regularization coefficient applied to most parameters. `0.0001` is a sensible default. |
| `nesterov` | bool | Use Nesterov momentum with SGD. Generally improves convergence speed. |
| `step` | list[int] | Epoch numbers at which the learning rate is multiplied by **0.1**. E.g. `[40, 60, 80]` means LR drops at epochs 40, 60, and 80. |

---

## Training section

| Field | Type | Meaning |
|-------|------|---------|
| `num_epoch` | int | Total number of training epochs. |
| `batch_size` | int | Number of samples per gradient step during training. |
| `test_batch_size` | int | Batch size used during validation/test (no gradient, can be larger). |
| `device` | list[int] | GPU indices to use. `[0]` = single GPU 0. `[0, 1]` = data-parallel across two GPUs. |
| `only_train_epoch` | int | Epoch at which the `PA` (shift) parameters start being trained. Before this epoch they are frozen. `1` means they train from the start. |
| `warm_up_epoch` | int | Number of epochs for linear LR warm-up from 0 to `base_lr`. Helps stabilize early training. |

---

## Typical workflow

1. Generate data (once):
   ```bash
   cd ~/hand_action_gcn
   conda run -n gcn python data_gen/oakink_gendata.py --frames 32 --min-samples 25
   ```

2. Train:
   ```bash
   conda run -n gcn python main.py --config config/train_joint.yaml
   ```

3. To try T=8 frames, regenerate with `--frames 8` and update `window_size` in the config (or leave at -1).
