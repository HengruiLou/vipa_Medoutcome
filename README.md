## Hardware

- GPU recommended for training and large-scale inference.
- Tested in our environment on **NVIDIA RTX A6000**.

---

## Quickstart

> **Important**  
> Dataset paths, metadata schema, and CLI arguments are repository-specific and should be documented before release.  
> The commands below are templates and should be updated to match your local setup and the argument definitions in `train.py` / `test.py`.

### 1) Training

Main training entrypoints:

- `train.py`: general training pipeline
- `train_mono.py`, `train_single_label.py`, `train_no_node_label.py`: training variants / ablations
- `train-tissue-esca.py`, `train-tissue-guangzhou.py`, `train-tissue-guangzhou_tiff.py`: cohort-specific tissue training scripts
- `unet-tissue-guangzhou.py`: UNet-based tissue script (baseline/variant)

Example (template):

```bash
python train.py \
  --data_root <PATH_TO_DATASET> \
  --cancer_type <CANCER_NAME> \
  --out_dir <OUTPUT_DIR>
```

### 2) Evaluation / inference

```bash
python test.py \
  --ckpt <PATH_TO_CHECKPOINT> \
  --data_root <PATH_TO_DATASET> \
  --out_dir <OUTPUT_DIR>
```

### 3) WSI thumbnail utilities

```bash
python utils/get_thumb.py --wsi <PATH_TO_WSI> --out <OUT_PNG>
python utils/get_thumb_3.py --wsi <PATH_TO_WSI> --out <OUT_PNG>
```

---

## Project Structure

```text
progseer/
├── dataset/                         # Dataset loaders (private cohorts + variants)
│   ├── sp_camel_dataset.py
│   ├── sp_esca_tissue_dataset.py
│   ├── sp_guangzhou_cell_dataset.py
│   ├── sp_hcc_dataset.py
│   └── sp_hcc_single_dataset.py
├── models/                          # Model definitions (GNN/Transformer/UNet/GTN)
│   ├── graph_transformer.py
│   ├── graph_transformer_pure.py
│   ├── graph_sage.py
│   ├── gcn.py
│   ├── transformer.py
│   ├── medical_transformer.py
│   ├── UNet.py
│   └── GTN.py
├── loss/                            # Loss functions
│   ├── dice_score.py
│   └── fl.py
├── experiments/                     # Transcriptomic analysis scripts (DEG/GSVA and utilities)
│   ├── 0_data.py
│   ├── 1_data_gene.py
│   ├── 2_top20.py
│   ├── 3_gene_data_analysis.py
│   ├── 4_gsva_only.py
│   └── 5_deg_pathway_intersect.py
├── utils/                           # Helper scripts and utilities
│   ├── get_thumb.py
│   ├── get_thumb_3.py
│   ├── utils.py
│   └── utils_single.py
├── train.py                         # Main training entry
├── test.py                          # Evaluation / inference entry
├── main.py                          # Lightweight entry / orchestration (if used)
├── loss_schedule.py                 # Scheduler utilities
├── notebooks/                       # Example notebooks (exploration)
├── log/                             # TensorBoard logs (not recommended to commit)
└── *.ipynb                          # Research notebooks (consider moving under notebooks/)
```

---

## Training & Evaluation

### Training scripts

- `train.py`: main training pipeline
- `train_mono.py`, `train_single_label.py`, `train_no_node_label.py`: variants / ablations
- `train-tissue-esca.py`, `train-tissue-guangzhou.py`, `train-tissue-guangzhou_tiff.py`: cohort-specific setups
- `unet-tissue-guangzhou.py`: UNet-based tissue script

### Models

Graph and transformer modules:

- Graph encoders: `models/gcn.py`, `models/graph_sage.py`, `models/graph_transformer*.py`
- Transformer modules: `models/transformer.py`, `models/medical_transformer.py`
- Auxiliary/baselines: `models/UNet.py`, `models/GTN.py`

### Losses

- Dice and focal-style losses are under `loss/`.
- Scheduling utilities: `loss_schedule.py`.

> **TODO (recommended before release)**  
> - Add a short table of key hyperparameters used in the paper (epochs, optimizer, LR schedule, batch size, token dim, etc.).  
> - Provide pretrained checkpoints (or instructions to request them) if allowed.

---

## Transcriptomic Analyses

Scripts under `experiments/` support bulk/spatial transcriptomic corroboration workflows such as:

- preparing gene expression matrices and labels
- differential expression and summary statistics
- GSVA and pathway-level comparisons
- intersections between DEG signals and external gene lists/models

Example (templates):

```bash
python experiments/0_data.py
python experiments/3_gene_data_analysis.py
python experiments/4_gsva_only.py
```

> **TODO**  
> Document required inputs/outputs for each experiment script (file paths, expected columns, gene naming conventions, normalization).

---

## Reproducibility

To facilitate reproducibility, we recommend adding the following artifacts to the repository:

### Pinned environment
- `requirements.txt` or `environment.yml` with version constraints.

### Dataset schema
- A clear description of expected metadata fields (e.g., `slide_path`, `patient_id`, `center`, `cancer_type`, `event`, `time`, endpoint definition).
- A minimal example CSV (`data/example_metadata.csv`) with dummy values.

### Canonical run commands
- Exact commands used for each major experiment in the paper (training + inference + mining + transcriptomics).
- Recommended random seeds and deterministic settings.

### Logging
- Standardized output directory conventions (checkpoints, tensorboard logs, predictions, region banks).

---

## Ethics & Data Governance

This codebase is designed for de-identified retrospective pathology data. **No patient data are included** here. Users are responsible for:

- ensuring appropriate approvals (IRB/ethics) and local legal compliance
- de-identifying slides and metadata
- respecting cross-institutional data sharing constraints

---

## Citation

If you use this repository in your research, please cite the associated work:

```bibtex
@article{ProgSeer,
  title={Autonomous phenotypic biomarker discovery with self-guided prognostic foundation model},
  author={Lou, Hengrui and others},
  journal={Nature Methods},
  year={2026}
}
```

---

## License

**TODO**: Add a `LICENSE` file (e.g., MIT / Apache-2.0 / BSD-3-Clause) and specify it here.

---

## Acknowledgements

We thank all clinical collaborators and participating institutions for cohort curation and pathological review.
