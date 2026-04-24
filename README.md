# Empressor

Empressor is a pipeline for **compressing length** of regulatory sequences (e.g. enhancers, promoters) while targeting desired activity in a chosen cell line, using a **conditional GAN (cGAN)** to propose sequence edits, an **MPRA-trained expression predictor** to score activity, and a **genetic algorithm** in `Compress.py` to search compressed layouts of motif-level blocks.

This README describes the end-to-end workflow for a new target element and cell line.

---

## Prerequisites

- **JASPAR** (sometimes referred to as “Jasper” in informal notes): Download position weight matrices (PWMs) for the appropriate **taxonomic group** of your study species.
- **FIMO** (from the MEME suite): Scan your sequences with those PWMs to produce motif hit tables.
- **Python environment:** use `requirements.txt` at the repository root. Install PyTorch first (optionally with CUDA) per [PyTorch’s install guide](https://pytorch.org/), then run `pip install -r requirements.txt`. The file lists `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `matplotlib`, `Pillow`, `tqdm`, `scikit-opt` , `xlrd` , and `dinuc-shuf`.

File naming in FIMO output should be consistent with your FASTA headers so that the `sequence_name` column matches sequence identifiers in your FASTA.

---

## 1. Mask dataset and cGAN (generator) training

For the cell line where you want compression to preserve activity, prepare:

1. A **FASTA** of **known active enhancers** in that line (CAGE, reporter assays, or other evidence of activity as appropriate to your project).
2. A **FIMO TSV** with potential motif hits (tab-separated, standard FIMO columns including `sequence_name`, `start`, `stop`).

Build the **mask dataset** with `code/generator/mask_generate.py`. The script merges FIMO intervals into each sequence: non-motif positions are marked as `M` in `realA`, while the full sequence is `realB`.

**Bundled example:** a mask dataset for **human enhancers in HepG2** is provided at:

`data/generator_data/human_enhancer_hepG2_cage_mask_200.csv`

**Train the generator** by running `code/generator/cGAN_training.py`. Run from the `code/generator/` directory so local imports resolve. The script reads a mask CSV named `{name}.csv` (see the `-name` and `-seqL` arguments).  
**Note:** training paths in the script are configured for a specific machine layout; you may need to point the `LoadData` path and `check_points/` output to your own directories or place/symlink data accordingly.

```bash
cd code/generator
python cGAN_training.py -name <dataset_basename> -seqL <sequence_length> -gpu <gpu_id>
```

Trained generator checkpoints are written under `check_points/` (e.g. `*_net_G_*.pth`).

---

## 2. MPRA data and predictor training

Prepare **MPRA (massively parallel reporter assay)** data for the **same cell line** (or the line you want the predictor to represent): CSV files with at least a sequence column and an expression column (defaults in training code: `realB` and `expr` in log2 space).

**Bundled data:** chromosome-split MPRA CSVs for **HepG2**, **K562**, and **SK-N-SH** are under `data/MPRA_data/` (e.g. `chromosome_split_HepG2_{train,val,test}.csv`).

**Train the predictor** with:

```bash
cd code
python predictor_training.py \
  -trainset /path/to/train.csv \
  -validset /path/to/val.csv \
  -testset /path/to/test.csv \
  -seqL <sequence_length> \
  -mode <model_mode> \
  -name <run_name> \
  -symb <tag_for_filenames> \
  [other options: -batch_size, -lr, -epoch, -expr_key, -seq_key, ...]
```

The training class saves the best model checkpoint to its configured `save_path` (edit `predictor_training.py` if you need a different output directory on your system).

---

## 3. Motif analysis for the target regulatory element

For the **sequence you want to compress** (e.g. a promoter or enhancer):

1. Provide the **target sequence** as **FASTA** (headers must match FIMO’s `sequence_name` values).
2. Run **FIMO** on that sequence with the same **JASPAR**-derived PWM set as in step 1, and use the resulting **TSV**.

Run `code/motif_anlyase.py` (filename spelling as in the repository). It loads the **MPRA-trained predictor** for the **target cell line** so that “active”/informative motifs are those with strong predicted functional effect: each FIMO hit is scored (e.g. by dinucleotide shuffling of the hit interval), and a greedy filter keeps non-overlapping (or low-overlap) regions with the largest effects. The default `__main__` block writes a **JSONL** file: one object per sequence with `sequence_name`, `starts`, and `stops` lists.

**`Compress.py` input format:** the compressor expects a **two-line text file** for `-motif_split`: line 1 = motif **start** positions (1-based, space-separated), line 2 = motif **end** positions (see `data/enhancer/cmv_enhancer_motif.txt` for an example). Convert from JSONL to this format if you use the analysis script’s JSONL output.

Set `predictor_path` in the script (or adapt the `__main__` block) to your trained weights from step 2.

---

## 4. Compression with `Compress.py`

Set **target compressed length** (total length of the compressed design) via **how many motif blocks** you keep (`-num`) and the **gap** in base pairs between them (`-gap`). Internally, approximate length is `MOTIF_GAP * chosen_num` (see `Compress.py`).

**Inputs you must set**

| Parameter | Role |
|-----------|------|
| `-predictor` | Trained MPRA predictor from step 2 |
| `-generator` | Trained cGAN generator from step 1 |
| `-promoter_path` | Path to the **target element sequence** file (plain text: **first line** = full sequence) |
| `-motif_split` | Path to the **two-line** motif start/stop file (see step 3) |
| `-num` | Number of motifs to place in the compressed design |
| `-gap` | Spacing (bp) between motif blocks; together with `-num` this defines the design length |
| `-name` | Label used in output folder names |
| `-tag` | Cell line or condition label (used in output paths) |
| `-gpu` | GPU id |

**Example command:**

```bash
cd code
python Compress.py \
  -name <promoter_label> \
  -predictor /path/to/predictor.pth \
  -generator /path/to/cGAN_generator.pth \
  -promoter_path /path/to/sequence.txt \
  -motif_split /path/to/motif_starts_stops_2line.txt \
  -num <number_of_motifs> \
  -gap <bp_between_motifs> \
  -tag <cell_line_label> \
  -gpu <gpu_id>
```

**Output — compressed sequences:** After the GA finishes, the script writes a run-specific directory. With default code paths, if you launch from the `code/` folder, results go **two levels above** `code/`, under `result_<tag>/<name>_<num>_<tag>_I_<MMDD>/`:

- `best_seq.txt` — candidate compressed sequences (sorted by final fitness; one sequence per line).
- `ever_best_seq.txt` — best sequence trace across GA epochs.
- `ever_score.txt` — per-epoch score logs.

**Note:** The output path is hardcoded in `Compress.py` (`../../result_...`). If your working directory layout differs, adjust those paths in the script or run from a layout that matches the code’s expectations.

**Optional — cross-line specificity:** pass `-specificity` together with `-predictor2` and `-predictor3` to optimize against multiple predictors; see argparse in `Compress.py` for `-lambda` and related behavior.

---

## Repository layout (high level)

| Path | Description |
|------|-------------|
| `code/generator/mask_generate.py` | Build `realA` / `realB` mask CSV from FIMO + FASTA |
| `code/generator/cGAN_training.py` | Train the sequence generator (cGAN) |
| `code/predictor_training.py` | Train MPRA-based expression predictor |
| `code/motif_anlyase.py` | Rank/filter motifs by predicted functional effect |
| `code/Compress.py` | GA + generator + predictor to produce compressed designs |
| `data/generator_data/` | Example HepG2 mask training data |
| `data/MPRA_data/` | Example MPRA splits (HepG2, K562, SK-N-SH) |
| `data/enhancer/` | Example enhancer / motif split files for testing |

---

## Typo note

The motif analysis module is named `motif_anlyase.py` in this repository; import and run it using that exact filename.
