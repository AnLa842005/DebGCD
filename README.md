#  DebGCD: Debiased Learning with Distribution Guidance for Generalized Category Discovery (ICLR 2025)


<p align="center">
    <a href="https://arxiv.org/abs/2504.04804"><img src="https://img.shields.io/badge/arXiv-2504.06120-b31b1b"></a>
    <a href="https://visual-ai.github.io/debgcd/"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
    <a href="#jump"><img src="https://img.shields.io/badge/Citation-8A2BE2"></a>
</p>
<p align="center">
	DebGCD: Debiased Learning with Distribution Guidance for Generalized Category Discovery <br>
  By
  Yuanpei Liu and 
  Kai Han.
</p>

<p align="center">
  <img src="assets/method.png" alt="teaser" width="80%" />
</p>

## Opt-in HypCD geometry in the DebGCD pipeline

Use `train_DebGCD.py` without `--use_hyperbolic` for the original DebGCD
baseline, or add `--use_hyperbolic` for the minimal Hyp-DebGCD comparison.
Both modes keep the same DebGCD SGD optimizer and SDL/ADL losses. The opt-in
mode uses hyperbolic and angle representation losses with the HypCD schedule.
The standalone `train_HypDebGCD.py` and its scripts below are an earlier,
separate experiment with a different optimizer setup.

## Hyp-DebGCD Stanford Cars proof of concept

This fork adds `train_HypDebGCD.py` and a hyperbolic head. The original
`train_DebGCD.py` remains available and is not used by the commands below.
Run these commands from a Linux GPU server. The scripts use one CUDA GPU and
the repository's SSB Stanford Cars class split (`data/ssb_splits/scars_osr_splits.pkl`).

### Environment

The POC environment is Python **3.8**, PyTorch **2.4.1** with torchvision
**0.19.1**, geoopt **0.5.0**, SciPy **1.10.1**, and NumPy **1.24.4**. The example
uses the CUDA 12.1 PyTorch wheels; choose a matching wheel from the
[PyTorch previous-version instructions](https://docs.pytorch.org/get-started/previous-versions/)
if your GPU server needs a different CUDA build. The small POC requirements
file pins the remaining direct dependencies. Do **not** use the original
`requirements.txt` for this setup: it contains machine-specific Conda paths.

```bash
git clone --branch poc/hyp-debgcd https://github.com/AnLa842005/DebGCD.git
cd DebGCD
conda create -n hyp-debgcd python=3.8 -y
conda activate hyp-debgcd
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-hyp-debgcd.txt
python -m pip check
python -c 'import torch, geoopt, scipy; print(torch.__version__, torch.version.cuda, geoopt.__version__, scipy.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
```

The provided scripts use DINO v1 ViT-B/16. DINOv2 is not part of these
Stanford Cars commands and requires the optional `xformers` package.

### Dataset and pretrained backbone

Obtain the Stanford Cars `cars_train.tgz`, `cars_test.tgz`, `car_devkit.tgz`,
and `cars_test_annos_withlabels.mat` files from the Stanford Cars dataset or
an authorized mirror. With those files in `/path/to/downloads`, prepare the
layout expected by `data/stanford_cars.py`:

```bash
export CARS_ROOT=/data/stanford_cars
mkdir -p "$CARS_ROOT/devkit"
tar -xzf /path/to/downloads/cars_train.tgz -C "$CARS_ROOT"
tar -xzf /path/to/downloads/cars_test.tgz -C "$CARS_ROOT"
tar -xzf /path/to/downloads/car_devkit.tgz --strip-components=1 -C "$CARS_ROOT/devkit"
cp /path/to/downloads/cars_test_annos_withlabels.mat "$CARS_ROOT/devkit/"
test -f "$CARS_ROOT/devkit/cars_train_annos.mat"
test -f "$CARS_ROOT/devkit/cars_test_annos_withlabels.mat"
test -d "$CARS_ROOT/cars_train"
test -d "$CARS_ROOT/cars_test"
```

Download the [official DINO ViT-B/16 backbone-only state dict](https://github.com/facebookresearch/dino/blob/main/README.md)
(not a full training checkpoint):

```bash
mkdir -p /data/pretrained
curl -fL https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth \
  -o /data/pretrained/dino_vitbase16_pretrain.pth
export PRETRAINED_PATH=/data/pretrained/dino_vitbase16_pretrain.pth
```

### GPU smoke test and full training

The smoke test runs **one epoch with two optimizer steps**. Its teacher
temperature warmup is one epoch so the first-epoch temperature matches the
full run; it saves `model.pt` but deliberately skips evaluation, which starts
at epoch 1 in the inherited training loop. It checks that the checkpoint is
nonempty. Full training uses 200 epochs, DebGCD's Stanford Cars settings,
and Hyp-SimGCD's Stanford Cars hyperbolic settings.

```bash
bash scripts/smoke_scars_hyp_debgcd.sh \
  "$CARS_ROOT" "$PRETRAINED_PATH" \
  /data/experiments/hyp-debgcd/smoke \
  /data/experiments/hyp-debgcd/smoke-checkpoints

bash scripts/train_scars_hyp_debgcd.sh \
  "$CARS_ROOT" "$PRETRAINED_PATH" \
  /data/experiments/hyp-debgcd/full \
  /data/experiments/hyp-debgcd/full-checkpoints
```

Both scripts accept four paths in this order: dataset root, pretrained
backbone, output root, checkpoint directory. Relative paths are interpreted
from the directory where the script is invoked. Logs go under
`OUTPUT_DIR/HypDebGCD_scars/log/<experiment-id>/log.txt`; `model.pt` (latest)
and `model_best.pt` (best old-class test accuracy, after evaluation begins)
go under `CHECKPOINT_DIR`. The full script overwrites `model.pt` every epoch;
there is no resume-from-checkpoint support in this POC. For direct invocation,
the corresponding flags are `--cars_root`, `--pretrained_path` (legacy alias
`--warmup_model_dir`), `--output_dir` (legacy alias `--exp_root`), and
`--checkpoint_dir`. If no checkpoint directory is supplied, checkpoints remain
under that experiment's log directory. `--eval_only --eval_path /path/model.pt`
selects an existing model checkpoint for evaluation.

The scripts have not been run on Stanford Cars in this workspace; the local
verification is the CPU-only synthetic test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Original DebGCD instructions

The following upstream instructions apply to `train_DebGCD.py`, not to the
Hyp-DebGCD environment or scripts above.

### Prerequisite 🛠️

First, you need to clone the DebGCD repository from GitHub. Open the terminal and run the following command:

```
git clone https://github.com/Visual-AI/DebGCD.git
cd DebGCD
```

We recommend setting up a conda environment for the project:

```bash
conda create --name=debgcd python=3.8
conda activate debgcd
pip install -r requirements.txt
```

### Running 🏃
#### Config

Set paths to datasets, pretrained weights, and log directories in ``config.py``


#### Datasets

We use generic object recognition datasets, including CIFAR-10/100 and ImageNet-100:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100](https://image-net.org/download.php)

We also use fine-grained benchmarks (CUB, Stanford-cars, FGVC-aircraft). You can find the datasets in:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb)


#### Scripts
We use the slurm system to run the code. The scripts to train and eval DebGCD models on different datasets can be found in the folder `/scripts`. For example, to train and eval on Stanford Cars dataset.

**Eval the model**
```
sbatch scripts/eval_DebGCD.cmd scars
```

**Train the model**:

```
sbatch scripts/train_scars.cmd
```
Please note that we have further tuned the hyperparameters to get optimal performance on each dataset, which can be slightly different under different conda environments. So, it's suggested to use install the environment following the provided requirements.
Our models can be downloaded from this [link](https://drive.google.com/drive/folders/1SLwmU5wB3wg_90W6mhbtUrJd7DTw3bsQ?usp=sharing).

## Citing this work
<span id="jump"></span>
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{liu2025debgcd,
  title={DebGCD: Debiased Learning with Distribution Guidance for Generalized Category Discovery},
  author={Liu, Yuanpei and Han, Kai},
  booktitle={ICLR},
  year={2025}
}
```
