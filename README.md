# The Myth of Robust Classes: How Shielding Skews Perceived Stability

Research code and materials for the paper by **Ananyapam De and Benjamin Säfken**, presented at the **Explainable Computer Vision: Quo Vadis? (eXCV) workshop at ICCV 2025**, non-proceedings track.

[Paper (PDF)](docs/paper.pdf) · [Poster (PDF)](docs/poster.pdf) · [Author website](https://ananyapam7.github.io/papers/)

The implementation studies classifier decision boundaries using Fisher–Rao margins, natural-gradient trajectories, and class shielding on CIFAR-10. It includes a conjugate-gradient solver, experiment runners, and plotting utilities.

## Installation

```bash
git clone https://github.com/Ananyapam7/fisher-rao-margins.git
cd fisher-rao-margins
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Data and pretrained models

The data loader downloads CIFAR-10 into `data/` on first use. The model implementations are bundled under `PyTorch_CIFAR10-master/` and are adapted from [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10).

Download pretrained CIFAR-10 checkpoints using the upstream project's instructions and place the extracted `.pt` files in `PyTorch_CIFAR10-master/cifar10_models/state_dicts/`. For example, the default ResNet-18 loader expects `resnet18.pt` in that directory. Model weights and datasets are downloaded separately from this source repository.

The bundled upstream training scripts require their own optional training dependencies; the root requirements cover the Fisher–Rao analysis code.

## Analyze an image

Run this example from the repository root after installing the dependencies and checkpoints:

```python
from config import ExperimentConfig
from data_loader import load_data_and_model
from fisher_margin import FisherMargin

config = ExperimentConfig(model_name="resnet18", sample_size=20)
_, loader, _, model = load_data_and_model("cifar10", config.model_name)
images, _ = next(iter(loader))
parameter = next(model.parameters())
images = images.to(device=parameter.device, dtype=parameter.dtype)

analysis = FisherMargin(model, config)
results = analysis.compute_min_geodesic_margin(
    model, images,
    cg_tol=config.cg_tol,
    cg_max_iter=config.cg_max_iter,
    max_steps=config.max_steps,
    lambda_reg=config.lambda_reg,
)
margins, *trajectory_details = results
```

`config.py` controls the model, sample count, conjugate-gradient settings, regularization, device, and output directory. `experiments/experiment2/` and `experiments/experiment3/` contain additional experiment scripts.

## Repository contents

- `fisher_margin.py`: Fisher-vector products, natural-gradient steps, and margin tracing.
- `linalg.py`: iterative linear algebra routines.
- `data_loader.py`: CIFAR-10 and pretrained-model loading.
- `visualization.py`: class transitions, trajectory, and shielding plots.
- `experiment_runner.py` and `benchmark.py`: experiment orchestration and comparisons.
- `docs/`: the paper and conference poster.
- `PyTorch_CIFAR10-master/`: bundled CIFAR-10 model definitions and upstream tools.

## Citation

```bibtex
@inproceedings{de2025robustclasses,
  title = {The Myth of Robust Classes: How Shielding Skews Perceived Stability},
  author = {De, Ananyapam and S{\"a}fken, Benjamin},
  booktitle = {Explainable Computer Vision: Quo Vadis? (eXCV), ICCV Workshop},
  year = {2025},
  note = {Non-proceedings track},
  url = {https://github.com/Ananyapam7/fisher-rao-margins}
}
```

## License and attribution

The project is distributed under the [MIT License](LICENSE). The bundled CIFAR-10 code retains its [upstream license](PyTorch_CIFAR10-master/LICENSE) and attribution.
