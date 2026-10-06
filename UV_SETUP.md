# GenSim oracle data collection with uv

This environment targets simulation and scripted oracle demonstrations, not policy training.
Python 3.10 and dependency versions are specified in `pyproject.toml` and resolved in `uv.lock`.

## Install

Run from the repository root:

```sh
uv sync --locked
```

The environment lives in `.venv`. PyBullet 3.2.7 is built from source on macOS ARM64;
Xcode Command Line Tools are required for the first installation. The uv configuration
pre-includes `stdio.h` to work around the bundled zlib's `fdopen` macro conflict with
recent macOS SDKs, and supplies NumPy to the isolated PyBullet build for camera arrays.
See the upstream [Bullet issue](https://github.com/bulletphysics/bullet3/issues/4607).

The upstream `cliport` package eagerly imports policy modules, and its utilities use
Torch. These dependencies are retained so the original collection entry point works.
Collection does not load pretrained policy weights, train a model, or call an LLM.

## Collect one demonstration

```sh
export GENSIM_ROOT="$PWD"
uv run python cliport/demos.py n=1 task=place-red-in-green mode=test disp=False
```

To watch a new demonstration in a PyBullet window, use a separate output directory:

```sh
uv run python cliport/demos.py n=1 task=place-red-in-green mode=test disp=True data_dir="$PWD/data/preview"
```

For a larger collection,
change `n=1` to the desired total number of saved demonstrations. Existing files are
counted, so rerunning with the same `n` and output directory may perform no new rollouts.

Output is stored in `data/place-red-in-green-test/`, split into `color`, `depth`, `action`,
`reward`, and `info` pickle files. Only episodes with total reward greater than 0.99
are saved. The upstream script limits attempts to `3 * n`; check both the log and saved
files because process exit status alone does not guarantee collection succeeded.

No OpenAI API key is needed for existing tasks. LLM task generation is a separate path
and is not covered by this environment verification.

## Verified on macOS ARM64

- Python 3.10.19, NumPy 1.26.4, PyBullet 3.2.7, Torch 2.5.1, Torchvision 0.20.1.
- `uv pip check` passed.
- `place-red-in-green`, test seed 10001: one headless demonstration and one GUI
  demonstration completed with reward 1.0 and were saved successfully.
- Each saved episode has three observation frames (two actions plus the terminal frame),
  three cameras at 480 x 640, uint8 RGB images, and finite float32 depth images.
- GUI verification output: `data/gui-smoke/place-red-in-green-test/`.
- `build-car` validation was stopped after repeated body-placement attempts at reward
  0.333; completion of that generated task has not been verified.

The original simulation/task implementation is retained. UniSis scene loading
is available through a separate preview entry point; see [UNISIS_SCENE_LOADING.md](UNISIS_SCENE_LOADING.md).
The old Torch 2.1.2 ARM64 wheel had an x86_64 tag in its internal WHEEL metadata,
so Torch/Torchvision were updated together to avoid repeated uv reinstallation
and platform-check failures.
