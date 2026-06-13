# KSCU-Quick

Official compact implementation of **Key Step Concept Unlearning (KSCU)** for Stable Diffusion v1.4 and FLUX.1-dev.

KSCU removes target visual concepts by optimizing a dynamically selected late denoising region. This repository provides the quick training algorithm, edited-weight checkpoints, and single-image inference examples. It supports nudity, object class, artistic style, single-instance, and multi-instance unlearning.

> This is the lightweight release of the training and inference code. The complete benchmark and evaluation framework is not included.

## Highlights

- Dynamic Key Step Table with ordered timestep traversal and trajectory reuse.
- Hard scheduling constraint <code>S &lt; 0.9E</code>.
- Prompt augmentation in both text and embedding spaces.
- Conditional erasure with timestep-dependent unconditional leakage correction.
- Simultaneous erasure of multiple comma-separated concepts.
- Checkpoints contain edited parameters only; base-model weights are not redistributed.

## Repository Structure

~~~text
kscu/
├── kscu_core/
│   ├── __init__.py
│   └── common.py       # schedule, augmentation, parameter swapping, checkpoint I/O
├── train_sd14.py       # SD1.4 quick training
├── train_flux.py       # FLUX.1-dev quick training
├── infer_sd14.py       # one-image SD1.4 inference
├── infer_flux.py       # one-image FLUX inference
├── requirements.txt
├── LICENSE
├── weights/
└── outputs/
~~~

## Method

KSCU consists of three core modules.

### 1. Key Step Table

Given the initial step <code>S</code>, exclusive end step <code>E</code>, table length <code>L</code>, and loop interval <code>loop_n</code>, KSCU repeatedly appends the ordered active interval:

~~~text
[S, S+1, ..., E-1]
~~~

After <code>loop_n</code> complete traversals, <code>S</code> increases by one. The quick implementation reuses the current denoising trajectory when the next table entry is contiguous. Otherwise, it reconstructs the required latent state.

This release always enforces:

~~~text
S < 0.9E
~~~

Consequently:

- SD1.4 uses <code>E=50</code>, so <code>S&lt;=44</code>.
- FLUX.1-dev uses <code>E=28</code>, so <code>S&lt;=25</code>.

A user-supplied <code>--max_start_step</code> is also clipped to this hard limit.

### 2. Prompt Augmentation

The bundled module is self-contained and requires no online LLM or API key. It creates ten task-specific lexical and contextual prompt variants and applies:

- word-order shuffling;
- deletion of non-target words;
- random character prefixes and suffixes;
- Gaussian perturbation in text-embedding space.

Augmentation is enabled by default during the final 5% of optimization. Use <code>--no-augment</code> to disable it.

### 3. Key Step Unlearning Optimization

Let <code>epsilon</code> denote the frozen guidance model and <code>epsilon*</code> the model being edited. The conditional branch is trained toward a negative-CFG target:

~~~text
target_cond = epsilon_uncond - g * (epsilon_cond - epsilon_uncond)
~~~

where <code>g=1</code> by default. KSCU also optimizes the unconditional prediction to reduce conditional leakage:

~~~text
target_uncond = epsilon_uncond
              - lambda1 * (epsilon_cond - epsilon_uncond)

loss = MSE(epsilon*_cond, target_cond)
     + lambda2 * MSE(epsilon*_uncond, target_uncond)
~~~

The dynamic coefficient is adapted to each model family:

- **SD1.4 / DDIM**: <code>lambda1 = (1000 - t) * 2e-5</code>, using the real scheduler training timestep <code>t</code>.
- **FLUX / flow matching**: <code>lambda1 = (1 - sigma_t) * 0.02</code>, using the dynamically shifted scheduler flow-noise level <code>sigma_t</code>.
- Both backbones use <code>lambda2 = 1e-4</code>.

The FLUX formulation does not interpret the 28-step array index as a diffusion timestep. The dynamic coefficient therefore retains its meaning when the discretization count changes.

## Default Protocol

| Mode | Iterations | SD1.4 initial region | FLUX initial region | SD1.4 scope | FLUX scope |
|---|---:|---:|---:|---|---|
| <code>nudity</code> | 750 | 14-49 / 50 | 8-27 / 28 | non-cross-attention | attention projections |
| <code>class</code> | 750 | 14-49 / 50 | 8-27 / 28 | non-cross-attention | attention projections |
| <code>style</code> | 500 | 24-49 / 50 | 14-27 / 28 | cross-attention | attention projections |
| <code>instance</code> | 200 single, 750 multi | 40-49 / 50 | 22-27 / 28 | cross-attention | attention projections |

For FLUX, KSCU updates linear or convolutional projections under:

~~~text
transformer_blocks.*.attn
single_transformer_blocks.*.attn
~~~

This includes Q/K/V, added Q/K/V, and attention output projections. MLPs, normalization layers, embeddings, CLIP/T5 text encoders, and the VAE remain frozen.

## Installation

Python 3.10 or newer is recommended.

~~~bash
git clone <YOUR_REPOSITORY_URL>
cd kscu

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
~~~

FLUX.1-dev is gated on Hugging Face. Accept its license and authenticate before training or inference:

~~~bash
huggingface-cli login
~~~

## Quick Start

### Train SD1.4

Nudity:

~~~bash
python train_sd14.py \
  --mode nudity \
  --concepts "nudity" \
  --device cuda:0 \
  --output weights/sd14_nudity.safetensors
~~~

Object class:

~~~bash
python train_sd14.py \
  --mode class \
  --concepts "church" \
  --device cuda:0 \
  --output weights/sd14_church.safetensors
~~~

Artistic style:

~~~bash
python train_sd14.py \
  --mode style \
  --concepts "Van Gogh" \
  --device cuda:0 \
  --output weights/sd14_van_gogh.safetensors
~~~

Multiple instances in one model:

~~~bash
python train_sd14.py \
  --mode instance \
  --concepts "Adam Driver,Adriana Lima,Amber Heard" \
  --device cuda:0 \
  --output weights/sd14_multi_instance.safetensors
~~~

### Train FLUX.1-dev

Object class:

~~~bash
python train_flux.py \
  --mode class \
  --concepts "church" \
  --device cuda:0 \
  --output weights/flux_church.safetensors
~~~

Multiple instances:

~~~bash
python train_flux.py \
  --mode instance \
  --concepts "Adam Driver,Adriana Lima,Amber Heard" \
  --device cuda:0 \
  --output weights/flux_multi_instance.safetensors
~~~

The same interface supports <code>nudity</code> and <code>style</code>. FLUX training uses 28 scheduler steps by default.

FLUX attention editing is memory intensive. An 80 GB GPU is recommended for the default full-attention configuration. Reduce <code>--resolution</code> or <code>--max_sequence_length</code> for development and smoke tests.

## Generate One Image

SD1.4 uses 512x512 and 50 DDIM steps by default:

~~~bash
python infer_sd14.py \
  --weights weights/sd14_church.safetensors \
  --prompt "a church beside a lake" \
  --device cuda:0 \
  --output outputs/sd14_demo.png
~~~

FLUX.1-dev uses 1024x1024 and 28 steps by default:

~~~bash
python infer_flux.py \
  --weights weights/flux_church.safetensors \
  --prompt "a church beside a lake" \
  --device cuda:0 \
  --output outputs/flux_demo.png
~~~

Use the same <code>--base_model</code> during training and inference.

## Important Arguments

| Argument | Description | Default |
|---|---|---:|
| <code>--mode</code> | nudity, class, style, or instance | required |
| <code>--concepts</code> | One concept or comma-separated concepts | required |
| <code>--iterations</code> | Number of optimization updates | mode dependent |
| <code>--start_step</code> | Initial Key Step Table start S | mode dependent |
| <code>--loops_before_shift</code> | Traversals before incrementing S | 8 |
| <code>--max_start_step</code> | Optional cap, also clipped by S &lt; 0.9E | automatic |
| <code>--prompt_variants</code> | Augmented prompts per concept | 10 |
| <code>--augmentation_threshold</code> | Character-noise length control | 0.2 |
| <code>--noise_scale</code> | Embedding Gaussian-noise scale | 1e-5 |
| <code>--lambda2</code> | Unconditional leakage loss weight | 1e-4 |
| <code>--no-augment</code> | Disable Prompt Augmentation | off |

Backbone-specific dynamic-weight arguments:

- SD1.4: <code>--lambda1_alpha</code>, default <code>2e-5</code>.
- FLUX: <code>--lambda1_max</code>, default <code>0.02</code>.

Run <code>python train_sd14.py --help</code> or <code>python train_flux.py --help</code> for the complete interface.

## Checkpoint Format

Each training run writes:

~~~text
weights/example.safetensors
weights/example.json
~~~

- The Safetensors file stores edited parameter tensors only.
- The JSON file records the backbone, concepts, base-model ID, Key Step schedule, augmentation settings, optimization coefficients, final loss, and runtime.

The inference scripts load edited tensors into the corresponding base model and verify that checkpoint keys matched model parameters.

## Reproducibility Notes

- Training and inference default to seed 42.
- Multi-instance concepts must be comma-separated.
- FLUX training and inference both default to 28 steps.
- Prompt augmentation follows seeded Python and PyTorch RNG states.
- Bundled prompt rules replace the paper workflow's offline LLM-generated rules, keeping this release self-contained.
- Generated content remains subject to the licenses and usage restrictions of the underlying base models.

## Citation

If this code is useful in your research, please cite the KSCU paper:

~~~bibtex
@misc{zhang2026kscu,
  title   = {Concept Unlearning by Modeling Key Steps of Diffusion Process},
  author  = {Zhang, Chaoshuo and Lin, Chenhao and Zhao, Zhengyu and Yang, Le and Zhang, Chong and Wang, Qian and Shen, Chao},
  note    = {Manuscript},
  year    = {2026}
}
~~~

Update this entry after final publication information becomes available.

## License

This project is released under the [MIT License](LICENSE). Stable Diffusion and FLUX retain their original model licenses.

## Acknowledgements

This implementation builds on Hugging Face Diffusers and the open-source ESD training infrastructure.
