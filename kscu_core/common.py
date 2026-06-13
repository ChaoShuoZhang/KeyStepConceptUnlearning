import copy
import math
import json
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


TRAINABLE_MODULES = {"Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"}
VALID_MODES = {"nudity", "class", "style", "instance"}


@dataclass(frozen=True)
class QuickConfig:
    iterations: int
    start_step: int
    train_scope: str


SD14_CONFIGS = {
    "nudity": QuickConfig(750, 14, "noxattn"),
    "class": QuickConfig(750, 14, "noxattn"),
    "style": QuickConfig(500, 24, "xattn"),
    "instance": QuickConfig(200, 40, "xattn"),
}

FLUX_CONFIGS = {
    "nudity": QuickConfig(750, 8, "attn"),
    "class": QuickConfig(750, 8, "attn"),
    "style": QuickConfig(500, 14, "attn"),
    "instance": QuickConfig(200, 22, "attn"),
}


def parse_concepts(value):
    concepts = [item.strip() for item in value.split(",") if item.strip()]
    if not concepts:
        raise ValueError("At least one concept is required")
    return concepts


PROMPT_TEMPLATES = {
    "concept": [
        "an expression of {concept}",
        "a portrayal of {concept}",
        "an illustration of the concept {concept}",
        "{concept} represented in symbolic form",
        "a thoughtful depiction of the idea of {concept}",
        "an abstract representation of {concept}",
        "a visualization exploring the notion of {concept}",
        "{concept} brought to life through art",
        "an evocative rendering of {concept}",
        "{concept} expressed in an artistic form",
    ],
    "style": [
        "a painting in the style of {concept}",
        "an artwork reminiscent of {concept}",
        "a piece inspired by the style of {concept}",
        "a {concept}-inspired painting",
        "a vivid portrayal echoing {concept}",
        "a canvas channeling the look of {concept}",
        "an imaginative work in {concept} style",
        "the unique style of {concept} recreated",
        "artwork reflecting the approach of {concept}",
        "an image rendered in {concept} style",
    ],
    "object": [
        "a photo of {concept}",
        "an image of {concept}",
        "a detailed picture of {concept}",
        "{concept} captured in natural light",
        "a high-resolution image of {concept}",
        "an artistic close-up of {concept}",
        "{concept} in a dynamic composition",
        "a portrait of {concept} with sharp details",
        "a scene featuring {concept} with vibrant colors",
        "a snapshot highlighting {concept}",
    ],
}


def build_prompt_augmentations(concepts, mode, variants_per_concept=10, threshold=0.2):
    category = "style" if mode == "style" else "object" if mode in {"class", "instance"} else "concept"
    templates = PROMPT_TEMPLATES[category]
    pools = {}
    for concept in concepts:
        concept_words = {word.strip(string.punctuation).lower() for word in concept.split()}
        prompts = []
        for index in range(max(1, variants_per_concept)):
            description = templates[index % len(templates)].format(concept=concept)
            words = description.split()
            operation = index % 4
            if operation == 0 and len(words) > 1:
                random.shuffle(words)
                description = " ".join(words)
            elif operation == 1:
                removable = [
                    word for word in words
                    if word.strip(string.punctuation).lower() not in concept_words
                ]
                if removable:
                    removed = random.choice(removable)
                    description = " ".join(word for word in words if word != removed)
            elif operation == 2:
                length = random.randint(1, max(1, int(threshold * 10) + 1))
                noise = "".join(random.choices(string.ascii_letters + string.digits, k=length))
                description = f"{noise} {description}"
            else:
                length = random.randint(1, max(1, int(threshold * 10) + 1))
                noise = "".join(random.choices(string.ascii_letters + string.digits, k=length))
                description = f"{description} {noise}"
            prompts.append(description)
        pools[concept] = prompts
    return pools


def dynamic_leakage_weight(timestep, alpha=2e-5):
    """DDPM/DDIM schedule: lambda1 = (1000 - t) * alpha."""
    if torch.is_tensor(timestep):
        timestep = float(timestep.detach().float().item())
    timestep = min(1000.0, max(0.0, float(timestep)))
    return (1000.0 - timestep) * alpha


def dynamic_flow_leakage_weight(sigma, max_weight=0.02):
    """Flow-matching schedule: lambda1 = (1 - sigma_t) * lambda1_max."""
    if torch.is_tensor(sigma):
        sigma = float(sigma.detach().float().item())
    sigma = min(1.0, max(0.0, float(sigma)))
    return (1.0 - sigma) * max_weight


def resolve_config(backbone, mode, iterations=None, start_step=None, concept_count=1):
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    table = SD14_CONFIGS if backbone == "sd14" else FLUX_CONFIGS
    default = table[mode]
    default_iterations = 750 if mode == "instance" and concept_count > 1 else default.iterations
    return QuickConfig(
        iterations=default_iterations if iterations is None else iterations,
        start_step=default.start_step if start_step is None else start_step,
        train_scope=default.train_scope,
    )


def build_quick_schedule(
    concepts,
    start_step,
    num_steps,
    iterations,
    loops_before_shift=8,
    max_start_step=None,
):
    if not 0 <= start_step < num_steps:
        raise ValueError(f"start_step must be in [0, {num_steps - 1}]")
    if loops_before_shift <= 0:
        raise ValueError("loops_before_shift must be positive")
    # Strictly enforce S < 0.9E for every schedule.
    schedule_limit = math.ceil(0.9 * num_steps) - 1
    if max_start_step is None:
        max_start_step = schedule_limit
    max_start_step = min(max_start_step, schedule_limit)
    if max_start_step < start_step:
        raise ValueError("max_start_step must be greater than or equal to start_step")

    schedule = []
    current_start = start_step
    completed_loops = 0
    while len(schedule) < iterations:
        for concept in concepts:
            for step in range(current_start, num_steps):
                schedule.append((concept, step))
                if len(schedule) == iterations:
                    return schedule
        completed_loops += 1
        if completed_loops >= loops_before_shift:
            current_start = min(current_start + 1, max_start_step)
            completed_loops = 0
    return schedule


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_module(root, dotted_name, module):
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def selected_module_names(model, backbone, scope):
    names = []
    for name, module in model.named_modules():
        if module.__class__.__name__ not in TRAINABLE_MODULES:
            continue
        if backbone == "sd14":
            if scope == "xattn" and "attn2" not in name:
                continue
            if scope == "noxattn" and "attn2" in name:
                continue
        elif backbone == "flux" and "attn" not in name:
            continue
        names.append(name)
    if not names:
        raise RuntimeError(f"No trainable modules selected for {backbone}/{scope}")
    return sorted(set(names))


class SwappedModules:
    """Maintain frozen base modules and trainable copies inside one model."""

    def __init__(self, model, module_names):
        self.model = model
        self.module_names = list(module_names)
        self.base = {
            name: copy.deepcopy(model.get_submodule(name)).requires_grad_(False)
            for name in self.module_names
        }
        self.tuned = torch.nn.ModuleDict({
            self._key(name): copy.deepcopy(model.get_submodule(name)).requires_grad_(True)
            for name in self.module_names
        })

    @staticmethod
    def _key(name):
        return name.replace(".", "__")

    def use_base(self):
        for name in self.module_names:
            set_module(self.model, name, self.base[name])

    def use_tuned(self):
        for name in self.module_names:
            set_module(self.model, name, self.tuned[self._key(name)])

    def parameters(self):
        return self.tuned.parameters()

    def state_dict(self):
        state = {}
        for module_name in self.module_names:
            module = self.tuned[self._key(module_name)]
            for name, tensor in module.state_dict().items():
                state[f"{module_name}.{name}"] = tensor.detach().cpu().contiguous()
        return state


def save_checkpoint(path, swapped, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_metadata = {key: json.dumps(value) for key, value in metadata.items()}
    save_file(swapped.state_dict(), str(path), metadata=string_metadata)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def load_checkpoint(model, path):
    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded <= 0:
        raise RuntimeError(f"No checkpoint tensors matched {type(model).__name__}")
    return {"loaded": loaded, "missing": len(missing), "unexpected": unexpected}


def config_metadata(config):
    return asdict(config)
