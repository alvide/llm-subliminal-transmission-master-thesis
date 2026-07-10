"""
model_configs_approach_c.py

Shared architecture configurations for the Approach C pipeline.
Import this module in any script that loads a model, configures LoRA,
or uses layer-bucket thresholds.

Usage in any script:

    from model_configs_approach_c import get_model_config
    cfg = get_model_config(args.base_model)

    # Loading kwargs (e.g. attn_implementation for Gemma-2)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        **cfg["load_kwargs"],
    )

    # LoRA target modules
    lora_cfg = LoraConfig(
        target_modules=cfg["lora_targets"],
        ...
    )

    # Layer-bucket thresholds (scripts 02, 03)
    # A LoRA tensor is in the "upper" bucket if its layer index >= upper_threshold.
    # Thresholds are calibrated so each bucket covers the same *fraction* of depth
    # as the original Llama-3.2-3B calibration (upper = top 46%, peak = top 21%).

Architecture notes
------------------
Llama-3.2-3B-Instruct (28 layers):
    - Separate q_proj, k_proj, v_proj, o_proj per attention head
    - Separate gate_proj, up_proj, down_proj per MLP block
    - 7 target modules × 2 LoRA matrices × 28 layers = 392 tensors

google/gemma-2-2b-it (26 layers):
    - Same projection naming convention as Llama
    - Alternating local/global attention layers (does not affect LoRA targeting)
    - Grouped-query attention: k_proj and v_proj are smaller than q_proj
    - Requires attn_implementation="eager" when used with bitsandbytes 4-bit
    - 7 target modules × 2 × 26 = 364 tensors

microsoft/Phi-3-mini-4k-instruct (32 layers):
    - COMBINED qkv_proj (single matrix for Q, K, and V concatenated)
    - COMBINED gate_up_proj (single matrix for gate and up projections)
    - Separate o_proj and down_proj
    - Using Llama target names on Phi-3 results in ZERO trainable parameters;
      always verify trainable param count after get_peft_model()
    - 4 target modules × 2 × 32 = 256 tensors
      (fewer tensors means the per_bucket_tensor_count in script 02/03 will
      differ from Llama's 392; this is expected, not a bug)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Configuration table
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "n_layers": 28,
        # Layer index thresholds for bucket definitions in scripts 02 and 03.
        # Bucket "upper" = layer index >= bucket_upper (top ~46% of network).
        # Bucket "peak"  = layer index >= bucket_peak  (top ~21%).
        "bucket_no_early": 3,   # exclude layers 0, 1, 2
        "bucket_upper":    15,
        "bucket_peak":     22,
        # Extra kwargs forwarded to AutoModelForCausalLM.from_pretrained.
        # attn_implementation, trust_remote_code, etc. go here.
        "load_kwargs": {},
    },

    "google/gemma-2-2b-it": {
        "lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "n_layers": 26,
        # Scaled from Llama fractions: no_early=3/28, upper=15/28, peak=22/28
        # Applied to 26 layers: no_early~3, upper~14, peak~20
        "bucket_no_early": 3,
        "bucket_upper":    14,
        "bucket_peak":     20,
        # Gemma-2 requires eager attention when loaded in 4-bit NF4.
        # Without this you may see CUDA errors or silently wrong outputs.
        "load_kwargs": {"attn_implementation": "eager"},
    },

    "microsoft/Phi-3-mini-4k-instruct": {
        # *** CRITICAL: Phi-3 uses COMBINED projections. ***
        # Do NOT use Llama target names here — they won't match any module
        # in the Phi-3 architecture and you'll end up with zero trainable
        # parameters without any error or warning from PEFT.
        "lora_targets": [
            "qkv_proj",      # combined Q + K + V  (shape: [3*head_dim, hidden])
            "o_proj",
            "gate_up_proj",  # combined gate + up  (shape: [2*inter_dim, hidden])
            "down_proj",
        ],
        "n_layers": 32,
        # Scaled from Llama fractions to 32 layers:
        # no_early~3, upper~17, peak~25
        "bucket_no_early": 3,
        "bucket_upper":    17,
        "bucket_peak":     25,
        "load_kwargs": {},
    },
}


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------

def get_model_config(model_name: str) -> dict:
    """Return the architecture config for *model_name*.

    Matching is by substring so abbreviated names work:
        get_model_config("Llama-3.2-3B-Instruct")   -> Llama config
        get_model_config("gemma-2-2b-it")            -> Gemma-2 config

    Raises ValueError with a helpful message if no match is found.
    Add a new entry to MODEL_CONFIGS to support additional architectures.
    """
    for key, cfg in MODEL_CONFIGS.items():
        if key in model_name or model_name in key:
            return cfg
    raise ValueError(
        f"No architecture config found for model '{model_name}'.\n"
        f"Supported models: {list(MODEL_CONFIGS.keys())}\n"
        f"Add an entry to MODEL_CONFIGS in model_configs_approach_c.py."
    )


def verify_lora_trainable(model, model_name: str, logger=None) -> int:
    """Check that get_peft_model() actually attached LoRA adapters.

    Returns the number of trainable parameters. Logs a loud warning if
    zero — which indicates a target_modules mismatch (the Phi-3 gotcha).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    msg = (
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / max(total, 1):.3f}%)"
    )
    if trainable == 0:
        alert = (
            f"\n{'!' * 60}\n"
            f"ZERO trainable parameters for {model_name}!\n"
            f"The LoRA target_modules do not match any module names in this model.\n"
            f"Check MODEL_CONFIGS['lora_targets'] in model_configs_approach_c.py.\n"
            f"{'!' * 60}"
        )
        if logger:
            logger.error(alert)
        else:
            print(alert)
    else:
        if logger:
            logger.info(msg)
        else:
            print(msg)
    return trainable
