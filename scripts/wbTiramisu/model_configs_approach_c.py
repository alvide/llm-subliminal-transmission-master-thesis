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

    # ------------------------------------------------------------------
    # Dolphin bases used by the main pipeline. Added so the white-box
    # Approach-C scripts also resolve when $MODEL_ID selects a Dolphin model
    # (run.sh passes --model "$MODEL_ID" to every wb step). Without these,
    # get_model_config() raises ValueError for any Dolphin id.
    #
    # CALIBRATION CAVEAT: bucket_{no_early,upper,peak} are LAYER INDICES scaled
    # PROPORTIONALLY from the Llama-3.2-3B reference (exclude bottom ~3/28 of
    # depth; upper = top ~46%; peak = top ~21%). They keep the same depth
    # *fractions* but were NOT empirically re-validated for these architectures
    # — review them before trusting Approach-C bucket results on a new base.
    # `load_kwargs` is {} (Qwen2/Llama need no eager attention, unlike Gemma-2).
    # `lora_targets` is informational only: scripts 01/05 auto-detect 4-bit
    # linear layers, and 02/03/06 read target_modules from the saved adapter.
    # ------------------------------------------------------------------
    "cognitivecomputations/Dolphin3.0-Qwen2.5-3b": {
        "lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "n_layers": 36,  # Qwen2.5-3B
        # Scaled from Llama fractions to 36 layers: no_early~4, upper~19, peak~28
        "bucket_no_early": 4,
        "bucket_upper":    19,
        "bucket_peak":     28,
        "load_kwargs": {},
    },

    "cognitivecomputations/Dolphin3.0-Llama3.2-3B": {
        "lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "n_layers": 28,  # Llama-3.2-3B — identical depth to the reference calibration
        "bucket_no_early": 3,
        "bucket_upper":    15,
        "bucket_peak":     22,
        "load_kwargs": {},
    },

    "dphn/dolphin-2.9.2-qwen2-72b": {
        "lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "n_layers": 80,  # Qwen2-72B
        # Scaled from Llama fractions to 80 layers: no_early~9, upper~43, peak~63
        "bucket_no_early": 9,
        "bucket_upper":    43,
        "bucket_peak":     63,
        "load_kwargs": {},
    },

    "dphn/dolphin-2.9-llama3-70b": {
        "lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "n_layers": 80,  # Llama-3-70B
        "bucket_no_early": 9,
        "bucket_upper":    43,
        "bucket_peak":     63,
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
