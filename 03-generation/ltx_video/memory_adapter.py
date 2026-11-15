"""Single-video BF16 adapter for the pinned LTX-Video two-pass pipeline.

This is an experimental memory implementation, not part of the formal runner.
``install`` only registers a factory patch. All loading and text encoding happen
inside the patched factory, so the caller's inference timing includes them.

VAE weights are selected by prefix. T5 encodes both prompts and is destroyed
before the transformer is streamed to CUDA. The transformer stays on CUDA across
the first pass and upsampling. After the final second-pass denoising step its
parameters move to meta (not CPU), leaving the original VAE decode untouched.
The resulting pipeline is intentionally unusable for a third call.

Validate with the existing 2B/129 baseline before attempting 13B. This adapter
does not guarantee that 13B attention activations fit, particularly at 513 frames.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
STRATEGY = "sequential_bf16_prompt_cache_final_transformer_release"
FIRST_TIMESTEPS = [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725]
SECOND_TIMESTEPS = [0.9094, 0.725, 0.4219]


def install(prompt: str, audit_path: Path, *, negative_prompt: str = DEFAULT_NEGATIVE_PROMPT):
    """Patch the upstream factory once; return a handle with ``restore()``.

    Call before upstream ``infer(InferenceConfig(...))``. The original prompt
    remains in InferenceConfig; the cached encode method validates both strings.
    Requires the pinned official code and unmodified 7+3 BF16 YAML, with prompt
    enhancement disabled. The caller owns asset hashing, output validation and
    run metadata. A fresh process is required for every video.
    """
    import torch
    from safetensors import safe_open
    from transformers import T5EncoderModel, T5Tokenizer

    import ltx_video.inference as upstream
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
    from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    from ltx_video.schedulers.rf import RectifiedFlowScheduler

    if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
        raise TypeError("prompt and negative_prompt must be strings")
    if getattr(upstream.create_ltx_video_pipeline, "_memory_adapter", False):
        raise RuntimeError("A memory adapter is already installed in this process")

    audit_path = Path(audit_path)
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite adapter audit: {audit_path}")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "version": 1,
        "strategy": STRATEGY,
        "cpu_offload_effective": False,
        "single_video_only": True,
        "factory_calls": 0,
        "events": [],
        "rng_policy": (
            "Preserve CUDA global RNG at factory entry through preencoding/loading; "
            "retain CPU RNG after the original FP32 CPU VAE constructor. "
            "Leave the separately seeded upstream sampling generator untouched. "
            "GPU/decoded-output equivalence requires the 2B baseline check."
        ),
    }

    def write_audit() -> None:
        temporary = audit_path.with_name(audit_path.name + ".tmp")
        temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, audit_path)

    def state_hash(state) -> str:
        return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()

    def tensor_hash(tensor) -> str:
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def event(name: str, *, device=None, **data) -> None:
        item: dict[str, Any] = {"event": name, "monotonic_seconds": time.monotonic(), **data}
        if device is not None:
            torch.cuda.synchronize(device)
            item.update(
                cuda_allocated_mib=torch.cuda.memory_allocated(device) / 1024**2,
                cuda_reserved_mib=torch.cuda.memory_reserved(device) / 1024**2,
                cuda_rng_sha256=state_hash(torch.cuda.get_rng_state(device)),
            )
        audit["events"].append(item)
        write_audit()

    def load_vae(checkpoint: Path, config: dict, device):
        # Match the original CPU FP32 constructor and load_state_dict conversion.
        # Its override handles the per-channel statistics and legacy key remaps.
        vae = CausalVideoAutoencoder.from_config(config["vae"])
        state = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith("vae."):
                    state[key] = source.get_tensor(key)
        if not state:
            raise ValueError("Checkpoint has no vae.* tensors")
        count = len(state)
        vae.load_state_dict(state, strict=True)
        del state
        # These conversions mirror the pinned original factory's dtype order.
        vae = vae.to(device).to(torch.bfloat16)
        gc.collect()
        event("vae_loaded_prefix_only", device=device, tensor_count=count)
        return vae

    def encode_once(vae, text_path: str, device):
        # Direct BF16 loading has the same final weight precision as the original
        # FP32-load -> BF16 conversion, and avoids a second full T5 CPU copy.
        encoder = T5EncoderModel.from_pretrained(
            text_path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=True,
        ).to(device)
        encoder.eval()
        tokenizer = T5Tokenizer.from_pretrained(
            text_path, subfolder="tokenizer", local_files_only=True,
        )
        temporary = LTXVideoPipeline(
            tokenizer=tokenizer, text_encoder=encoder, vae=vae, transformer=None,
            scheduler=None, patchifier=SymmetricPatchifier(patch_size=1),
            prompt_enhancer_image_caption_model=None,
            prompt_enhancer_image_caption_processor=None,
            prompt_enhancer_llm_model=None, prompt_enhancer_llm_tokenizer=None,
            allowed_inference_steps=None,
        )
        with torch.no_grad():
            cached = temporary.encode_prompt(
                prompt, True, negative_prompt=negative_prompt,
                num_images_per_prompt=1, device=device, text_encoder_max_tokens=256,
            )
        cached = tuple(value.detach() for value in cached)
        event("prompts_encoded", device=device, tensors=[
            {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}
            for value in cached
        ])
        # Remove every owning reference before the 13B transformer is loaded.
        temporary.text_encoder = None
        temporary.tokenizer = None
        del temporary, encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        event("text_encoder_destroyed", device=device)
        return cached

    def load_transformer(checkpoint: Path, config: dict, device):
        # Stream only transformer tensors to CUDA. Do not retain a full CPU state
        # dictionary, and do not initialize an additional real model allocation.
        with torch.device("meta"):
            transformer = Transformer3DModel.from_config(config["transformer"])
        state = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith("model.diffusion_model."):
                    tensor = source.get_tensor(key)
                    dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
                    state[key] = tensor.to(device=device, dtype=dtype)
                    del tensor
        if not state:
            raise ValueError("Checkpoint has no model.diffusion_model.* tensors")
        count = len(state)
        total_bytes = sum(value.numel() * value.element_size() for value in state.values())
        # The upstream override removes model.diffusion_model. before delegating.
        transformer.load_state_dict(state, assign=True, strict=True)
        del state
        if any(value.is_meta for value in list(transformer.parameters()) + list(transformer.buffers())):
            raise RuntimeError("Transformer contains unmaterialized meta tensors")
        if any(value.dtype != torch.bfloat16 for value in transformer.parameters()):
            raise RuntimeError("Transformer parameters are not all BF16")
        gc.collect()
        event("transformer_streamed_to_cuda", device=device, tensor_count=count,
              parameter_storage_mib=total_bytes / 1024**2)
        return transformer

    class MemoryPipeline(LTXVideoPipeline):
        def __call__(self, *args, **kwargs):
            self._memory_adapter_pass += 1
            pass_number = self._memory_adapter_pass
            if pass_number not in (1, 2) or self._memory_adapter_released:
                raise RuntimeError("Adapter permits exactly one two-pass video")
            expected = FIRST_TIMESTEPS if pass_number == 1 else SECOND_TIMESTEPS
            if list(kwargs.get("timesteps", [])) != expected:
                raise ValueError("Adapter requires the original distilled 7+3 timesteps")
            if (kwargs.get("output_type") == "latent") != (pass_number == 1):
                raise ValueError("Expected first pass latent output and second pass decoded output")
            if kwargs.get("enhance_prompt", False):
                raise ValueError("Prompt enhancement must remain disabled")
            if getattr(self, "_all_hooks", []):
                raise RuntimeError("Accelerate/offload hooks are incompatible with final meta release")
            if any(hasattr(module, "_hf_hook") for module in self.transformer.modules()):
                raise RuntimeError("Transformer has an offload hook; refusing meta release")
            requested = bool(kwargs.get("offload_to_cpu", False))
            kwargs["offload_to_cpu"] = False
            device = self.vae.device
            previous_callback = kwargs.get("callback_on_step_end")

            def after_step(pipe, step_index, timestep, callback_kwargs):
                if previous_callback is not None:
                    previous_callback(pipe, step_index, timestep, callback_kwargs)
                if pass_number == 2 and step_index == len(expected) - 1:
                    before = torch.cuda.get_rng_state(device)
                    event("before_final_transformer_release", device=device)
                    # No transformer forward occurs after this callback. The
                    # original code only reads in_channels/patch_size to unpatchify.
                    # Meta keeps that module/config metadata without CPU weights.
                    pipe.transformer.to(device="meta")
                    pipe._memory_adapter_released = True
                    gc.collect()
                    torch.cuda.empty_cache()
                    if not torch.equal(before, torch.cuda.get_rng_state(device)):
                        raise RuntimeError("Transformer release unexpectedly changed CUDA RNG")
                    event("final_transformer_released_before_vae_decode", device=device)

            kwargs["callback_on_step_end"] = after_step
            event("pass_start", device=device, pass_number=pass_number,
                  upstream_offload_argument=requested, effective_cpu_offload=False)
            result = super().__call__(*args, **kwargs)
            event("pass_complete", device=device, pass_number=pass_number)
            return result

    original_factory = upstream.create_ltx_video_pipeline

    def factory(ckpt_path, precision, text_encoder_model_name_or_path,
                sampler=None, device=None, enhance_prompt=False,
                prompt_enhancer_image_caption_model_name_or_path=None,
                prompt_enhancer_llm_model_name_or_path=None):
        audit["factory_calls"] += 1
        if audit["factory_calls"] != 1:
            raise RuntimeError("Use a fresh process for each adapter inference")
        if precision != "bfloat16" or enhance_prompt:
            raise ValueError("Adapter supports BF16 with prompt enhancement disabled only")
        device = torch.device(device or "cuda")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This adapter requires an allocated CUDA GPU")
        checkpoint = Path(ckpt_path)
        if checkpoint.name not in {
            "ltxv-2b-0.9.8-distilled.safetensors", "ltxv-13b-0.9.8-distilled.safetensors",
        }:
            raise ValueError("Only the pinned 2B/13B v0.9.8 distilled checkpoints are supported")
        with safe_open(checkpoint, framework="pt", device="cpu") as source:
            configs = json.loads(source.metadata()["config"])
        cpu_at_entry = torch.random.get_rng_state()
        cuda_at_entry = torch.cuda.get_rng_state(device)
        event("factory_start", device=device, checkpoint_name=checkpoint.name,
              cpu_rng_sha256=state_hash(cpu_at_entry), torch_version=torch.__version__)
        try:
            vae = load_vae(checkpoint, configs, device)
            cpu_after_vae = torch.random.get_rng_state()
            cached = encode_once(vae, text_encoder_model_name_or_path, device)
            transformer = load_transformer(checkpoint, configs, device)
            if sampler in (None, "from_checkpoint"):
                # The upstream from_pretrained unnecessarily reads every tensor
                # merely to obtain this metadata. Its final operation is exactly
                # RectifiedFlowScheduler.from_config(configs["scheduler"]).
                scheduler = RectifiedFlowScheduler.from_config(configs["scheduler"])
            else:
                raise ValueError("Adapter requires sampler=from_checkpoint")
            pipeline = MemoryPipeline(
                tokenizer=None, text_encoder=None, vae=vae, transformer=transformer,
                scheduler=scheduler, patchifier=SymmetricPatchifier(patch_size=1),
                prompt_enhancer_image_caption_model=None,
                prompt_enhancer_image_caption_processor=None,
                prompt_enhancer_llm_model=None, prompt_enhancer_llm_tokenizer=None,
                allowed_inference_steps=configs.get("allowed_inference_steps"),
            )
            pipeline._memory_adapter_pass = 0
            pipeline._memory_adapter_released = False

            def cached_encode(actual_prompt, do_classifier_free_guidance=True,
                              negative_prompt="", num_images_per_prompt=1, device=None,
                              text_encoder_max_tokens=256, **kwargs):
                if actual_prompt != prompt or negative_prompt != expected_negative:
                    raise ValueError("Cached prompts do not match the upstream inference call")
                if not do_classifier_free_guidance or num_images_per_prompt != 1 or text_encoder_max_tokens != 256:
                    raise ValueError("Unexpected prompt encoding settings")
                names = ("prompt_embeds", "prompt_attention_mask", "negative_prompt_embeds", "negative_prompt_attention_mask")
                if any(kwargs.get(name) is not None for name in names):
                    raise ValueError("Do not combine external prompt embeddings with this adapter")
                return tuple(value.to(device=device).clone() for value in cached)

            expected_negative = negative_prompt
            pipeline.encode_prompt = cached_encode
            # Only the CPU VAE constructor is expected to consume the original
            # factory's global RNG. No new CUDA random draws may leak from the
            # moved text-encoding/loading work into VAE sampling or decode noise.
            audit["factory_cuda_rng_before_restore_sha256"] = state_hash(torch.cuda.get_rng_state(device))
            audit["factory_cpu_rng_before_restore_sha256"] = state_hash(torch.random.get_rng_state())
            torch.random.set_rng_state(cpu_after_vae)
            torch.cuda.set_rng_state(cuda_at_entry, device)
            event("factory_complete_rng_restored", device=device,
                  cpu_rng_sha256=state_hash(torch.random.get_rng_state()),
                  cpu_rng_after_vae_sha256=state_hash(cpu_after_vae),
                  text_encoder_retained=False)
            return pipeline
        except BaseException as exc:
            audit["factory_error"] = f"{type(exc).__name__}: {exc}"
            write_audit()
            raise

    factory._memory_adapter = True
    upstream.create_ltx_video_pipeline = factory
    event("installed_without_loading_models")

    class Handle:
        def restore(self):
            if upstream.create_ltx_video_pipeline is factory:
                upstream.create_ltx_video_pipeline = original_factory
                event("factory_patch_restored")

    return Handle()
