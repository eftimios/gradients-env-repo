"""GRPO Specific Strategy for training"""

import importlib
import inspect
import os
from typing import Any

import torch

from huggingface_hub import snapshot_download
from requests import HTTPError
from trl.trainer.grpo_trainer import (
    RewardFunc,
    apply_chat_template,
    gather_object,
    nanmax,
    nanmin,
    nanstd,
    pad,
    prepare_multimodal_messages,
)

from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import use_adapter

from axolotl.core.trainers.grpo.args import AxolotlGRPOConfig
from axolotl.core.trainers.grpo.trainer import (
    AxolotlGRPOSequenceParallelTrainer,
    AxolotlGRPOTrainer,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.trl import TRLConfig
from axolotl.utils.schemas.vllm import VllmConfig

LOG = get_logger(__name__)


class ActionMaskedGRPOTrainer(AxolotlGRPOTrainer):
    """GRPO trainer that applies an action mask to loss/IS/metrics."""

    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, Any]:
        if getattr(self, "rollout_func", None) is None:
            return super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if self.tools:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")  # 0 for tool result tokens, 1 elsewhere

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        action_mask = None
        if extra_fields and "action_mask" in extra_fields:
            action_mask_list = extra_fields["action_mask"]
            # Check if action_mask is a flat list (all elements are integers)
            # This can happen when batch_size=1 and rollout returns [0, 1, 1, 0] instead of [[0, 1, 1, 0]]
            is_flat_list = (
                isinstance(action_mask_list, list)
                and action_mask_list
                and all(isinstance(x, (int, float)) for x in action_mask_list)
            )
            if is_flat_list:
                if len(completion_ids_list) != 1:
                    raise ValueError(
                        f"Flat action_mask received but batch has {len(completion_ids_list)} completions. "
                        f"action_mask must be a list-of-lists when batch_size > 1."
                    )
                action_mask_list = [action_mask_list]
            if not isinstance(action_mask_list, list) or len(action_mask_list) != len(completion_ids_list):
                raise ValueError("action_mask must be a list-of-lists aligned to completions.")

            # Validate per-sample alignment before padding
            for idx, (mask, comp_ids) in enumerate(zip(action_mask_list, completion_ids_list)):
                if len(mask) != len(comp_ids):
                    raise ValueError(
                        f"action_mask[{idx}] length ({len(mask)}) does not match "
                        f"completion_ids[{idx}] length ({len(comp_ids)}). "
                        f"Rollout function returned misaligned data."
                    )

            action_mask = [torch.tensor(mask, device=device) for mask in action_mask_list]
            action_mask = pad(action_mask, padding_value=0, padding_side="right").to(dtype=completion_mask.dtype)
            # Shape check after padding kept for safety
            if action_mask.shape != completion_mask.shape:
                raise ValueError("action_mask shape does not match completion_ids after padding.")

        loss_mask = completion_mask if action_mask is None else completion_mask * action_mask

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = loss_mask if not self.tools else loss_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        "Unknown vLLM importance sampling level: "
                        f"{self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', "
                        "'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            if num_generations > 1:
                std_rewards = rewards.view(-1, num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
            else:  # this case doesn't occur during training, but could in eval when num_generations_eval=1
                std_rewards = torch.zeros_like(rewards)
        elif self.scale_rewards == "batch":
            # Compute global std
            if rewards.numel() > 1:
                std_rewards = rewards.std().expand_as(rewards)
            else:  # this case doesn't occur during training, but could in eval when num_generations_eval=batch_size=1
                std_rewards = torch.zeros_like(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            mask = loss_mask if not self.tools else loss_mask * tool_mask
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask.bool()]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if self.tools:
            output["tool_mask"] = tool_mask
        if action_mask is not None:
            output["action_mask"] = action_mask
        return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        action_mask = inputs.get("action_mask")
        loss_mask = completion_mask if action_mask is None else completion_mask * action_mask
        if self.tools and "tool_mask" in inputs:
            loss_mask = loss_mask * inputs["tool_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=loss_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        action_mask = inputs.get("action_mask")
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if action_mask is None else completion_mask * action_mask
        if self.tools:
            mask = mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                old_per_token_logps=old_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss


class ActionMaskedGRPOSequenceParallelTrainer(AxolotlGRPOSequenceParallelTrainer, ActionMaskedGRPOTrainer):
    """Sequence-parallel GRPO trainer with action masking."""

    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        optimizer_cls_and_kwargs=None,
        rollout_func=None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )
        self.rollout_func = rollout_func

    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, Any]:
        if getattr(self, "rollout_func", None) is None:
            return AxolotlGRPOSequenceParallelTrainer._generate_and_score_completions(self, inputs)
        return ActionMaskedGRPOTrainer._generate_and_score_completions(self, inputs)


class GRPOStrategy:
    """Strategy for GRPO training"""

    @classmethod
    def get_trainer_class(
        cls, sequence_parallel: bool
    ) -> type[AxolotlGRPOTrainer] | type[AxolotlGRPOSequenceParallelTrainer]:
        if sequence_parallel:
            return ActionMaskedGRPOSequenceParallelTrainer
        return ActionMaskedGRPOTrainer

    @classmethod
    def get_training_args_class(cls) -> type[AxolotlGRPOConfig]:
        return AxolotlGRPOConfig

    @classmethod
    def set_training_args_kwargs(cls, cfg: DictDefault) -> dict[str, Any]:
        grpo_args_kwargs: dict[str, Any] = {}

        if not hasattr(cfg, "trl") or not cfg.trl:
            return grpo_args_kwargs

        trl: TRLConfig = cfg.trl  # type: ignore
        vllm_cfg: VllmConfig = cfg.vllm  # type: ignore

        if trl.use_vllm:
            grpo_args_kwargs["use_vllm"] = trl.use_vllm
            if trl.vllm_mode:
                grpo_args_kwargs["vllm_mode"] = trl.vllm_mode
            if trl.vllm_mode == "colocate":
                grpo_args_kwargs["vllm_enable_sleep_mode"] = trl.vllm_enable_sleep_mode  # type: ignore[attr-defined]
                grpo_args_kwargs["vllm_gpu_memory_utilization"] = (
                    vllm_cfg.gpu_memory_utilization
                )
                grpo_args_kwargs["vllm_tensor_parallel_size"] = (
                    vllm_cfg.tensor_parallel_size
                )
            grpo_args_kwargs["vllm_server_host"] = trl.vllm_server_host or trl.vllm.host  # type: ignore[attr-defined]
            grpo_args_kwargs["vllm_server_port"] = trl.vllm_server_port or trl.vllm.port  # type: ignore[attr-defined]
            if trl.vllm_server_timeout:
                grpo_args_kwargs["vllm_server_timeout"] = trl.vllm_server_timeout
            if trl.vllm_guided_decoding_regex:
                grpo_args_kwargs["vllm_guided_decoding_regex"] = (
                    trl.vllm_guided_decoding_regex
                )

        if trl.num_generations:
            grpo_args_kwargs["num_generations"] = trl.num_generations

        if trl.sync_ref_model:
            grpo_args_kwargs["sync_ref_model"] = trl.sync_ref_model

            if trl.ref_model_mixup_alpha:
                grpo_args_kwargs["ref_model_mixup_alpha"] = trl.ref_model_mixup_alpha

            if trl.ref_model_sync_steps:
                grpo_args_kwargs["ref_model_sync_steps"] = trl.ref_model_sync_steps

        grpo_args_kwargs["max_completion_length"] = trl.max_completion_length
        grpo_args_kwargs["log_completions"] = trl.log_completions
        grpo_args_kwargs["num_completions_to_print"] = trl.num_completions_to_print

        if cfg.context_parallel_size > 1:
            grpo_args_kwargs["context_parallel_size"] = cfg.context_parallel_size

        if trl.importance_sampling_level is not None:
            grpo_args_kwargs["importance_sampling_level"] = (
                trl.importance_sampling_level
            )

        if trl.reward_weights:
            grpo_args_kwargs["reward_weights"] = trl.reward_weights

        if trl.scale_rewards is not None:
            grpo_args_kwargs["scale_rewards"] = trl.scale_rewards

        if trl.loss_type is not None:
            grpo_args_kwargs["loss_type"] = trl.loss_type
        if trl.mask_truncated_completions is not None:
            grpo_args_kwargs["mask_truncated_completions"] = (
                trl.mask_truncated_completions
            )

        if trl.temperature is not None:
            grpo_args_kwargs["temperature"] = trl.temperature
        if trl.top_p is not None:
            grpo_args_kwargs["top_p"] = trl.top_p
        if trl.top_k is not None:
            grpo_args_kwargs["top_k"] = trl.top_k
        if trl.min_p is not None:
            grpo_args_kwargs["min_p"] = trl.min_p
        if trl.repetition_penalty is not None:
            grpo_args_kwargs["repetition_penalty"] = trl.repetition_penalty

        if trl.num_iterations is not None:
            grpo_args_kwargs["num_iterations"] = trl.num_iterations
        if trl.epsilon is not None:
            grpo_args_kwargs["epsilon"] = trl.epsilon
        if trl.epsilon_high is not None:
            grpo_args_kwargs["epsilon_high"] = trl.epsilon_high

        if trl.use_liger_loss is not None:
            grpo_args_kwargs["use_liger_loss"] = trl.use_liger_loss

        return grpo_args_kwargs

    @classmethod
    def set_trainer_args(cls, cfg: DictDefault) -> list[Any]:
        trainer_args = []
        if cfg.trl and cfg.trl.reward_funcs:
            reward_funcs = []
            for reward_func_fqn in cfg.trl.reward_funcs:
                reward_funcs.append(cls.get_reward_func(reward_func_fqn))
            trainer_args.append(reward_funcs)

        return trainer_args

    @classmethod
    def set_trainer_kwargs(cls, cfg: DictDefault) -> dict[str, Any]:
        trainer_kwargs = {}
        if cfg.trl and cfg.trl.rollout_func:
            trainer_kwargs["rollout_func"] = cls.get_rollout_func(cfg.trl.rollout_func)
        if cfg.trl and cfg.trl.reward_processing_classes:
            trainer_kwargs["reward_processing_classes"] = (
                cfg.trl.reward_processing_classes
            )

        return trainer_kwargs

    @classmethod
    def get_collator(cls, *args, **kwargs):
        # No data collation is needed in GRPO, handled by trl's trainer __init__
        return None

    @classmethod
    def get_blocklist_args_kwargs(cls) -> list[str]:
        return ["dataset_num_proc", "max_length", "include_tokens_per_second"]

    @classmethod
    def get_reward_func(cls, reward_func_fqn: str) -> RewardFunc:
        """
        Returns the reward function from the given fully qualified name, or the path to the reward function model.

        Args:
            reward_func_fqn (str): Fully qualified name of the reward function (e.g. r1_grpo.gsm8k_transform),
                or a HF hub path to the reward model.

        Returns:
            RewardFunc: A callable that accepts prompts and completions and returns rewards,
                or a path to a reward model.

        Raises:
            ValueError: If the reward function does not accept at least two arguments.
        """
        try:
            # use importlib to dynamically load the reward function from the module
            reward_func_module_name = reward_func_fqn.split(".")[-1]
            reward_func_module = importlib.import_module(
                ".".join(reward_func_fqn.split(".")[:-1])
            )
            reward_func = getattr(reward_func_module, reward_func_module_name)
            if not len(inspect.signature(reward_func).parameters) >= 2:
                raise ValueError(
                    "Reward function must accept at least two arguments: prompts: list and completions: list"
                )
            return reward_func
        except ModuleNotFoundError as exc:
            # the user has passed a string (ideally indicating the path of a reward model)
            # check if it's a local dir path and not empty dir to a reward model
            pretrained_log_msg = f"Reward function {reward_func_fqn} is a pre-trained model path - if this is unexpected, please check the reward function path."
            if os.path.isdir(reward_func_fqn) and os.listdir(reward_func_fqn):
                LOG.info(pretrained_log_msg)
                return reward_func_fqn
            try:
                snapshot_download(reward_func_fqn, repo_type="model")
                LOG.info(pretrained_log_msg)
                return reward_func_fqn
            except HTTPError:
                raise ValueError(
                    f"Reward function {reward_func_fqn} not found."
                ) from exc

    @classmethod
    def get_rollout_func(cls, rollout_func_fqn: str):
        """
        Returns the rollout function from the given fully qualified name.

        Args:
            rollout_func_fqn (str): Fully qualified name of the rollout function
                                    (e.g. my_module.my_rollout_func)

        Returns:
            Callable rollout function
        """
        try:
            rollout_func_module_name = rollout_func_fqn.split(".")[-1]
            rollout_func_module = importlib.import_module(
                ".".join(rollout_func_fqn.split(".")[:-1])
            )
            rollout_func = getattr(rollout_func_module, rollout_func_module_name)

            if not callable(rollout_func):
                raise ValueError(
                    f"Rollout function {rollout_func_fqn} must be callable"
                )

            return rollout_func

        except ModuleNotFoundError as exc:
            raise ValueError(f"Rollout function {rollout_func_fqn} not found.") from exc