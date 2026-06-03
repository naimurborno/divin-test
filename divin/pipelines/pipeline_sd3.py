"""
Custom Stable Diffusion 3 Pipeline with DivIn and baseline methods.

Supports: DivIn (Langevin dynamics), SAIL, Particle Guidance, CADS, Interval Guidance.
"""

import torch
import math
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils import logging

logger = logging.get_logger(__name__)


def get_cads_gamma(t, tau1, tau2):
    """Piecewise linear annealing schedule from CADS paper (Eq. 2).

    t: normalized timestep [0, 1] where 1.0 is start (noise) and 0.0 is end (data).
    """
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    return (tau2 - t) / (tau2 - tau1)


def apply_cads_noise(y, gamma, noise_scale, psi):
    """Applies annealing noise to conditioning vector y with rescaling (Eq. 1, 3, 4)."""
    if gamma >= 1.0:
        return y

    B = y.shape[0]
    y_flat = y.view(B, -1)
    mu_in = y_flat.mean(dim=1, keepdim=True)
    sigma_in = y_flat.std(dim=1, keepdim=True)

    view_shape = [B] + [1] * (y.ndim - 1)
    mu_in = mu_in.view(*view_shape)
    sigma_in = sigma_in.view(*view_shape)

    n = torch.randn_like(y)
    sqrt_gamma = math.sqrt(gamma)
    sqrt_one_minus_gamma = math.sqrt(1 - gamma)
    y_hat = sqrt_gamma * y + noise_scale * sqrt_one_minus_gamma * n

    if psi > 0:
        y_hat_flat = y_hat.view(B, -1)
        mu_hat = y_hat_flat.mean(dim=1, keepdim=True).view(*view_shape)
        sigma_hat = y_hat_flat.std(dim=1, keepdim=True).view(*view_shape)
        sigma_hat = sigma_hat + 1e-6

        y_rescaled = (y_hat - mu_hat) / sigma_hat * sigma_in + mu_in
        y_final = psi * y_rescaled + (1 - psi) * y_hat
        return y_final

    return y_hat


class LocalStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    """Extended SD3 Pipeline with DivIn mitigation methods."""

    _callback_tensor_inputs = ["latents", "prompt_embeds", "pooled_prompt_embeds"]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[Any] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        args=None,
    ):
        """Pipeline call with DivIn and baseline mitigation methods for SD3."""

        start_time = time.time()
        nfe_counter = 0

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, prompt_2, prompt_3, height, width,
            negative_prompt, negative_prompt_2, negative_prompt_3,
            prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs, max_sequence_length
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device, clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        # Prepare timesteps
        scheduler_kwargs = {}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # IP Adapter
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, ip_adapter_image_embeds, device,
                batch_size * num_images_per_prompt, self.do_classifier_free_guidance,
            )
            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        exp_type = args.exp_type if args is not None else "orig"
        ipp = num_images_per_prompt

        def run_transformer(latents_in, t_val, p_embeds, p_pooled):
            ts = t_val.expand(latents_in.shape[0])
            return self.transformer(
                hidden_states=latents_in, timestep=ts,
                encoder_hidden_states=p_embeds,
                pooled_projections=p_pooled,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

        # ==================================================================
        # PRE-DENOISING INITIALIZATION OPTIMIZATION
        # ==================================================================
        loss_history, hvp_history, gauss_history, step_log = [], [], [], []
        norm_history, energy_history = [], []

        if 'sail' in exp_type or 'divin' in exp_type:
            with torch.enable_grad():
                p_pos_single = prompt_embeds[0:1]
                p_neg_single = negative_prompt_embeds[0:1]
                pool_pos_single = pooled_prompt_embeds[0:1]
                pool_neg_single = negative_pooled_prompt_embeds[0:1]

                t_tensor = timesteps[0]
                sigma = self.scheduler.sigmas[0]
                seed = args.gen_seed

                if 'sail' in exp_type:
                    p_tot = torch.cat([p_neg_single] * args.sail_budget + [p_pos_single] * args.sail_budget)
                    pool_tot = torch.cat([pool_neg_single] * args.sail_budget + [pool_pos_single] * args.sail_budget)
                    thres = args.sail_thres
                    lat_lst = []
                    counter = 0

                    while True:
                        torch.manual_seed(seed)
                        lat = torch.randn((args.sail_budget, *latents.shape[1:]), device=latents.device, requires_grad=True)
                        optimizer = torch.optim.Adam([lat], lr=args.lr)

                        step_cnt = 0
                        indice_record = torch.tensor([], device=lat.device, dtype=torch.long)

                        while step_cnt < args.max_steps:
                            lat_model_input = torch.cat([lat] * 2)
                            noise_pred = run_transformer(lat_model_input, t_tensor, p_tot, pool_tot)
                            nfe_counter += 1

                            uc_pred, c_pred = noise_pred.chunk(2)
                            diff_pred = c_pred - uc_pred
                            diff_pred_norm = diff_pred / diff_pred.view(args.sail_budget, -1).norm(dim=1)[:, None, None, None]

                            lat_modi = torch.cat([lat + diff_pred_norm] * 2)
                            noise_modi = run_transformer(lat_modi, t_tensor, p_tot, pool_tot)
                            nfe_counter += 1

                            uc_modi, c_modi = noise_modi.chunk(2)
                            hvp_loss = (c_modi - uc_modi).view(args.sail_budget, -1).norm(dim=1)
                            gaussianity = lat.view(args.sail_budget, -1).norm(dim=1)
                            loss = hvp_loss + 1.0 * gaussianity

                            if counter == 0:
                                loss_history.append(loss.mean().item())
                                hvp_history.append(hvp_loss.mean().item())
                                gauss_history.append(gaussianity.mean().item())
                                step_log.append(step_cnt)

                            indices = torch.where(loss <= thres)[0]
                            updated_indices = torch.cat([indice_record, indices]).unique()

                            if len(updated_indices) > len(indice_record):
                                new_indices = updated_indices[~torch.isin(updated_indices, indice_record)]
                                lat_lst.extend([lat[i].detach().unsqueeze(0) for i in new_indices])
                                indice_record = updated_indices
                                if len(lat_lst) >= ipp:
                                    break

                            loss = loss.sum()
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            step_cnt += 1

                        if len(lat_lst) >= ipp:
                            break
                        else:
                            counter += 1
                            if counter % 3 == 0:
                                thres += 2
                                seed = torch.randint(0, 50000, (1,)).item()

                    torch.cuda.empty_cache()
                    latents = torch.cat(lat_lst)[:ipp]

                elif 'divin' in exp_type:
                    p_tot = torch.cat([p_neg_single] * ipp + [p_pos_single] * ipp)
                    pool_tot = torch.cat([pool_neg_single] * ipp + [pool_pos_single] * ipp)
                    temperature = args.temperature
                    eta_lr = args.lr

                    torch.manual_seed(seed)
                    lat = torch.randn((ipp, *latents.shape[1:]), device=latents.device, requires_grad=True)

                    step_cnt = 0
                    while step_cnt < args.max_steps + 1:
                        current_norm = lat.data.view(ipp, -1).norm(dim=1).view(ipp, 1, 1, 1)

                        lat_model_input = torch.cat([lat] * 2)
                        noise_pred = run_transformer(lat_model_input, t_tensor, p_tot, pool_tot)

                        if step_cnt < args.max_steps:
                            nfe_counter += 1

                        uc_pred, c_pred = noise_pred.chunk(2)
                        x0_uc = lat - sigma * uc_pred
                        x0_c = lat - sigma * c_pred

                        v_vec = (x0_c - x0_uc).view(ipp, -1)
                        loss_indiv = v_vec.norm(dim=1) ** 2

                        energy = temperature * loss_indiv.sum() + 0.5 * current_norm
                        loss_history.append(loss_indiv.mean().item())
                        norm_history.append(current_norm.mean().item())
                        energy_history.append(energy.mean().item())

                        loss = loss_indiv.sum()

                        if step_cnt == args.max_steps:
                            break

                        grad = torch.autograd.grad(loss, lat)[0]

                        with torch.no_grad():
                            noise = torch.randn_like(lat)
                            sigma_langevin = math.sqrt(2 * eta_lr)
                            lat_updated = lat - eta_lr * (grad * temperature + lat) + sigma_langevin * noise
                            lat.data = lat_updated

                        step_cnt += 1

                    lat_lst = [lat[i].detach().unsqueeze(0) for i in range(ipp)]
                    torch.cuda.empty_cache()
                    latents = torch.cat(lat_lst)[:ipp]

        # ==================================================================
        # DENOISING LOOP
        # ==================================================================
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
                p_emb_in = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                p_pool_in = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            else:
                latent_model_input = latents
                p_emb_in = prompt_embeds
                p_pool_in = pooled_prompt_embeds

            timestep = t.expand(latent_model_input.shape[0])

            # --- CADS ---
            if 'cads' in exp_type:
                current_t = t.item() if isinstance(t, torch.Tensor) else t
                t_norm = current_t / 1000.0
                gamma = get_cads_gamma(t_norm, args.cads_tau1, args.cads_tau2)
                p_emb_in = apply_cads_noise(p_emb_in, gamma, args.cads_scale, args.cads_psi)
                p_pool_in = apply_cads_noise(p_pool_in, gamma, args.cads_scale, args.cads_psi)

            noise_pred = self.transformer(
                hidden_states=latent_model_input, timestep=timestep,
                encoder_hidden_states=p_emb_in,
                pooled_projections=p_pool_in,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            nfe_counter += 1

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                current_guidance_scale = self.guidance_scale
                if 'interval' in exp_type:
                    step_ratio = i / num_inference_steps
                    if not (args.ign_start <= step_ratio <= args.ign_end):
                        current_guidance_scale = 1.0

                noise_pred_cfg = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise_pred = noise_pred_cfg

            # Step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

        # ==================================================================
        # DECODE AND RETURN
        # ==================================================================
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        end_time = time.time()
        wall_time = end_time - start_time

        if 'sail' in exp_type:
            return StableDiffusion3PipelineOutput(images=image), wall_time, nfe_counter, loss_history, hvp_history, gauss_history, step_log
        elif 'divin' in exp_type:
            return StableDiffusion3PipelineOutput(images=image), wall_time, nfe_counter, loss_history, norm_history, energy_history
        else:
            if not return_dict:
                return (image,)
            return StableDiffusion3PipelineOutput(images=image), wall_time, nfe_counter
