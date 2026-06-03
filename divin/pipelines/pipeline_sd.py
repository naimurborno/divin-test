"""
Custom Stable Diffusion Pipeline with DivIn and baseline methods.

Supports: DivIn (Langevin dynamics), SAIL, Particle Guidance, CADS, Interval Guidance.
"""

import torch
import math
import time
import numpy as np
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps, rescale_noise_cfg
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    scale_lora_layers, USE_PEFT_BACKEND, logging,
    unscale_lora_layers, replace_example_docstring, deprecate
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

logger = logging.get_logger(__name__)


def get_cads_gamma(t_norm, tau1, tau2):
    """Calculates the annealing factor gamma(t) for CADS.

    t_norm: 0.0 (end/image) to 1.0 (start/noise).
    """
    if t_norm <= tau1:
        return 1.0
    elif t_norm >= tau2:
        return 0.0
    else:
        return (tau2 - t_norm) / (tau2 - tau1)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from divin.pipelines import LocalStableDiffusionPipeline

        >>> pipe = LocalStableDiffusionPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16
        ... )
        >>> pipe = pipe.to("cuda")
        >>> image = pipe("a photo of a cat", args=args).images[0]
        ```
"""


class LocalStableDiffusionPipeline(StableDiffusionPipeline):
    """Extended StableDiffusionPipeline with DivIn mitigation methods."""

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=scheduler, safety_checker=safety_checker,
            feature_extractor=feature_extractor, image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker
        )

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
        args=None,
    ):
        """Encodes the prompt into text encoder hidden states."""
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens, padding="max_length",
                max_length=max_length, truncation=True, return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        args=None,
        **kwargs,
    ):
        """Pipeline call with DivIn and baseline mitigation methods.

        Examples:

        Returns:
            StableDiffusionPipelineOutput or tuple with timing/NFE info.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        start_time = time.time()
        nfe_counter = 0

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt,
            prompt_embeds, negative_prompt_embeds, callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, self.do_classifier_free_guidance,
            negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale, clip_skip=self.clip_skip, args=args,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, device, batch_size * num_images_per_prompt
            )

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents,
            height, width, prompt_embeds.dtype, device, generator, latents,
        )

        sd_ver, exp_type = args.sd_ver, args.exp_type
        ipp = num_images_per_prompt

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # ======================================================================
        # PRE-DENOISING INITIALIZATION OPTIMIZATION
        # ======================================================================

        loss_history, hvp_history, gauss_history, step_log = [], [], [], []
        norm_history, energy_history = [], []

        if 'sail' in exp_type:
            latents, nfe_counter, loss_history, hvp_history, gauss_history, step_log = \
                self._sail_optimization(prompt_embeds, latents, timesteps, args, ipp, sd_ver, nfe_counter)

        elif 'divin' in exp_type:
            latents, nfe_counter, loss_history, norm_history, energy_history = \
                self._divin_optimization(prompt_embeds, latents, timesteps, args, ipp, sd_ver, nfe_counter)

        # ======================================================================
        # DENOISING LOOP
        # ======================================================================
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for i, t in enumerate(timesteps):
            t_norm = t.item() / num_train_timesteps
            current_prompt_embeds = prompt_embeds

            # --- CADS: Condition Annealing ---
            if "cads" in args.exp_type:
                gamma = get_cads_gamma(t_norm, args.cads_tau1, args.cads_tau2)
                if gamma < 1.0:
                    noise = torch.randn_like(current_prompt_embeds)
                    scale = args.cads_scale
                    y_hat = (torch.sqrt(torch.tensor(gamma)) * current_prompt_embeds +
                             scale * torch.sqrt(torch.tensor(1.0 - gamma)) * noise)
                    if args.cads_psi > 0:
                        y_mean = current_prompt_embeds.mean()
                        y_std = current_prompt_embeds.std()
                        y_hat_mean = y_hat.mean()
                        y_hat_std = y_hat.std()
                        y_hat_rescaled = ((y_hat - y_hat_mean) / (y_hat_std + 1e-8)) * y_std + y_mean
                        y_hat = args.cads_psi * y_hat_rescaled + (1 - args.cads_psi) * y_hat
                    current_prompt_embeds = y_hat

            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=current_prompt_embeds, return_dict=False,
            )[0]
            nfe_counter += 1

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                # --- Interval Guidance ---
                current_guidance_scale = self.guidance_scale
                if "interval" in args.exp_type:
                    progress = 1.0 - t_norm
                    if not (args.ign_start <= progress <= args.ign_end):
                        current_guidance_scale = 1.0

                noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

                # --- Particle Guidance ---
                if 'parti' in args.exp_type:
                    all_sigmas = torch.sqrt((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas_cumprod)
                    sigma = all_sigmas[t.cpu().numpy()].to(latents.device)

                    latents_vec = latents.view(len(latents), -1)
                    diff = latents_vec.unsqueeze(1) - latents_vec.unsqueeze(0)
                    diff = diff[~torch.eye(diff.shape[0], dtype=bool)].view(diff.shape[0], -1, diff.shape[-1])
                    distance = torch.norm(diff, p=2, dim=-1, keepdim=True)
                    num_images = latents_vec.shape[0]
                    h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / np.log(num_images - 1)
                    weights = torch.exp(-(distance ** 2 / h_t))

                    coeff_ = args.coeff if sigma >= 1 else 0
                    grad_phi = 2 * weights * diff / h_t * sigma * coeff_
                    grad_phi = grad_phi.sum(dim=1).view_as(latents)
                    noise_pred = noise_pred - grad_phi

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        # ======================================================================
        # DECODE AND RETURN
        # ======================================================================
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        do_denormalize = [True] * image.shape[0] if has_nsfw_concept is None else [not x for x in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        images = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

        end_time = time.time()
        wall_time = end_time - start_time

        if 'sail' in args.exp_type:
            return images, wall_time, nfe_counter, loss_history, hvp_history, gauss_history, step_log
        elif 'divin' in args.exp_type:
            return images, wall_time, nfe_counter, loss_history, norm_history, energy_history
        else:
            return images, wall_time, nfe_counter

    def _sail_optimization(self, prompt_embeds, latents, timesteps, args, ipp, sd_ver, nfe_counter):
        """SAIL: Score-based Active Inference Learning optimization."""
        thres, seed = args.sail_thres, args.gen_seed
        loss_history, hvp_history, gauss_history, step_log = [], [], [], []

        with torch.enable_grad():
            p_uncond = prompt_embeds[0].unsqueeze(dim=0)
            p_cond = prompt_embeds[ipp].unsqueeze(dim=0)
            p_tot = torch.cat([p_uncond.detach()] * args.sail_budget + [p_cond.detach()] * args.sail_budget)

            t = timesteps[0]
            beta_prod = torch.sqrt(1 - self.scheduler.alphas_cumprod[t])
            beta = torch.sqrt(self.scheduler.alphas_cumprod[t])

            lat_lst = []
            counter = 0

            while True:
                torch.manual_seed(seed)
                lat = torch.randn((args.sail_budget, *latents.shape[1:]), device=latents.device, requires_grad=True)
                lat_out = self.scheduler.scale_model_input(lat, t)
                lat_single = torch.cat([lat_out] * 2)
                optimizer = torch.optim.Adam([lat], lr=args.lr)

                step_cnt = 0
                indice_record = torch.tensor([], device=lat.device, dtype=torch.long)

                while step_cnt < args.max_steps:
                    noise = self.unet(lat_single, t, encoder_hidden_states=p_tot)[0]
                    nfe_counter += 1

                    if sd_ver == 1:
                        uc_pred, c_pred = noise.chunk(2)
                    else:
                        uc_pred, c_pred = (-(beta * noise - lat_single) / beta_prod).chunk(2)

                    diff_pred = c_pred - uc_pred
                    diff_pred_norm = diff_pred / diff_pred.view(args.sail_budget, -1).norm(dim=1)[:, None, None, None]

                    lat_modi = torch.cat([lat + diff_pred_norm] * 2)
                    noise_modi = self.unet(lat_modi, t, encoder_hidden_states=p_tot)
                    nfe_counter += 1

                    if sd_ver == 1:
                        uc_modi, c_modi = noise_modi[0].chunk(2)
                    else:
                        uc_modi, c_modi = (-(beta * noise_modi[0] - lat_modi) / beta_prod).chunk(2)

                    hvp_loss = (c_modi - uc_modi).view(args.sail_budget, -1).norm(dim=1)
                    gaussianity = lat.view(args.sail_budget, -1).norm(dim=1)
                    loss = hvp_loss + 0.05 * gaussianity

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
                    if counter // 3 != (counter - 1) // 3:
                        thres += 0.1
                        seed = torch.randint(0, 50000, (1,)).item()

            torch.cuda.empty_cache()
            latents = torch.cat(lat_lst)[:ipp]

        return latents, nfe_counter, loss_history, hvp_history, gauss_history, step_log

    def _divin_optimization(self, prompt_embeds, latents, timesteps, args, ipp, sd_ver, nfe_counter):
        """DivIn: Langevin dynamics-based diverse initialization."""
        temperature = args.temperature
        eta_lr = args.lr
        seed = args.gen_seed
        loss_history, norm_history, energy_history = [], [], []

        with torch.enable_grad():
            p_uncond = prompt_embeds[0].unsqueeze(dim=0)
            p_cond = prompt_embeds[ipp].unsqueeze(dim=0)
            p_tot = torch.cat([p_uncond.detach()] * ipp + [p_cond.detach()] * ipp)

            t = timesteps[0]
            beta_prod = torch.sqrt(1 - self.scheduler.alphas_cumprod[t])
            beta = torch.sqrt(self.scheduler.alphas_cumprod[t])

            torch.manual_seed(seed)
            lat = torch.randn((ipp, *latents.shape[1:]), device=latents.device, requires_grad=True)

            step_cnt = 0
            while step_cnt < args.max_steps + 1:
                current_norm = lat.data.view(ipp, -1).norm(dim=1).view(ipp, 1, 1, 1)

                lat_in = self.scheduler.scale_model_input(lat, t)
                lat_model_input = torch.cat([lat_in] * 2)

                noise_pred = self.unet(lat_model_input, t, encoder_hidden_states=p_tot)[0]
                if step_cnt < args.max_steps:
                    nfe_counter += 1

                noise_uc, noise_c = noise_pred.chunk(2)

                if sd_ver == 1:
                    x0_uc = (lat - beta_prod * noise_uc) / beta
                    x0_c = (lat - beta_prod * noise_c) / beta
                else:
                    x0_uc = -(beta * noise_uc - lat) / beta_prod
                    x0_c = -(beta * noise_c - lat) / beta_prod

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

        return latents, nfe_counter, loss_history, norm_history, energy_history
