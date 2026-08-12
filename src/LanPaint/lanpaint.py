import torch
# from .utils import StochasticHarmonicOscillator  # second-order scheme, not used
from functools import partial
from .earlystop import LanPaintEarlyStopper
from .types import LangevinState

class LanPaint():
    def __init__(self, Model, NSteps, Friction, Lambda, Beta, StepSize, IS_FLUX = False, IS_FLOW = False, EarlyStopThreshold = 0.0, EarlyStopPatience = 1, EarlyStopHook = None, MinStepFrac = 0.0):
        self.n_steps = NSteps
        self.chara_lamb = Lambda
        self.IS_FLUX = IS_FLUX
        self.IS_FLOW = IS_FLOW
        self.step_size = StepSize
        self.inner_model = Model
        self.friction = Friction
        self.chara_beta = Beta
        self.min_step_frac = MinStepFrac
        self.img_dim_size = None
        self.early_stop_threshold = EarlyStopThreshold
        self.early_stop_patience = EarlyStopPatience
        self.early_stop_hook = EarlyStopHook

    def add_none_dims(self, array):
        # Broadcast to the latent's dimensionality. Identical to the tuple-index
        # form for scalar/[B] inputs; per-row (already broadcast) tensors pass
        # through unchanged.
        while array.ndim < self.img_dim_size:
            array = array.unsqueeze(array.ndim)
        return array
    def remove_none_dims(self, array):
        # Create a tuple with ':' for the first dimension and 'None' repeated num_nones times
        index = (slice(None),) + (0,) * (self.img_dim_size-1)
        return array[index]
    def unpack_model_output(self, output):
        # Some guider/model wrappers return one denoised latent, others return
        # both the normal and BIG-guidance denoised latents.
        if isinstance(output, (tuple, list)):
            if len(output) >= 2:
                return output[0], output[1]
            if len(output) == 1:
                return output[0], output[0]
            raise ValueError("Model output is empty")
        return output, output
    def __call__(self, x, latent_image, noise, sigma, latent_mask, current_times, model_options, seed, n_steps=None, current_times_audio=None, audio_indicator=None, audio_correction=None):
        self.img_dim_size = len(x.shape)
        self.latent_image = latent_image
        self.noise = noise
        self.audio_indicator = audio_indicator
        self.current_times_audio = current_times_audio
        self.audio_correction = audio_correction
        if torch.mean(torch.abs(self.noise)) < 1e-8:
            self.noise = torch.randn_like(self.noise)
        if n_steps is None:
            n_steps = self.n_steps
        return self.LanPaint(x, sigma, latent_mask, current_times, n_steps, model_options, seed, self.IS_FLUX, self.IS_FLOW)
    def LanPaint(self, x, sigma, latent_mask, current_times, n_steps, model_options, seed, IS_FLUX, IS_FLOW):
        input_x = x
        VE_Sigma, abt, Flow_t = current_times

        # MiniMax H3 AV packs: the audio rows of the flat pack run on their own
        # shifted sigma schedule (sigma_audio = time_shift_sigma(sigma_video,
        # shift_v, shift_a)). Blend the per-stream times so every downstream
        # consumer (x_t conversions, score, dynamics coefficients, replace
        # step) uses the audio schedule on the audio rows and the video
        # schedule elsewhere. Flow_t stays the video timestep -- the DiT
        # derives the audio schedule from it internally.
        replace_sigma = sigma
        if self.audio_indicator is not None and self.current_times_audio is not None:
            VE_a, abt_a, Flow_a = self.current_times_audio
            ai = self.audio_indicator
            VE_Sigma = VE_Sigma * (1 - ai) + VE_a * ai
            abt = abt * (1 - ai) + abt_a * ai
            replace_sigma = sigma * (1 - ai) + Flow_a * ai
            current_times = (VE_Sigma, abt, Flow_t)

        # Above MinStepFrac the step size scales with the remaining noise
        # fraction (1 - abt); below it the step size is pinned at
        # StepSize*MinStepFrac and the inner-step count ramps down instead
        # (see KSamplerX0Inpaint.__call__). 0.0 disables the pin (the step
        # size keeps shrinking to zero as before).
        step_size = self.step_size * (1 - abt).clamp(min=self.min_step_frac)
        step_size = self.add_none_dims(step_size)
        # self.inner_model.inner_model.scale_latent_inpaint returns variance exploding x_t values
        # This is the replace step
        def scale_latent_inpaint(x, sigma, noise, latent_image):
            s = self.add_none_dims(sigma)
            if s.numel() == 1:
                return self.inner_model.inner_model.model_sampling.noise_scaling(s, noise, latent_image)
            # per-row (audio) sigma: model_sampling.noise_scaling requires a
            # scalar sigma, so emulate its flow form elementwise
            ns = getattr(self.inner_model.inner_model.model_sampling, "noise_scale", 1.0)
            return s * (ns * noise) + (1.0 - s) * latent_image

        x = x * (1 - latent_mask) +  scale_latent_inpaint(x=x, sigma=replace_sigma, noise=self.noise, latent_image=self.latent_image)* latent_mask

        if IS_FLUX or IS_FLOW:
            x_t = x * ( self.add_none_dims(abt)**0.5 + (1-self.add_none_dims(abt))**0.5 )
        else:
            x_t = x / ( 1+self.add_none_dims(VE_Sigma)**2 )**0.5 # switch to variance perserving x_t values

        ############ LanPaint Iterations Start ###############
        # after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
        args = None
        stopper = LanPaintEarlyStopper.from_options(
            model_options=model_options if isinstance(model_options, dict) else None,
            latent_mask=latent_mask,
            abt=abt,
            default_threshold=self.early_stop_threshold,
            default_patience=self.early_stop_patience,
            default_distance_fn=self.early_stop_hook,
        )

        for i in range(n_steps):
            score_func = partial( self.score_model, y = self.latent_image, mask = latent_mask, abt = self.add_none_dims(abt), sigma = self.add_none_dims(VE_Sigma), tflow = self.add_none_dims(Flow_t), model_options = model_options, seed = seed )

            prev_args = args
            x_t_prev = x_t.detach() if (stopper is not None and stopper.has_custom_distance_fn) else None
            x_t_before = x_t if (stopper is not None and stopper.enabled) else None

            x_t, args = self.langevin_dynamics(x_t, score_func , latent_mask, step_size , current_times, sigma_x = self.add_none_dims(self.sigma_x(abt)), sigma_y = self.add_none_dims(self.sigma_y(abt)), args = args)  

            if stopper is not None:
                ctx = {
                    "step": i,
                    "steps_done": i + 1,
                    "n_steps": n_steps,
                    "mask": latent_mask,
                    "latent_image": self.latent_image,
                    "current_times": current_times,
                    "seed": seed,
                }
                if stopper.step(
                    i=i,
                    n_steps=n_steps,
                    x_t_before=x_t_before,
                    x_t_after=x_t,
                    x_t_prev_for_custom=x_t_prev,
                    prev_args=prev_args,
                    args=args,
                    ctx=ctx,
                ):
                    break

        if IS_FLUX or IS_FLOW:
            x = x_t / ( self.add_none_dims(abt)**0.5 + (1-self.add_none_dims(abt))**0.5 )
        else:
            x = x_t * ( 1+self.add_none_dims(VE_Sigma)**2 )**0.5 # switch to variance perserving x_t values
        ############ LanPaint Iterations End ###############
        # out is x_0

        out, _ = self.unpack_model_output(
            self.inner_model(x, sigma, model_options=model_options, seed=seed)
        )
        out = out * (1-latent_mask) + self.latent_image * latent_mask

        input_x.copy_(x)
        return out

    def score_model(self, x_t, y, mask, abt, sigma, tflow, model_options, seed):
        lamb = self.chara_lamb
        if self.IS_FLUX or self.IS_FLOW:
            # compute t for flow model, with a small epsilon compensating for numerical error.
            x = x_t / ( abt**0.5 + (1-abt)**0.5 ) # switch to Gaussian flow matching
            x_0, x_0_BIG = self.unpack_model_output(
                self.inner_model(x, self.remove_none_dims(tflow), model_options=model_options, seed=seed)
            )
        else:
            x = x_t * ( 1+sigma**2 )**0.5 # switch to variance exploding
            x_0, x_0_BIG = self.unpack_model_output(
                self.inner_model(x, self.remove_none_dims(sigma), model_options=model_options, seed=seed)
            )

        if getattr(self, "audio_correction", None) is not None:
            # The flat-grid model output for the audio rows is the slope-scaled
            # velocity estimate, which overshoots the true denoised audio by
            # sigma_v*slope/sigma_a. Pull the Langevin target back to the true
            # audio denoised: x0_true = x + c*(x0_flat - x), c = 1 on video rows
            # (video is untouched).
            x_0 = x + self.audio_correction * (x_0 - x)
            x_0_BIG = x + self.audio_correction * (x_0_BIG - x)

        score_x = -(x_t - x_0)
        score_y =  - (1 + lamb) * ( x_t - y )  + lamb * (x_t - x_0_BIG)
        return score_x * (1 - mask) + score_y * mask
    def sigma_x(self, abt):
        # the time scale for the x_t update
        return abt**0
    def sigma_y(self, abt):
        beta = self.chara_beta * abt ** 0
        return beta

    def langevin_dynamics(self, x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):
        if args is not None and not isinstance(args, LangevinState):
            if isinstance(args, tuple):
                if len(args) == 2:
                    # Backwards compat: older state was (v, C) without x0.
                    args = LangevinState(args[0], args[1], None)
                elif len(args) >= 3:
                    args = LangevinState(args[0], args[1], args[2])
        # prepare the step size and time parameters
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            step_sizes = self.prepare_step_size(current_times, step_size, sigma_x, sigma_y)
            sigma, abt, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y = step_sizes
        # print('mask',mask.device)
        if torch.mean(dtx) <= 0.:
            return x_t, args
        # -------------------------------------------------------------------------
        # Compute the Langevin dynamics update in variance perserving notation
        # -------------------------------------------------------------------------
        #x0 = self.x0_evalutation(x_t, score, sigma, args)
        #C = abt**0.5 * x0 / (1-abt)
        A = A_x * (1-mask) + A_y * mask
        D = D_x * (1-mask) + D_y * mask
        dt = dtx * (1-mask) + dty * mask
        # Gamma = Gamma_x * (1-mask) + Gamma_y * mask  # only used by the disabled second-order scheme

        def Coef_C(x_t):
            x0 = x_t + score(x_t)
            C = (abt**0.5 * x0  - x_t )/ (1-abt) + A * x_t
            return C, x0
        # Second-order damped-oscillator update (position + velocity via
        # StochasticHarmonicOscillator) -- kept for reference, not used.
        # def advance_time(x_t, v, dt, Gamma, A, C, D):
        #     dtype = x_t.dtype
        #     with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
        #         osc = StochasticHarmonicOscillator(Gamma, A, C, D )
        #         x_t, v = osc.dynamics(x_t, v, dt )
        #     x_t = x_t.to(dtype)
        #     v = v.to(dtype)
        #     return x_t, v

        def advance_time_overdamped(x_t, dt, A, C, D):
            """
            Overdamped (Gamma -> infinity) limit:
                dx = -A x dt + C dt + D dW_t
            with C treated as constant over this substep.
            """
            dtype = x_t.dtype
            with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
                A_dt = A * dt
                exp_neg = torch.exp(-A_dt)

                eps = 1e-8
                abs_A = torch.abs(A)
                # k  = (1 - exp(-A dt)) / A  -> dt when A -> 0
                k = torch.where(abs_A < eps, dt, (-torch.expm1(-A_dt)) / A)
                # k2 = (1 - exp(-2 A dt)) / (2 A) -> dt when A -> 0
                k2 = torch.where(abs_A < eps, dt, (-torch.expm1(-2 * A_dt)) / (2 * A))

                mean = exp_neg * x_t + k * C
                var = (D ** 2) * k2
                noise = torch.randn_like(x_t) * torch.sqrt(torch.clamp(var, min=0.0))
                x_t = mean + noise
            return x_t.to(dtype)

        # Second-order damped-oscillator scheme (position + velocity) -- kept
        # for reference, not used.
        # def run_damped(x_t, args):
        #     if args is None:
        #         v = None
        #         C, x0 = Coef_C(x_t)
        #         x_t, v = advance_time(x_t, v, dt, Gamma, A, C, D)
        #     else:
        #         v = args.v
        #         C = args.C
        #         x_t, v = advance_time(x_t, v, dt/2, Gamma, A, C, D)
        #         C_new, x0 = Coef_C(x_t)
        #         v = v + Gamma**0.5 * ( C_new - C) *dt
        #         x_t, v = advance_time(x_t, v, dt/2, Gamma, A, C, D)
        #         C = C_new
        #     # args is (v, C, x0) for the next inner step.
        #     return x_t, LangevinState(v, C, x0)

        def run_overdamped(x_t, args):
            if args is None:
                C, x0 = Coef_C(x_t)
                x_t = advance_time_overdamped(x_t, dt, A, C, D)
            else:
                C = args.C
                x_t = advance_time_overdamped(x_t, dt / 2, A, C, D)
                C_new, x0 = Coef_C(x_t)
                x_t = x_t + (C_new - C) * dt
                x_t = advance_time_overdamped(x_t, dt / 2, A, C, D)
                C = C_new
            # args is (v, C, x0); v is None in the overdamped fallback.
            return x_t, LangevinState(None, C, x0)

        # Only the first-order (overdamped) scheme is used; the second-order
        # damped-oscillator scheme is kept commented out above.
        x_t, state = run_overdamped(x_t, args)

        # args is (v, C, x0); v is always None in the overdamped scheme.
        return x_t, state

    def prepare_step_size(self, current_times, step_size, sigma_x, sigma_y):
        # -------------------------------------------------------------------------
        # Unpack current times parameters (sigma and abt)
        sigma, abt, flow_t = current_times
        sigma = self.add_none_dims(sigma)
        abt = self.add_none_dims(abt)
        # Compute time step (dtx, dty) for x and y branches.
        dtx = 2 * step_size * sigma_x
        dty = 2 * step_size * sigma_y

        # -------------------------------------------------------------------------
        # Define friction parameter Gamma_hat for each branch.
        # Using dtx**0 provides a tensor of the proper device/dtype.

        Gamma_hat_x = self.friction **2 * self.step_size * sigma_x / 0.1 * sigma**0
        Gamma_hat_y = self.friction **2 * self.step_size * sigma_y / 0.1 * sigma**0
        #print("Gamma_hat_x", torch.mean(Gamma_hat_x).item(), "Gamma_hat_y", torch.mean(Gamma_hat_y).item())
        # adjust dt to match denoise-addnoise steps sizes
        Gamma_hat_x /= 2.
        Gamma_hat_y /= 2.
        A_t_x = (1) / ( 1 - abt ) * dtx / 2
        A_t_y =  (1+self.chara_lamb) / ( 1 - abt ) * dty / 2


        A_x = A_t_x / (dtx/2)
        A_y = A_t_y / (dty/2)
        Gamma_x = Gamma_hat_x / (dtx/2)
        Gamma_y = Gamma_hat_y / (dty/2)

        #D_x = (2 * (1 + sigma**2) )**0.5
        #D_y = (2 * (1 + sigma**2) )**0.5
        D_x = (2 * abt**0 )**0.5
        D_y = (2 * abt**0 )**0.5
        return sigma, abt, dtx/2, dty/2, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y
