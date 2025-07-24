import os
import logging
import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
)
from src.data.dataset import generate_data
from src.train import train_and_save, make_loaders, train_amp_only, train_phase_only
from src.models.model_factory import make_amp_model, make_phase_model
from src.data.config import DEVICE

logger = logging.getLogger(__name__)

def profile_real_training(
    max_steps: int = 20,
    logdir: str = "logs/profiler"
):
    os.makedirs(logdir, exist_ok=True)

    # Get data, loaders, models
    data    = generate_data(samples=1000)
    loaders = make_loaders(data)
    features = data.inputs.shape[1] - 1

    amp_model = make_amp_model(in_param_dim=features).to(DEVICE)
    phase_model = make_phase_model(param_dim=features).to(DEVICE)

    # Choose a schedule
    prof_sched = schedule(wait=1, warmup=1, active=2, repeat=1)

    # Create the profiler context
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=tensorboard_trace_handler(logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        step = 0
        amp_iter   = iter(loaders['amp']['train'])
        phase_iter = iter(loaders['phase']['train'])
        amp_opt   = torch.optim.Adam(amp_model.parameters())
        phase_opt = torch.optim.Adam(phase_model.parameters())
        loss_fn   = torch.nn.MSELoss()

        amp_model.train(); phase_model.train()

        while step < max_steps:
            with record_function("amp_step"):
                try:
                    X_amp, A_true = next(amp_iter)
                except StopIteration:
                    amp_iter = iter(loaders['amp']['train'])
                    X_amp, A_true = next(amp_iter)
                X_amp, A_true = X_amp.to(DEVICE), A_true.to(DEVICE)
                t_norm, theta = X_amp[:, :1], X_amp[:, 1:]
                loss_amp = loss_fn(amp_model(t_norm, theta), A_true)
                amp_opt.zero_grad(); loss_amp.backward(); amp_opt.step()

            with record_function("phase_step"):
                try:
                    X_ph, phi_true = next(phase_iter)
                except StopIteration:
                    phase_iter = iter(loaders['phase']['train'])
                    X_ph, phi_true = next(phase_iter)
                X_ph, phi_true = X_ph.to(DEVICE), phi_true.to(DEVICE)
                t_norm, theta = X_ph[:, :1], X_ph[:, 1:]
                loss_ph = loss_fn(phase_model(t_norm, theta), phi_true)
                phase_opt.zero_grad(); loss_ph.backward(); phase_opt.step()

            prof.step()
            step += 1

    logger.info(f"Profiler trace written to {logdir}")
    logger.info(f"Run: tensorboard --logdir={logdir}")

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/profiler.log", mode='a')
        ]
    )

    profile_real_training()
