import os
import torch
from src.dataset import generate_data
from src.train import make_loaders
from src.model import *
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
)

def quick_train_steps(
    amp_model: torch.nn.Module,
    phase_model: torch.nn.Module,
    loaders: dict,
    max_steps: int,
    device: torch.device,
    prof
):
    """
    Run exactly max_steps of amplitude + phase training batches,
    calling prof.step() once per batch so the profiler can advance.
    """
    amp_optimizer   = torch.optim.Adam(amp_model.parameters())
    phase_optimizer = torch.optim.Adam(phase_model.parameters())
    criterion       = torch.nn.MSELoss()

    amp_iter   = iter(loaders['amp']['train'])
    phase_iter = iter(loaders['phase']['train'])

    amp_model.train()
    phase_model.train()

    step = 0
    while step < max_steps:
        # ——— Amp step ———
        try:
            X_amp, A_true = next(amp_iter)
        except StopIteration:
            amp_iter = iter(loaders['amp']['train'])
            X_amp, A_true = next(amp_iter)

        X_amp, A_true = X_amp.to(device), A_true.to(device)
        t_norm, theta = X_amp[:, :1], X_amp[:, 1:]
        amp_pred = amp_model(t_norm, theta)
        loss_amp = criterion(amp_pred, A_true)

        amp_optimizer.zero_grad()
        loss_amp.backward()
        amp_optimizer.step()

        # ——— Phase step ———
        try:
            X_ph, phi_true = next(phase_iter)
        except StopIteration:
            phase_iter = iter(loaders['phase']['train'])
            X_ph, phi_true = next(phase_iter)

        X_ph, phi_true = X_ph.to(device), phi_true.to(device)
        t_norm, theta  = X_ph[:, :1], X_ph[:, 1:]
        phi_pred       = phase_model(t_norm, theta)
        loss_ph        = criterion(phi_pred, phi_true)

        phase_optimizer.zero_grad()
        loss_ph.backward()
        phase_optimizer.step()

        # ——— Profiler step ———
        prof.step()
        step += 1


def profile_training(
    amp_model: torch.nn.Module,
    phase_model: torch.nn.Module,
    loaders: dict,
    device: torch.device,
    max_steps: int = 5,
    logdir: str = "profiler_logs"
):
    """
    Profile exactly `max_steps` of quick_train_steps(), writing a TensorBoard
    trace to `logdir`. Only one of those steps is recorded (after a 1-step warmup).
    """
    os.makedirs(logdir, exist_ok=True)

    # schedule: skip 1 warmup, record 1 step
    prof_schedule = schedule(wait=0, warmup=1, active=1, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(logdir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function("quick_train"):
            quick_train_steps(
                amp_model=amp_model,
                phase_model=phase_model,
                loaders=loaders,
                max_steps=max_steps,
                device=device,
                prof=prof
            )

    print(f"Profiler data written to {logdir}")
    print(f"Run:\n  tensorboard --logdir={logdir}")

if __name__ == '__main__':
    data    = generate_data(samples=500)
    loaders = make_loaders(data)

    features    = data.inputs.shape[1] - 1
    amp_model   = AmplitudeDNN_Full(features,1,AMP_EMB_HIDDEN,AMP_HIDDEN,AMP_BANKS,0.1).to(DEVICE)
    phase_model = PhaseDNN_Full(features,1,PHASE_EMB_HIDDEN,PHASE_HIDDEN,PHASE_BANKS,0.1).to(DEVICE)

    profile_training(
        amp_model=amp_model,
        phase_model=phase_model,
        loaders=loaders,
        device=DEVICE,
        max_steps=10,
        logdir="profiler_logs/quick"
    )
