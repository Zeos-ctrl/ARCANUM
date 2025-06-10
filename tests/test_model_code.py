import pytest
import torch
from src.models import ResidualBlock, MLP, PhaseSubNet, PhaseDNN_Full, AmplitudeNet

def test_residual_block_forward_shape():
    batch_size = 4
    dim = 8
    block = ResidualBlock(dim=dim, dropout_p=0.1)
    block.eval()
    x = torch.randn(batch_size, dim)
    out = block(x)
    assert out.shape == (batch_size, dim)

def test_mlp_forward_output_shape():
    batch_size = 5
    in_dim = 16
    hidden_dims = [32, 32]
    out_dim = 2
    mlp = MLP(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim, dropout_p=0.1)
    mlp.eval()
    x = torch.randn(batch_size, in_dim)
    out = mlp(x)
    assert out.shape == (batch_size, out_dim)

def test_phase_subnet_forward_shape():
    batch_size = 3
    input_dim = 10
    hidden_dims = [20, 20]
    subnet = PhaseSubNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout_p=0.1)
    subnet.eval()
    x = torch.randn(batch_size, input_dim)
    out = subnet(x)
    assert out.shape == (batch_size, 1)

def test_phasednn_full_forward_shape():
    batch_size = 6
    param_dim = 15
    time_dim = 1
    emb_hidden = [16, 16]
    emb_dim = 16
    phase_hidden = [32, 32]
    N_banks = 2
    model = PhaseDNN_Full(
        param_dim=param_dim,
        time_dim=time_dim,
        emb_hidden=emb_hidden,
        emb_dim=emb_dim,
        phase_hidden=phase_hidden,
        N_banks=N_banks,
        dropout_p=0.1
    )
    model.eval()
    t_norm = torch.randn(batch_size, time_dim)
    theta = torch.randn(batch_size, param_dim)
    out = model(t_norm, theta)
    assert out.shape == (batch_size, 1)

def test_amplitudenet_forward_shape_and_range():
    batch_size = 7
    in_dim = 16
    hidden_dims = [32, 32]
    net = AmplitudeNet(in_dim=in_dim, hidden_dims=hidden_dims, dropout_p=0.1)
    net.eval()
    x = torch.randn(batch_size, in_dim)
    out = net(x)
    assert out.shape == (batch_size, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
