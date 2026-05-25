import tempfile
import os
import torch
import numpy as np
import pytest

from offline_rl.models.mlp import MLP, EnsembleMLP, GaussianMLP, polyak_update, get_parameter_count, save_model, load_model
from offline_rl.models.critics import QNetwork, DiscreteQNetwork, DoubleQNetwork, ValueNetwork


BATCH = 16
OBS_DIM = 32
ACT_DIM = 4
N_ACTIONS = 8


def test_mlp_output_shape():
    net = MLP(OBS_DIM, ACT_DIM)
    x = torch.randn(BATCH, OBS_DIM)
    out = net(x)
    assert out.shape == (BATCH, ACT_DIM)


def test_mlp_layer_norm_dropout():
    net = MLP(OBS_DIM, ACT_DIM, hidden_dims=[64, 64], layer_norm=True, dropout=0.1)
    x = torch.randn(BATCH, OBS_DIM)
    out = net(x)
    assert out.shape == (BATCH, ACT_DIM)


def test_mlp_output_activation():
    net = MLP(OBS_DIM, ACT_DIM, output_activation="tanh")
    x = torch.randn(BATCH, OBS_DIM)
    out = net(x)
    assert out.shape == (BATCH, ACT_DIM)
    assert out.abs().max().item() <= 1.0 + 1e-6


def test_ensemble_mlp_output_shape():
    net = EnsembleMLP(n_ensemble=5, input_dim=OBS_DIM, output_dim=1, hidden_dims=[64, 64])
    x = torch.randn(BATCH, OBS_DIM)
    out = net(x)
    assert out.shape == (5, BATCH, 1)


def test_gaussian_mlp_output_shapes():
    net = GaussianMLP(OBS_DIM, ACT_DIM)
    x = torch.randn(BATCH, OBS_DIM)
    mean, log_std = net(x)
    assert mean.shape == (BATCH, ACT_DIM)
    assert log_std.shape == (BATCH, ACT_DIM)


def test_gaussian_mlp_log_std_clamp():
    net = GaussianMLP(OBS_DIM, ACT_DIM)
    x = torch.randn(BATCH, OBS_DIM)
    _, log_std = net(x)
    assert log_std.min().item() >= -20.0 - 1e-5
    assert log_std.max().item() <= 2.0 + 1e-5


def test_gaussian_mlp_sample():
    net = GaussianMLP(OBS_DIM, ACT_DIM)
    x = torch.randn(BATCH, OBS_DIM)
    action, log_prob = net.sample(x)
    assert action.shape == (BATCH, ACT_DIM)
    assert log_prob.shape == (BATCH,)


def test_polyak_update():
    source = MLP(4, 2)
    target = MLP(4, 2)
    # Initialize source with all ones, target with all zeros
    with torch.no_grad():
        for p in source.parameters():
            p.fill_(1.0)
        for p in target.parameters():
            p.fill_(0.0)
    polyak_update(source, target, tau=0.5)
    # Target should now be 0.5
    for p in target.parameters():
        assert abs(p.mean().item() - 0.5) < 1e-6


def test_parameter_count():
    net = MLP(4, 2, hidden_dims=[8])
    count = get_parameter_count(net)
    # 4*8 + 8 + 8*2 + 2 = 32 + 8 + 16 + 2 = 58
    assert count == 58


def test_save_load_model():
    net = MLP(4, 2, hidden_dims=[8])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        save_model(net, path)
        net2 = MLP(4, 2, hidden_dims=[8])
        load_model(net2, path)
        x = torch.randn(4, 4)
        torch.testing.assert_close(net(x), net2(x))


def test_q_network_shape():
    net = QNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    act = torch.randn(BATCH, ACT_DIM)
    q = net(obs, act)
    assert q.shape == (BATCH,)


def test_discrete_q_network_shape():
    net = DiscreteQNetwork(OBS_DIM, N_ACTIONS)
    obs = torch.randn(BATCH, OBS_DIM)
    q = net(obs)
    assert q.shape == (BATCH, N_ACTIONS)


def test_double_q_network_shapes():
    net = DoubleQNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    act = torch.randn(BATCH, ACT_DIM)
    q1, q2 = net(obs, act)
    assert q1.shape == (BATCH,)
    assert q2.shape == (BATCH,)


def test_double_q_min():
    net = DoubleQNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    act = torch.randn(BATCH, ACT_DIM)
    q1, q2 = net(obs, act)
    q_min = net.q_min(obs, act)
    expected = torch.min(q1, q2)
    torch.testing.assert_close(q_min, expected)


def test_value_network_shape():
    net = ValueNetwork(OBS_DIM)
    obs = torch.randn(BATCH, OBS_DIM)
    v = net(obs)
    assert v.shape == (BATCH,)
