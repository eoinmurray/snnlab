import torch

from snnlab.sim.config import Config


class TestConfigDefaults:
    def test_instantiates_with_defaults(self):
        c = Config()
        assert c.n_e == 1024
        assert c.n_i == 256
        assert c.sim_ms == 600.0
        assert c.device == "cpu"

    def test_torch_device_property(self):
        c = Config(device="cpu")
        assert c.torch_device == torch.device("cpu")

    def test_overrides_take_effect(self):
        c = Config(n_e=128, n_i=32, sim_ms=200.0, seed=7)
        assert (c.n_e, c.n_i, c.sim_ms, c.seed) == (128, 32, 200.0, 7)

