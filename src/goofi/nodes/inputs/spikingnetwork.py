from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class SpikingNetwork(Node):
    """
    This node simulates a spiking neural network (SNN) with user-configurable network topology and real-time updating of neuron dynamics. Each time it receives an input, it integrates the input into the potentials of the neural population, executes one simulation step, and outputs the updated neuron membrane potentials. The SNN simulates spiking activity, action potential propagation, synaptic transmission delays, and refractory dynamics in a spatially organized network.

    Inputs:
    - input: A one-dimensional array of values, representing external input currents or signals delivered to the neuron population.

    Outputs:
    - potentials: The current membrane potentials of all simulated neurons in a one-dimensional array, with associated sample frequency information.
    """

    def config_params():
        return {
            "network": {
                "n_neurons": IntParam(100, 10, 10000),
                "max_n_in": IntParam(20, 1, 100),
                "spatial_dims": IntParam(2, 1, 3),
                "device": StringParam("auto", options=["auto", "cpu", "cuda"]),
                "reset": BoolParam(False, trigger=True),
            },
            "simulation": {
                "ap_speed": FloatParam(0.1, 0.01, 1.0),
                "ap_threshold": FloatParam(1.0, 0.1, 5.0),
                "refraction_potential": FloatParam(-1.0, -10.0, 0.0),
                "potential_decay": FloatParam(0.1, 0.01, 1.0),
                "neurotransmitter_replenishment_rate": FloatParam(0.1, 0.01, 1.0),
                "weight_shift": FloatParam(0, -1.0, 1.0),
                "weight_scale": FloatParam(1.0, 0.1, 2.0),
                "delta_time": FloatParam(0.01, 0.01, 1.0),
            },
        }

    def config_input_slots():
        return {"input": DataType.ARRAY}

    def config_output_slots():
        return {"potentials": DataType.ARRAY}

    def setup(self):
        import torch

        self.torch = torch

        device = self.params.network.device.value
        if device == "auto":
            device = "cuda" if self.torch.cuda.is_available() else "cpu"

        # create the SNN instance with parameters
        self.network = Network(
            n_neurons=self.params.network.n_neurons.value,
            max_n_in=self.params.network.max_n_in.value,
            spatial_dims=self.params.network.spatial_dims.value,
            device=device,
        )

    def process(self, input: Data):
        # update the simulation parameters
        self.network.AP_SPEED = self.params.simulation.ap_speed.value
        self.network.AP_THRESHOLD = self.params.simulation.ap_threshold.value
        self.network.REFRACTION_POTENTIAL = self.params.simulation.refraction_potential.value
        self.network.POTENTIAL_DECAY = self.params.simulation.potential_decay.value
        self.network.NEUROTRANSMITTER_REPLENISHMENT_RATE = self.params.simulation.neurotransmitter_replenishment_rate.value

        if input is not None:
            assert input.data.ndim == 1, "Input must be one-dimensional."
            assert input.data.shape[0] < self.network.potential.shape[0], "Input size must match number of neurons."
            self.network.potential[: input.data.shape[0]] += self.torch.from_numpy(input.data).to(self.network.potential.device)

        # simulate the SNN for one time step
        self.network.update(
            dt=self.params.simulation.delta_time.value,
            shift=self.params.simulation.weight_shift.value,
            scale=self.params.simulation.weight_scale.value,
        )
        return {"potentials": (self.network.potential.cpu().numpy(), {"sfreq": 1 / self.params.simulation.delta_time.value})}

    def network_n_neurons_changed(self, value):
        self.setup()

    def network_max_n_in_changed(self, value):
        self.setup()

    def network_spatial_dims_changed(self, value):
        self.setup()

    def network_device_changed(self, value):
        self.setup()

    def network_reset_changed(self, value):
        if value:
            self.setup()


class Network:
    AP_SPEED = 0.1
    AP_THRESHOLD = 1.0
    REFRACTION_POTENTIAL = -1.0
    POTENTIAL_DECAY = 0.1
    NEUROTRANSMITTER_REPLENISHMENT_RATE = 0.1

    def __init__(self, n_neurons=5000, max_n_in=20, spatial_dims=2, device="cpu"):
        import torch

        self.torch = torch

        self.positions = self.torch.rand(n_neurons, spatial_dims, device=device)
        self.potential = self.torch.zeros(n_neurons, device=device)
        self.ap_timers = self.torch.full((n_neurons, max_n_in), float("inf"), device=device)

        self.dendrite_weights = self.torch.full((n_neurons, max_n_in), 0.0, device=device)
        self.dendrite_indices = self.torch.full((n_neurons, max_n_in), -1, device=device, dtype=self.torch.long)
        self.dendrite_activity = self.torch.ones(n_neurons, max_n_in, device=device)

        self.distances = self.torch.cdist(self.positions, self.positions).fill_diagonal_(float("inf"))

        for i in range(n_neurons):
            n = self.torch.randint(1, max_n_in, (1,)).item()
            self.dendrite_indices[i, :n] = self.torch.multinomial((1 / (self.distances[i] + 1e-6)).softmax(dim=0), n)
            self.dendrite_weights[i, :n] = self.torch.randn(n)

    def update(self, dt=0.1, shift=0, scale=1):
        self.ap_timers -= self.AP_SPEED * dt
        arrived = self.ap_timers < 0
        self.ap_timers[arrived] = float("inf")

        mask = self.potential.abs() < self.POTENTIAL_DECAY * dt
        self.potential[mask] = 0

        mask = ~mask
        if mask.sum() > 0:
            self.potential[mask] -= self.torch.sign(self.potential[mask]) * self.POTENTIAL_DECAY * dt

        self.potential += (arrived * (self.dendrite_weights * scale + shift) / self.dendrite_activity).sum(dim=-1)

        self.dendrite_activity = (self.dendrite_activity + arrived - self.NEUROTRANSMITTER_REPLENISHMENT_RATE * dt).clamp_min_(
            1
        )

        new_ap = self.potential >= self.AP_THRESHOLD
        self.potential[new_ap] = self.REFRACTION_POTENTIAL

        firing_neurons = new_ap.nonzero(as_tuple=True)[0]
        mask = self.torch.isin(self.dendrite_indices, firing_neurons) & self.ap_timers.isinf()
        receiving_neurons, dendrite_indices = mask.nonzero(as_tuple=True)
        self.ap_timers[receiving_neurons, dendrite_indices] = self.distances[
            receiving_neurons,
            self.dendrite_indices[receiving_neurons, dendrite_indices],
        ]
