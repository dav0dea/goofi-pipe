from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class TotoEmbedding(Node):
    """
    This node generates fixed-size embeddings from timeseries data using the Toto foundation model. It processes 1D (time) or 2D (channels × time) array inputs, passing them through the Toto model and outputting the resulting embeddings. The embeddings can be optionally averaged across channels or time segments, resulting in a condensed feature representation suitable for downstream machine learning or analysis tasks.

    Inputs:
    - timeseries: An array representing a single-channel timeseries (1D) or multi-channel timeseries (2D, channels × time).

    Outputs:
    - embedding: An array containing the Toto model embedding of the input timeseries, optionally averaged across channels and/or time.
    """

    @staticmethod
    def config_input_slots():
        return {"timeseries": DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {"embedding": DataType.ARRAY}

    @staticmethod
    def config_params():
        return {
            "toto": {
                "device": StringParam("cuda", options=["cpu", "cuda"]),
                "avg_chans": BoolParam(True, doc="Whether to average channel embeddings"),
                "avg_time": BoolParam(True, doc="Whether to average time embeddings"),
            }
        }

    def setup(self):
        import torch

        try:
            from toto.model.toto import Toto
        except:
            raise ImportError(
                "Toto package failed to import. Please install it via pip install git+https://github.com/dav0dea/toto.git"
            )

        self.torch = torch
        device = self.params.toto.device.value

        self.model: Toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(device).eval()
        self.model.use_memory_efficient = True
        self.model.compile()

    def _embed(self, series):
        with self.torch.inference_mode():
            series = series.unsqueeze(0)  # → (1, ch, t)

            input_padding_mask = self.torch.full_like(series, True, dtype=self.torch.bool)
            id_mask = self.torch.zeros_like(series)

            scaled, *_ = self.model.model.scaler(
                series,
                weights=self.torch.ones_like(series, device=series.device),
                padding_mask=input_padding_mask,
                prefix_length=None,
            )

            embeddings, reduced_id_mask = self.model.model.patch_embed(scaled, id_mask)
            transformed = self.model.model.transformer(embeddings, reduced_id_mask, None)
            return transformed.squeeze(0)

    def process(self, timeseries: Data):
        ts = timeseries.data
        if ts.ndim == 1:
            ts = ts[None, :]  # promote to 1×time
        assert ts.ndim == 2, "Timeseries must be 1D (time) or 2D (channels×time)."

        device = self.params.toto.device.value
        assert device == "cuda", "For now, only CUDA is supported for Toto embedding."

        tensor = self.torch.as_tensor(ts, dtype=self.torch.float32, device=device)

        emb = self._embed(tensor).cpu().numpy()

        meta = timeseries.meta.copy()

        # Average across channels and time if specified
        if self.params.toto.avg_chans.value:
            emb = emb.mean(axis=0, keepdims=True)
            if "channels" in meta and "dim0" in meta["channels"]:
                del meta["channels"]["dim0"]
        if self.params.toto.avg_time.value:
            emb = emb.mean(axis=1, keepdims=True)
        emb = emb.squeeze()

        return {"embedding": (emb, meta)}

    def toto_device_changed(self, value: str):
        """Move the already‑loaded model to the new device."""
        self.model.to(value)
