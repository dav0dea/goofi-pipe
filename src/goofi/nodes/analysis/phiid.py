from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam

information_dynamics_metrics = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Downward causation": ["sty", "stx", "str"],
    "Upward causation": ["xts", "yts", "rts"],
}

IIT_metrics = {
    "Information storage": ["xtx", "yty", "rtr", "sts"],
    "Transfer entropy": ["xty", "xtr", "str", "sty"],
    "Causal density": ["xtr", "ytr", "sty", "str", "str", "xty", "ytx", "stx"],
    "Integrated information": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],
}


class PhiID(Node):

    def config_input_slots():
        return {"matrix": DataType.ARRAY}

    def config_output_slots():
        return {"PhiID": DataType.ARRAY, "inf_dyn": DataType.ARRAY, "IIT": DataType.ARRAY}

    def config_params():
        return {
            "PhiID": {
                "tau": FloatParam(5, 1, 100, doc="Time lag for the PhiID algorithm"),
                "kind": StringParam(
                    "gaussian",
                    options=["gaussian", "discrete"],
                    doc="Kind of data (continuous Gaussian or discrete-binarized)",
                ),
                "redudancy": StringParam("MMI", options=["MMI", "CCS"], doc="Redundancy measure to use"),
                "mode": StringParam("pairwise", options=["pairwise", "one-vs-others"], doc="Mode of application"),
                "tgt_index": IntParam(0, -1, 10, doc="Target channel index for one-vs-others mode"),
            }
        }

    def setup(self):
        try:
            from phyid.calculate import calc_PhiID
        except ImportError:
            raise ImportError(
                "The phyid package is not installed. Please install it with the following command:\n"
                "pip install git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git"
            )

        self.calc_PhiID = calc_PhiID

    def process(self, matrix: Data):
        # If no input, do nothing
        if matrix is None or matrix.data is None:
            return None
        # Ensure data is a 2D array: channels x timepoints
        data = np.asarray(matrix.data, dtype=float)

        n_channels, n_time = data.shape

        # Read parameters

        tau = int(self.params["PhiID"]["tau"].value)
        kind = self.params["PhiID"]["kind"].value
        redundancy = self.params["PhiID"]["redudancy"].value

        # List of atom names in fixed order
        atom_names = [
            "rtr",
            "rtx",
            "rty",
            "rts",
            "xtr",
            "xtx",
            "xty",
            "xts",
            "ytr",
            "ytx",
            "yty",
            "yts",
            "str",
            "stx",
            "sty",
            "sts",
        ]
        n_atoms = len(atom_names)

        # retrieve channel labels from metadata if available, otherwise create default labels
        channel_labels = None
        if matrix.meta and "channels" in matrix.meta and "dim0" in matrix.meta["channels"]:
            channel_labels = matrix.meta["channels"]["dim0"]
        else:
            channel_labels = [f"ch{i}" for i in range(n_channels)]

        if self.params.PhiID.mode.value == "pairwise":
            # Prepare output array: one row per channel, one col per atom
            PhiID_vals = np.zeros((n_atoms, n_channels, n_channels), dtype=np.float32)
            inf_dyn_vals = np.zeros((len(information_dynamics_metrics), n_channels, n_channels), dtype=np.float32)
            IIT_vals = np.zeros((len(IIT_metrics), n_channels, n_channels), dtype=np.float32)

            # Compute PhiID for each channel vs. each other channel
            for i in range(n_channels):
                src = data[i]
                for j in range(n_channels):
                    if i == j:
                        continue
                    src = data[j]
                    tgt = data[i]
                    # Run the PhiID calculation
                    atoms_res, _ = self.calc_PhiID(src, tgt, tau, kind=kind, redundancy=redundancy)
                    # Each atoms_res[name] is a vector length n_time - tau
                    # We average over time to get a single scalar per atom
                    for k, name in enumerate(atom_names):
                        PhiID_vals[k, i, j] = float(np.mean(atoms_res[name]))
                    for k, name in enumerate(information_dynamics_metrics):
                        # Get the indices of the atoms in the information_dynamics_metrics dict
                        atom_indices = [atom_names.index(atom) for atom in information_dynamics_metrics[name]]
                        # Sum the values of the atoms and average over time
                        inf_dyn_vals[k, i, j] = float(np.mean(np.sum(PhiID_vals[atom_indices, i, j], axis=0)))
                    for k, name in enumerate(IIT_metrics):
                        # Get the indices of the atoms in the IIT_metrics dict
                        atom_indices = [atom_names.index(atom) for atom in IIT_metrics[name]]
                        # Sum the values of the atoms and average over time
                        IIT_vals[k, i, j] = float(np.mean(np.sum(PhiID_vals[atom_indices, i, j], axis=0)))
                        if name == "Integrated information":
                            # Subtract rtr
                            IIT_vals[k, i, j] -= float(np.mean(atoms_res["rtr"]))

        elif self.params.PhiID.mode.value == "one-vs-others":
            # TODO: validate number of channels

            # Prepare output array: one row per channel, one col per atom
            PhiID_vals = np.zeros((n_atoms, n_channels), dtype=np.float32)
            inf_dyn_vals = np.zeros((len(information_dynamics_metrics), n_channels), dtype=np.float32)
            IIT_vals = np.zeros((len(IIT_metrics), n_channels), dtype=np.float32)

            # Compute PhiID for each channel vs. the mean of all other channels
            tgt_index = self.params.PhiID.tgt_index.value
            tgt = data[tgt_index]
            for i in range(n_channels):
                if i == tgt_index:
                    continue

                # Run the PhiID calculation
                src = data[i]
                atoms_res, _ = self.calc_PhiID(src, tgt, tau, kind=kind, redundancy=redundancy)

                # Each atoms_res[name] is a vector length n_time - tau
                # We average over time to get a single scalar per atom
                for j, name in enumerate(atom_names):
                    PhiID_vals[j, i] = float(np.mean(atoms_res[name]))
                for j, name in enumerate(information_dynamics_metrics):
                    # Get the indices of the atoms in the information_dynamics_metrics dict
                    atom_indices = [atom_names.index(atom) for atom in information_dynamics_metrics[name]]
                    # Sum the values of the atoms and average over time
                    inf_dyn_vals[j, i] = float(np.mean(np.sum(PhiID_vals[atom_indices, i], axis=0)))
                for j, name in enumerate(IIT_metrics):
                    # Get the indices of the atoms in the IIT_metrics dict
                    atom_indices = [atom_names.index(atom) for atom in IIT_metrics[name]]
                    # Sum the values of the atoms and average over time
                    IIT_vals[j, i] = float(np.mean(np.sum(PhiID_vals[atom_indices, i], axis=0)))
                    if name == "Integrated information":
                        # Subtract rtr
                        IIT_vals[j, i] -= float(np.mean(atoms_res["rtr"]))

            mask = np.arange(n_channels) != tgt_index
            PhiID_vals = PhiID_vals[:, mask]
            inf_dyn_vals = inf_dyn_vals[:, mask]
            IIT_vals = IIT_vals[:, mask]
            channel_labels = [channel_labels[i] for i in range(n_channels) if i != tgt_index]

        out_phiid = matrix.meta.copy()
        out_phiid["channels"] = {"dim0": atom_names, "dim1": channel_labels}

        out_inf_dyn = matrix.meta.copy()
        out_inf_dyn["channels"] = {"dim0": list(information_dynamics_metrics.keys()), "dim1": channel_labels}

        out_iit = matrix.meta.copy()
        out_iit["channels"] = {"dim0": list(IIT_metrics.keys()), "dim1": channel_labels}

        return {"PhiID": (PhiID_vals, out_phiid), "inf_dyn": (inf_dyn_vals, out_inf_dyn), "IIT": (IIT_vals, out_iit)}
