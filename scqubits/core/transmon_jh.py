# transmon_jh.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import math

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
from scipy.stats import pearsonr


from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy import ndarray

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.core.transmon as pure_transmon

import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem
from scqubits.core.storage import WaveFunction

LevelsTuple = Tuple[int, ...]
Transition = Tuple[int, int]
TransitionsTuple = Tuple[Transition, ...]

# Transmon with harmonics


class Transmon_jh(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the Cooper-pair-box and transmon qubit. The Hamiltonian is
    represented in dense form in the number basis,
    :math:`H_\text{CPB}=4E_\text{C}(\hat{n}-n_g)^2-\frac{E_\text{J}}{2}(
    |n\rangle\langle n+1|+\text{h.c.})`.
    Initialize with, for example::

        Transmon_jh(EJs=[10.0, -2.0, 0.8], EC=2.0, ng=0.2, ncut=30)

    Parameters
    ----------
    EJs:
       Josephson energies with harmonics :math:`E_\text{Js}`
    EC:
        charging energy
    ng:
        offset charge
    ncut:
        charge basis cutoff, `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    esys_method:
        method for esys diagonalization, callable or string representation
    esys_method_options:
        dictionary with esys diagonalization options
    evals_method:
        method for evals diagonalization, callable or string representation
    evals_method_options:
        dictionary with evals diagonalization options
    """

    EJs = descriptors.WatchedProperty(List[float], "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")


    def __init__(
        self,
        EJs: List[float],  # List of Josephson energies for each harmonic
        EC: float,
        ng: float,
        ncut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
        evals_method: Union[Callable, str, None] = None,
        evals_method_options: Union[dict, None] = None,
        esys_method: Union[Callable, str, None] = None,
        esys_method_options: Union[dict, None] = None,
    ) -> None:
        base.QubitBaseClass.__init__(
            self,
            id_str=id_str,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.EJs = EJs
        self.m = len(EJs)
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-np.pi, np.pi, 151)
        self._default_n_range = (-5, 6)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {"EJs": [15.0], "EC": 0.3, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_charge_impedance",
        ]

    @classmethod
    def effective_noise_channels(cls) -> List[str]:
        """Return a default list of channels used when calculating effective t1 and
        t2 noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    def _hamiltonian_diagonal(self) -> ndarray:
        dimension = self.hilbertdim()
        return 4.0 * self.EC * (np.arange(dimension) - self.ncut - self.ng) ** 2

    def _hamiltonian_offdiagonal(self) -> ndarray:
        dimension = self.hilbertdim()
        off_diagonal = np.zeros(shape=(dimension - 1,))
        for m, EJ_m in enumerate(self.EJs, start=1):
            off_diagonal -= EJ_m / 2.0
        return off_diagonal

    def _evals_calc(self, evals_count: int) -> ndarray:
        diagonal = self._hamiltonian_diagonal()
        off_diagonal = self._hamiltonian_offdiagonal()

        evals = sp.linalg.eigvalsh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, evals_count - 1),
            check_finite=False,
        )
        return evals

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        diagonal = self._hamiltonian_diagonal()
        off_diagonal = self._hamiltonian_offdiagonal()

        evals, evecs = sp.linalg.eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, evals_count - 1),
            check_finite=False,
        )
        return evals, evecs

    @staticmethod
    def find_EJ_EC(
        E01: float, anharmonicity: float, ng=0, ncut=30
    ) -> Tuple[float, float]:
        """
        Finds the EJs and EC values given a qubit splitting `E01` and `anharmonicity`.

        Parameters
        ----------
            E01:
                qubit transition energy
            anharmonicity:
                absolute qubit anharmonicity, (E2-E1) - (E1-E0)
            ng:
                offset charge (default: 0)
            ncut:
                charge number cutoff (default: 30)

        Returns
        -------
            A tuple of the EJs and EC values representing the best fit.
        """
        tmon = Transmon_jh(EJs=[10.0], EC=0.1, ng=ng, ncut=ncut)
        start_EJ_EC = np.array([tmon.EJs, tmon.EC])

        def cost_func(EJ_EC: Tuple[float, float]) -> float:
            EJs, EC = EJ_EC
            tmon.EJs = EJs
            tmon.EC = EC
            energies = tmon.eigenvals(evals_count=3)
            computed_E01 = energies[1] - energies[0]
            computed_anharmonicity = energies[2] - energies[1] - computed_E01
            cost = (E01 - computed_E01) ** 2
            cost += (anharmonicity - computed_anharmonicity) ** 2
            return cost

        return sp.optimize.minimize(cost_func, start_EJ_EC).x

    def n_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns charge operator n in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns charge operator n in the charge basis.
            If `True`, energy eigenspectrum is computed, returns charge operator n in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns charge operator n in the energy eigenbasis, and does not have to recalculate the
            eigenspectrum.

        Returns
        -------
            Charge operator n in chosen basis as ndarray.
            For `energy_esys=True`, n has dimensions of `truncated_dim` x `truncated_dim`.
            If an actual eigensystem is handed to `energy_sys`, then `n` has dimensions of m x m,
            where m is the number of given eigenvectors.
        """
        diag_elements = np.arange(-self.ncut, self.ncut + 1, 1)
        native = np.diag(diag_elements)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def exp_i_phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`e^{i\\varphi}` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`e^{i\\varphi}` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`e^{i\\varphi}` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`e^{i\\varphi}` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`e^{i\\varphi}` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`e^{i\\varphi}` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`e^{i\\varphi}` has dimensions of m x m,
            for m given eigenvectors.
        """
        dimension = self.hilbertdim()
        entries = np.repeat(1.0, dimension - 1)
        exp_op = np.diag(entries, -1)
        return self.process_op(native_op=exp_op, energy_esys=energy_esys)

    def exp_i_m_phi_operator(
            self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`e^{i m \\varphi}` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`e^{i m \\varphi}` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`e^{i m \\varphi}` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`e^{i m \\varphi}` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`e^{i m \\varphi}` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`e^{i m \\varphi}` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`e^{i m \\varphi}` has dimensions of n x n,
            for n given eigenvectors.
        """
        dimension = self.hilbertdim()
        entries = np.repeat(1.0, dimension - self.m)
        exp_op = np.diag(entries, - self.m)
        return self.process_op(native_op=exp_op, energy_esys=energy_esys)

    def cos_phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos \\varphi` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos \\varphi` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos \\varphi` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos \\varphi` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos \\varphi` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos \\varphi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\varphi` has dimensions of m x m,
            for m given eigenvectors.
        """
        cos_op = 0.5 * self.exp_i_phi_operator()
        cos_op += cos_op.T
        return self.process_op(native_op=cos_op, energy_esys=energy_esys)

    def cos_m_phi_operator(
            self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos m \\varphi` in the charge or eigenenergy basis with higher harmonics.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos m \\varphi` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos m \\varphi` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos m \\varphi` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos m \\varphi` in chosen basis as ndarray with higher harmonics. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos m \\varphi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\varphi` has dimensions of m x m,
            for m given eigenvectors.
        """
        cos_op = 0.5 * self.exp_i_m_phi_operator()
        cos_op += cos_op.T
        return self.process_op(native_op=cos_op, energy_esys=energy_esys)

    def sin_phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\sin \\varphi` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\sin \\varphi` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\sin \\varphi` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\sin \\varphi` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\sin \\varphi` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\sin \\varphi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\sin \\varphi` has dimensions of m x m,
            for m given eigenvectors.
        """
        sin_op = -1j * 0.5 * self.exp_i_phi_operator()
        sin_op += sin_op.conjugate().T
        return self.process_op(native_op=sin_op, energy_esys=energy_esys)

    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns Hamiltonian in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns Hamiltonian in the charge basis.
            If `True`, the energy eigenspectrum is computed; returns Hamiltonian in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors); then return the Hamiltonian in the energy eigenbasis, do not recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian in chosen basis as ndarray. For `energy_esys=False`, the Hamiltonian has dimensions of
            `truncated_dim` x `truncated_dim`. For `energy_sys=esys`, the Hamiltonian has dimensions of m x m,
            for m given eigenvectors.
        """
        dimension = self.hilbertdim()
        hamiltonian_mat = np.diag(
            [
                4.0 * self.EC * (ind - self.ncut - self.ng) ** 2
                for ind in range(dimension)
            ]
        )
        # Tunneling terms with harmonics
        for m, EJ_m in enumerate(self.EJs, start=1):
            ind = np.arange(dimension - m)
            hamiltonian_mat[ind, ind + m] -= EJ_m / 2.0
            hamiltonian_mat[ind + m, ind] -= EJ_m / 2.0
        return self.process_hamiltonian(
            native_hamiltonian=hamiltonian_mat, energy_esys=energy_esys
        )

    def d_hamiltonian_d_ng(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        charge offset `ng` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -8 * self.EC * self.n_operator(energy_esys=energy_esys)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        EJs in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -self.cos_m_phi_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return 2 * self.ncut + 1

    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """Transmon phase-basis potential evaluated at `phi`.

        Parameters
        ----------
        phi:
            phase variable value
        """
        tot_potential = np.zeros_like(phi)
        for m, EJ_m in enumerate(self.EJs, start=1):
            tot_potential -= EJ_m * np.cos(m * phi)
        return tot_potential

    def plot_n_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        mode: str = "real",
        which: int = 0,
        nrange: Tuple[int, int] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plots transmon wave function in charge basis

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        mode:
            `'abs_sqr', 'abs', 'real', 'imag'`
        which:
             index or indices of wave functions to plot (default value = 0)
        nrange:
             range of `n` to be included on the x-axis (default value = (-5,6))
        **kwargs:
            plotting parameters
        """
        if nrange is None:
            nrange = self._default_n_range
        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_wavefunc.amplitudes = amplitude_modifier(n_wavefunc.amplitudes)
        kwargs = {
            **defaults.wavefunction1d_discrete(mode),
            **kwargs,
        }  # if any duplicates, later ones survive
        return plot.wavefunction1d_discrete(n_wavefunc, xlim=nrange, **kwargs)

    def plot_phi_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        which: int = 0,
        phi_grid: Grid1d = None,
        mode: str = "abs_sqr",
        scaling: float = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Alias for plot_wavefunction"""
        return self.plot_wavefunction(
            esys=esys,
            which=which,
            phi_grid=phi_grid,
            mode=mode,
            scaling=scaling,
            **kwargs
        )

    def numberbasis_wavefunction(
        self, esys: Tuple[ndarray, ndarray] = None, which: int = 0
    ) -> WaveFunction:
        """Return the transmon wave function in number basis. The specific index of the
        wave function to be returned is `which`.

        Parameters
        ----------
        esys:
            if `None`, the eigensystem is calculated on the fly; otherwise, the provided
            eigenvalue, eigenvector arrays as obtained from `.eigensystem()`,
            are used (default value = None)
        which:
            eigenfunction index (default value = 0)
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count=evals_count)
        evals, evecs = esys

        n_vals = np.arange(-self.ncut, self.ncut + 1)
        return storage.WaveFunction(n_vals, evecs[:, which], evals[which])

    def wavefunction(
        self,
        esys: Optional[Tuple[ndarray, ndarray]] = None,
        which: int = 0,
        phi_grid: Grid1d = None,
    ) -> WaveFunction:
        """Return the transmon wave function in phase basis. The specific index of the
        wavefunction is `which`. `esys` can be provided, but if set to `None` then it is
        calculated on the fly.

        Parameters
        ----------
        esys:
            if None, the eigensystem is calculated on the fly; otherwise, the provided
            eigenvalue, eigenvector arrays as obtained from `.eigensystem()` are used
        which:
            eigenfunction index (default value = 0)
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count=evals_count)
        else:
            evals, evecs = esys

        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)

        phi_grid = phi_grid or self._default_grid
        phi_basis_labels = phi_grid.make_linspace()
        phi_wavefunc_amplitudes = np.empty(phi_grid.pt_count, dtype=np.complex_)
        for k in range(phi_grid.pt_count):
            phi_wavefunc_amplitudes[k] = (1j**which / math.sqrt(2 * np.pi)) * np.sum(
                n_wavefunc.amplitudes
                * np.exp(1j * phi_basis_labels[k] * n_wavefunc.basis_labels)
            )
        return storage.WaveFunction(
            basis_labels=phi_basis_labels,
            amplitudes=phi_wavefunc_amplitudes,
            energy=evals[which],
        )

    def _compute_dispersion(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        transitions_tuple: TransitionsTuple = ((0, 1),),
        levels_tuple: Optional[LevelsTuple] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
    ) -> Tuple[ndarray, ndarray]:
        if dispersion_name != "ng":
            return super()._compute_dispersion(
                dispersion_name,
                param_name,
                param_vals,
                transitions_tuple=transitions_tuple,
                levels_tuple=levels_tuple,
                point_count=point_count,
                num_cpus=num_cpus,
            )

        max_level = (
            np.max(transitions_tuple) if levels_tuple is None else np.max(levels_tuple)
        )
        previous_ng = self.ng
        self.ng = 0.0
        specdata_ng_0 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.ng = 0.5
        specdata_ng_05 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.ng = previous_ng

        if levels_tuple is not None:
            dispersion = np.asarray(
                [
                    [
                        np.abs(
                            specdata_ng_0.energy_table[param_index, j]
                            - specdata_ng_05.energy_table[param_index, j]
                        )
                        for param_index, _ in enumerate(param_vals)
                    ]
                    for j in levels_tuple
                ]
            )
            return specdata_ng_0.energy_table, dispersion

        dispersion_list = []
        for i, j in transitions_tuple:
            list_ij = []
            for param_index, _ in enumerate(param_vals):
                ei_0 = specdata_ng_0.energy_table[param_index, i]
                ei_05 = specdata_ng_05.energy_table[param_index, i]
                ej_0 = specdata_ng_0.energy_table[param_index, j]
                ej_05 = specdata_ng_05.energy_table[param_index, j]
                list_ij.append(
                    np.max([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                    - np.min([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                )
            dispersion_list.append(list_ij)
        return specdata_ng_0.energy_table, np.asarray(dispersion_list)

    def plot_cosine_harmonics(self):
            """
            Plots the current-phase relation (CφR) with harmonics corrections for a given set of Josephson energies.

            Parameters:
            EJms (list or array): List of Josephson energies for each harmonic, in units of energy.
            phi_range (array, optional): Phase difference range over which to plot the CφR.
            """
            # Constants
            phi_range = np.linspace(-3.4, 3.4, 200)
            EJms = self.EJs
            hbar = 1  # Planck's constant
            e_charge = sp.constants.e  # Elementary charge, in coulombs
            GHz_to_rad_s = 2 * np.pi * 1e9  # Conversion factor from GHz to radians per second
            nanoampere = 1e-9  # Conversion factor for nanoamperes

            # Scale factor to convert energy units to current
            scale_factor = (2 * e_charge / hbar) * (GHz_to_rad_s / nanoampere)

            # Calculate critical currents based on Josephson energies
            Ics = [ejm * scale_factor for ejm in EJms]

            # Generate pure sine function for the first harmonic
            pure_sine = Ics[0] * np.sin(phi_range)

            # Sum harmonics to create total cosine function
            harm_sine = np.sum([Ic * np.sin(m * phi_range) for m, Ic in enumerate(Ics, start=1)], axis=0)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(phi_range, pure_sine, linestyle="--", color="gray", label='Pure sine ($T_n → 0$ limit)')
            plt.plot(phi_range, harm_sine, label='Sine with higher harmonics')

            # Highlight difference area between pure sine and sine with harmonics
            plt.fill_between(phi_range, 0, pure_sine - harm_sine, color="lightblue", alpha=0.6, label='Difference area')

            plt.title('Current-phase relation (CφR) with harmonics corrections')
            plt.xlabel('φ (radians)')
            plt.ylabel('I(φ) (nA)')
            ax = plt.gca()

            # Set and label x-axis ticks
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels(['-π', '-½π', '0', '½π', 'π'])

            # Annotate zero lines
            ax.annotate("", xy=(-3.4, 0), xytext=(3.4, 0), arrowprops=dict(arrowstyle="<-", color='k'))
            ax.annotate("", xy=(0, min(pure_sine)*1.10), xytext=(0, max(pure_sine)*1.10), arrowprops=dict(arrowstyle="<-", color='k'))

            # Labeling critical currents on the plot
            ax.text(0.09, max(harm_sine), '$I_c$', verticalalignment='center', horizontalalignment='left', backgroundcolor='w')
            ax.text(0.09, min(harm_sine), '$-I_c$', verticalalignment='center', horizontalalignment='left', backgroundcolor='w')

            plt.legend()
            plt.show()

            # Compute the Pearson correlation coefficient
            correlation, _ = pearsonr(pure_sine, harm_sine)

            print(f"Matching percentage (via correlation): {correlation*100:.2f}%")
            print(f"Critical current Ic (pure): {max(pure_sine):.2f} nA")
            print(f"Critical current Ic (harmonics): {max(harm_sine):.2f} nA")

    def plot_cosine_harmonics_and_transmon_eigenenergies(self,
                                                         phi_range=np.linspace(-3.4, 3.4, 200),
                                                         num_states=5):
            """
            Plots the pure and harmonics-included cosine potentials and eigenenergies of a Transmon qubit.

            Parameters:
            transmon: A Transmon qubit object from scqubits.
            phi_range (array): Phase values to plot over.
            num_states (int): Number of eigenstates to plot.
            """
            EJms = self.EJs

            # Total potential including harmonics
            harmonic_potential = np.zeros_like(phi_range)
            for m, EJm in enumerate(EJms, start=1):
                harmonic_potential -= EJm * np.cos(m * phi_range)
            harmonic_potential = harmonic_potential - self.eigenvals()[0]

            # eigenenergies
            eigenenergies = np.zeros((len(phi_range), num_states))
            for i, phi in enumerate(phi_range):
                eigenenergies[i, :] = self.eigenvals(evals_count=num_states)
            eigenenergies = eigenenergies - self.eigenvals()[0]

            # pure eigenenergies
            self.EJs = [EJms[0]]
            eigenenergies_pure = np.zeros((len(phi_range), num_states))
            pure_potential = -self.EJs[0] * np.cos(phi_range) - self.eigenvals()[0]
            for i, phi in enumerate(phi_range):
                eigenenergies_pure[i, :] = self.eigenvals(evals_count=num_states)
            eigenenergies_pure = eigenenergies_pure - self.eigenvals()[0]

            self.EJs = EJms

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(phi_range, pure_potential, label='Pure cosine potential', linestyle='--', color='gray')
            plt.plot(phi_range, harmonic_potential, label='Harmonics-included potential', color='#1f77b4')

            for idx in range(num_states):
                plt.plot(phi_range, eigenenergies_pure[:, idx],'--', color='gray')
                plt.plot(phi_range, eigenenergies[:, idx],'--' , label=f'|{idx}⟩')

            plt.title('Pure and harmonics-included potentials')
            plt.xlabel('Phase φ')
            plt.ylabel('Energy')
            plt.legend()
            plt.show()

# - Flux-tunable transmon with with harmonics -----------------------------------------
