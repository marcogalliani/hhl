"""Sketch of the HHL algorithm in Qiskit.
- 1 system qubit (amplitude encoding of b/x)
- n phase-estimation qubits (basis encoding of eigenvalues)
- 1 ancilla qubit for the reciprocal rotation
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import NamedTuple

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFT, PauliEvolutionGate, RYGate
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis.evolution import LieTrotter, SuzukiTrotter

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler


class SimulationResult(NamedTuple):
    raw: dict[str, float]          # P(outcome) for all outcomes
    conditional: dict[str, float]  # P(outcome | anc=1), keyed by same bitstrings

@dataclass(frozen=True)
class HHLConfig:
    phase_qubits: int = 2
    evolution_time: float = 3 * np.pi / 4
    reciprocal_scale: float = 2 / 3  # must satisfy C ≤ λ_min
    trotter_steps: int = 1

class HHLAlgorithm:
    """HHL algorithm for 2×2 Hermitian systems using Trotterized Hamiltonian simulation.

    Pass a Qiskit backend to run on real hardware or an IBM simulator; omit it to
    fall back to local statevector-based shot simulation via BasicSimulator.
    """

    def __init__(self, config: HHLConfig = HHLConfig(), backend=None) -> None:
        self.config = config
        self.backend = backend
        self.system = QuantumRegister(1, "sys")
        self.phase = QuantumRegister(config.phase_qubits, "phase")
        self.ancilla = QuantumRegister(1, "anc")
        self.classical = ClassicalRegister(2, "c")
        self.circuit: QuantumCircuit | None = None

    # ------------------------------------------------------------------
    # Private helpers — operate on self.circuit and self.*Register
    # ------------------------------------------------------------------

    def __prepare_b_state(self, b: np.ndarray) -> None:
        b = np.asarray(b, dtype=complex)
        norm = np.linalg.norm(b)
        if norm == 0:
            raise ValueError("Input state b must be non-zero.")
        self.circuit.initialize(b / norm, self.system[0])

    def __apply_phase_estimation(self, hamiltonian: SparsePauliOp) -> None:
        # PauliEvolutionGate(H, t) = e^{-iHt}; we want e^{iAt} so pass time = -total_time
        #synthesis = LieTrotter(reps=self.config.trotter_steps)
        synthesis = SuzukiTrotter(order=2, reps=self.config.trotter_steps)
        for phase_qubit in self.phase:
            self.circuit.h(phase_qubit)
        for power, phase_qubit in enumerate(self.phase):
            total_time = self.config.evolution_time * (2 ** power)
            evo = PauliEvolutionGate(hamiltonian, time=-total_time, synthesis=synthesis).control(1)
            self.circuit.append(evo, [phase_qubit, self.system[0]])
        self.circuit.append(QFT(len(self.phase), inverse=True, do_swaps=True), self.phase)

    def __apply_reciprocal_rotation(self) -> None:
        n = len(self.phase)
        num_phase_states = 2 ** n
        for basis_state in range(num_phase_states):
            bitstring = format(basis_state, f"0{n}b")
            # QPE encodes λ as j = round(2^n · λ·t / 2π), so λ̃ = 2π·j / (2^n·t)
            estimated_lambda = max(
                1e-8,
                2.0 * np.pi * basis_state / (num_phase_states * self.config.evolution_time),
            )
            angle = 2.0 * np.arcsin(np.clip(self.config.reciprocal_scale / estimated_lambda, -1.0, 1.0))
            for index, bit in enumerate(reversed(bitstring)):
                if bit == "0":
                    self.circuit.x(self.phase[index])
            self.circuit.append(RYGate(angle).control(n), list(self.phase) + [self.ancilla[0]])
            for index, bit in enumerate(reversed(bitstring)):
                if bit == "0":
                    self.circuit.x(self.phase[index])

    def __uncompute_phase_estimation(self, hamiltonian: SparsePauliOp) -> None:
        # inverse of e^{iAt} is e^{-iAt} = PauliEvolutionGate(H, +total_time)
        synthesis = LieTrotter(reps=self.config.trotter_steps)
        self.circuit.append(QFT(len(self.phase), inverse=False, do_swaps=True), self.phase)
        for power, phase_qubit in reversed(list(enumerate(self.phase))):
            total_time = self.config.evolution_time * (2 ** power)
            evo = PauliEvolutionGate(hamiltonian, time=total_time, synthesis=synthesis).control(1)
            self.circuit.append(evo, [phase_qubit, self.system[0]])
        for phase_qubit in self.phase:
            self.circuit.h(phase_qubit)

    def __extract_solution(self) -> tuple[np.ndarray, float]:
        anc_bit = 1 + self.config.phase_qubits
        sv = Statevector(self.circuit)
        amplitudes = np.array([sv.data[v + (1 << anc_bit)] for v in range(2)])
        p_success = float(np.sum(np.abs(amplitudes) ** 2))
        if p_success > 1e-12:
            amplitudes = amplitudes / np.sqrt(p_success)
        return amplitudes, p_success

    @staticmethod
    def __counts_to_result(counts: dict[str, int], shots: int) -> SimulationResult:
        raw = {outcome: count / shots for outcome, count in sorted(counts.items())}
        success = {o: p for o, p in raw.items() if o.endswith("1")}
        p_anc1 = sum(success.values())
        conditional = {o: p / p_anc1 for o, p in success.items()} if p_anc1 > 0 else {}
        return SimulationResult(raw=raw, conditional=conditional)

    def __measure_pauli_expectation(
        self, A: np.ndarray, b: np.ndarray, pauli: str, shots: int
    ) -> float:
        """Return ⟨x|P|x⟩ post-selected on ancilla=1 via shot-based measurement.

        Applies a basis rotation so that the Pauli eigenstates map to |0⟩/|1⟩,
        then reads ⟨P⟩ = P(+1 eigenvalue | anc=1) − P(−1 eigenvalue | anc=1).
        """
        if pauli == "I":
            return 1.0
        circuit = self.build_circuit(A, b, add_measurements=False)
        if pauli == "X":
            circuit.h(self.system[0])
        elif pauli == "Y":
            circuit.sdg(self.system[0])
            circuit.h(self.system[0])
        circuit.measure(self.ancilla[0], self.classical[0])
        circuit.measure(self.system[0], self.classical[1])
        result = self.__counts_to_result(self.run_circuit(circuit, shots), shots)
        p_plus  = result.conditional.get("01", 0.0)  # sys=0 in rotated basis → +1 eigenvalue
        p_minus = result.conditional.get("11", 0.0)  # sys=1 in rotated basis → −1 eigenvalue
        return p_plus - p_minus

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_observable(
        self, M: np.ndarray, A: np.ndarray, b: np.ndarray, shots: int = 1024
    ) -> float:
        """Compute ⟨x|M|x⟩ for the normalized solution |x⟩ = A⁻¹|b⟩ / ‖A⁻¹b‖.
        To recover the un-normalized quadratic form multiply by ‖A⁻¹b‖².
        """
        M = np.asarray(M, dtype=complex)
        A = np.asarray(A, dtype=complex)
        b = np.asarray(b, dtype=complex)

        if self.backend is None:
            x_hhl, _ = self.solve(A, b)
            quantum = float(np.real(x_hhl.conj() @ M @ x_hhl))
        else:
            M_op = SparsePauliOp.from_operator(M)
            quantum = sum(
                float(np.real(coeff)) * self.__measure_pauli_expectation(A, b, str(pauli), shots)
                for pauli, coeff in zip(M_op.paulis, M_op.coeffs)
            )
        return quantum

    def build_circuit(self, A: np.ndarray, b: np.ndarray, add_measurements: bool = True) -> QuantumCircuit:
        """Build and return the HHL circuit for system Ax=b."""
        A = np.asarray(A, dtype=complex)
        if A.shape != (2, 2):
            raise ValueError("This sketch currently targets 2×2 Hermitian systems.")
        if not np.allclose(A, A.conj().T):
            raise ValueError("The system matrix must be Hermitian.")

        self.circuit = QuantumCircuit(self.system, self.phase, self.ancilla, self.classical, name="hhl_2x2")
        hamiltonian = SparsePauliOp.from_operator(A)

        self.__prepare_b_state(b)
        self.__apply_phase_estimation(hamiltonian)
        self.__apply_reciprocal_rotation()
        self.__uncompute_phase_estimation(hamiltonian)

        if add_measurements:
            self.circuit.measure(self.ancilla[0], self.classical[0])
            self.circuit.measure(self.system[0], self.classical[1])
            
        return self.circuit

    def solve(self, A: np.ndarray, b: np.ndarray, shots: int = 1024) -> tuple[np.ndarray, float]:
        """Return (normalized solution amplitudes, p_success).
        """        
        sim = self.simulate(A, b, shots=shots)
        p0 = sim.conditional.get("01", 0.0)  # P(sys=0 | anc=1)
        p1 = sim.conditional.get("11", 0.0)  # P(sys=1 | anc=1)
        p_success = sum(p for o, p in sim.raw.items() if o.endswith("1"))
        return np.array([np.sqrt(p0), np.sqrt(p1)]), p_success

    def simulate(self, A: np.ndarray, b: np.ndarray, shots: int = 1024) -> SimulationResult:
        """Run the HHL circuit and return raw and ancilla-conditioned outcome probabilities.

        The 2-bit outcome key is 'sys anc' (c[1] c[0]).
        HHL success outcomes (ancilla=1): '01' (sys=0) and '11' (sys=1).
        """
        self.build_circuit(A, b, add_measurements=True)
        return self.__counts_to_result(self.run_circuit(shots), shots)
    
    def run_circuit(self, shots: int) -> dict[str, int]:
        """Run a circuit on self.backend (or BasicSimulator) and return raw counts."""
        if self.backend is None:
            local = BasicSimulator()
            return local.run(transpile(self.circuit, local), shots=shots).result().get_counts()
        isa = generate_preset_pass_manager(backend=self.backend, optimization_level=3).run(self.circuit)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots = shots
        return sampler.run([isa]).result()[0].data.c.get_counts()
