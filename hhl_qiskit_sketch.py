"""Sketch of the HHL algorithm in Qiskit.
- 1 system qubit (amplitude encoding of b/x)
- n phase-estimation qubits (basis encoding of eigenvalues)
- 1 ancilla qubit for the reciprocal rotation
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, RYGate


@dataclass(frozen=True)
class HHLConfig:
    phase_qubits: int = 2
    evolution_time: float = 3 * np.pi / 4
    reciprocal_scale: float = 2 / 3  # must satisfy C ≤ λ_min; paper's C=1 assumes λ_min≥1
    trotter_steps: int = 1


class HHLAlgorithm:
    """HHL algorithm for 2×2 Hermitian systems using Trotterized Hamiltonian simulation."""

    def __init__(self, config: HHLConfig = HHLConfig()) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def __pauli_decompose_2x2(matrix: np.ndarray) -> tuple[float, float, float, float]:
        """Decompose A = α_I·I + α_X·X + α_Y·Y + α_Z·Z via α_j = Tr(σ_j A) / 2."""
        alpha_i = float(np.real(matrix[0, 0] + matrix[1, 1])) / 2
        alpha_x = float(np.real(matrix[0, 1] + matrix[1, 0])) / 2
        alpha_y = float(-np.imag(matrix[0, 1]))
        alpha_z = float(np.real(matrix[0, 0] - matrix[1, 1])) / 2
        return alpha_i, alpha_x, alpha_y, alpha_z

    def __append_controlled_trotter_step(
        self,
        circuit: QuantumCircuit,
        control_qubit,
        system_qubit,
        pauli_coeffs: tuple[float, float, float, float],
        dt: float,
    ) -> None:
        """One first-order Trotter step for controlled-e^{iA·dt}.

        Identity term → P(α_I·dt) on control qubit (phase kickback).
        Pauli terms → CRK(-2α·dt) since RK(θ) = e^{-iσθ/2}.
        """
        alpha_i, alpha_x, alpha_y, alpha_z = pauli_coeffs
        if abs(alpha_i) > 1e-12:
            circuit.p(alpha_i * dt, control_qubit)
        if abs(alpha_x) > 1e-12:
            circuit.crx(-2.0 * alpha_x * dt, control_qubit, system_qubit)
        if abs(alpha_y) > 1e-12:
            circuit.cry(-2.0 * alpha_y * dt, control_qubit, system_qubit)
        if abs(alpha_z) > 1e-12:
            circuit.crz(-2.0 * alpha_z * dt, control_qubit, system_qubit)

    def __prepare_b_state(
        self, circuit: QuantumCircuit, system: QuantumRegister, b: np.ndarray
    ) -> None:
        b = np.asarray(b, dtype=complex)
        norm = np.linalg.norm(b)
        if norm == 0:
            raise ValueError("Input state b must be non-zero.")
        circuit.initialize(b / norm, system[0])

    def __apply_phase_estimation(
        self,
        circuit: QuantumCircuit,
        system_qubit,
        phase_register,
        pauli_coeffs: tuple[float, float, float, float],
    ) -> None:
        for phase_qubit in phase_register:
            circuit.h(phase_qubit)
        for power, phase_qubit in enumerate(phase_register):
            total_time = self.config.evolution_time * (2 ** power)
            dt = total_time / self.config.trotter_steps
            for _ in range(self.config.trotter_steps):
                self.__append_controlled_trotter_step(
                    circuit, phase_qubit, system_qubit, pauli_coeffs, dt
                )
        circuit.append(QFT(len(phase_register), inverse=True, do_swaps=True), phase_register)

    def __apply_reciprocal_rotation(
        self,
        circuit: QuantumCircuit,
        phase_register,
        ancilla_qubit,
    ) -> None:
        n = len(phase_register)
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
                    circuit.x(phase_register[index])
            circuit.append(RYGate(angle).control(n), list(phase_register) + [ancilla_qubit])
            for index, bit in enumerate(reversed(bitstring)):
                if bit == "0":
                    circuit.x(phase_register[index])

    def __uncompute_phase_estimation(
        self,
        circuit: QuantumCircuit,
        system_qubit,
        phase_register,
        pauli_coeffs: tuple[float, float, float, float],
    ) -> None:
        circuit.append(QFT(len(phase_register), inverse=False, do_swaps=True), phase_register)
        for power, phase_qubit in reversed(list(enumerate(phase_register))):
            total_time = self.config.evolution_time * (2 ** power)
            dt = total_time / self.config.trotter_steps
            for _ in range(self.config.trotter_steps):
                self.__append_controlled_trotter_step(
                    circuit, phase_qubit, system_qubit, pauli_coeffs, -dt
                )
        for phase_qubit in phase_register:
            circuit.h(phase_qubit)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_circuit(
        self, A: np.ndarray, b: np.ndarray, add_measurements: bool = True
    ) -> QuantumCircuit:
        """Build and return the HHL circuit for system Ax=b."""
        A = np.asarray(A, dtype=complex)
        if A.shape != (2, 2):
            raise ValueError("This sketch currently targets 2×2 Hermitian systems.")
        if not np.allclose(A, A.conj().T):
            raise ValueError("The system matrix must be Hermitian.")

        system = QuantumRegister(1, "sys")
        phase = QuantumRegister(self.config.phase_qubits, "phase")
        ancilla = QuantumRegister(1, "anc")
        classical = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(system, phase, ancilla, classical, name="hhl_2x2")

        pauli_coeffs = self.__pauli_decompose_2x2(A)

        self.__prepare_b_state(circuit, system, b)
        self.__apply_phase_estimation(circuit, system[0], phase, pauli_coeffs)
        self.__apply_reciprocal_rotation(circuit, phase, ancilla[0])
        self.__uncompute_phase_estimation(circuit, system[0], phase, pauli_coeffs)

        if add_measurements:
            circuit.measure(ancilla[0], classical[0])
            circuit.measure(system[0], classical[1])
        return circuit

    def solve(self, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        """Simulate the HHL circuit and return (normalized solution amplitudes, p_success)."""
        from qiskit.quantum_info import Statevector

        sv = Statevector(self.build_circuit(A, b, add_measurements=False))
        return self.__extract_solution(sv.data, self.config.phase_qubits)

    @staticmethod
    def __extract_solution(sv_data: np.ndarray, phase_qubits: int) -> tuple[np.ndarray, float]:
        """Post-select on ancilla=|1⟩, phase=|0…0⟩.

        Qubit layout: sys(0), phase[0..n-1](1..n), anc(n+1).
        Index for sys=v, phase=0, anc=1 is v + 2^(n+1).
        """
        anc_bit = 1 + phase_qubits
        amplitudes = np.array([sv_data[v + (1 << anc_bit)] for v in range(2)])
        p_success = float(np.sum(np.abs(amplitudes) ** 2))
        if p_success > 1e-12:
            amplitudes = amplitudes / np.sqrt(p_success)
        return amplitudes, p_success


if __name__ == "__main__":
    A = np.array([[1.0, -1.0 / 3.0], [-1.0 / 3.0, 1.0]])
    b_vec = np.array([1.0, 0.0])

    solver = HHLAlgorithm()
    print(solver.build_circuit(A, b_vec).draw(output="text"))

    x_cl = np.linalg.solve(A, b_vec)
    x_cl_norm = x_cl / np.linalg.norm(x_cl)

    x_hhl, p_success = solver.solve(A, b_vec)

    print("\n=== Solution comparison ===")
    print(f"  A = {A.tolist()}")
    print(f"  b = {b_vec.tolist()}")
    print()
    print(f"  Classical x = A⁻¹b          : {x_cl}")
    print(f"  Classical normalized |x_cl⟩  : {x_cl_norm}")
    print()
    print(f"  HHL |x_hhl⟩ (anc=1, phase=00): {x_hhl.real}")
    print(f"  P(ancilla=1, phase=|00⟩)      : {p_success:.6f}")
    print()
    probs_cl = x_cl_norm ** 2
    probs_hhl = np.abs(x_hhl) ** 2
    print(f"  Measurement probabilities  sys=0    sys=1")
    print(f"    classical                {probs_cl[0]:.4f}   {probs_cl[1]:.4f}")
    print(f"    HHL                      {probs_hhl[0]:.4f}   {probs_hhl[1]:.4f}")
    print()
    fidelity = abs(x_cl_norm @ x_hhl.conj()) ** 2
    print(f"  Fidelity |⟨x_cl|x_hhl⟩|²    : {fidelity:.6f}")
