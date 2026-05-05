"""Sketch of the HHL algorithm in Qiskit.

This is a teaching-oriented scaffold that mirrors the 4-qubit walkthrough
structure from the PDF:

- 1 system qubit to hold |x>
- 2 phase-estimation qubits
- 1 ancilla qubit for the reciprocal rotation

The circuit is intentionally narrow and uses a 2x2 Hermitian example where the
Hamiltonian evolution can be approximated with a single-qubit X rotation.
Replace the placeholder Hamiltonian block with a proper simulation routine when
you move beyond the sketch stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # pyright: ignore[reportMissingImports]
from qiskit.circuit.library import QFT, RYGate, RXGate  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class HHLConfig:
    matrix: np.ndarray
    b: np.ndarray
    phase_qubits: int = 2
    evolution_time: float = np.pi / 2
    reciprocal_scale: float = 0.25


def _normalize_state(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=complex)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Input state must be non-zero.")
    return vector / norm


def _prepare_b_state(circuit: QuantumCircuit, system: QuantumRegister, b: np.ndarray) -> None:
    circuit.initialize(_normalize_state(b), system[0])


def _apply_phase_estimation(circuit: QuantumCircuit, system_qubit, phase_register, evolution_time: float) -> None:
    # The paper's 2x2 example can be reduced to a single-qubit X evolution up to
    # a global phase. This is only a sketch of the Hamiltonian simulation step.
    base_angle = 2.0 * evolution_time / 3.0

    for phase_qubit in phase_register:
        circuit.h(phase_qubit)

    for power, phase_qubit in enumerate(phase_register):
        angle = base_angle * (2**power)
        circuit.append(RXGate(angle).control(1), [phase_qubit, system_qubit])

    circuit.append(QFT(len(phase_register), inverse=True, do_swaps=False), phase_register)


def _uncompute_phase_estimation(circuit: QuantumCircuit, system_qubit, phase_register, evolution_time: float) -> None:
    base_angle = 2.0 * evolution_time / 3.0

    circuit.append(QFT(len(phase_register), inverse=False, do_swaps=False), phase_register)

    for power, phase_qubit in reversed(list(enumerate(phase_register))):
        angle = -base_angle * (2**power)
        circuit.append(RXGate(angle).control(1), [phase_qubit, system_qubit])

    for phase_qubit in phase_register:
        circuit.h(phase_qubit)


def _apply_reciprocal_rotation(
    circuit: QuantumCircuit,
    phase_register,
    ancilla_qubit,
    reciprocal_scale: float,
) -> None:
    # Basis-state controlled RY rotations. For a 2-qubit phase register this is
    # easy to inspect and good enough for a sketch.
    num_phase_states = 2 ** len(phase_register)

    for basis_state in range(num_phase_states):
        bitstring = format(basis_state, f"0{len(phase_register)}b")
        estimated_lambda = max(1e-8, 2.0 * np.pi * basis_state / num_phase_states)
        ratio = np.clip(reciprocal_scale / estimated_lambda, -1.0, 1.0)
        angle = 2.0 * np.arcsin(ratio)

        for index, bit in enumerate(reversed(bitstring)):
            if bit == "0":
                circuit.x(phase_register[index])

        controlled_ry = RYGate(angle).control(len(phase_register))
        circuit.append(controlled_ry, list(phase_register) + [ancilla_qubit])

        for index, bit in enumerate(reversed(bitstring)):
            if bit == "0":
                circuit.x(phase_register[index])


def build_hhl_sketch(config: HHLConfig) -> QuantumCircuit:
    """Build a compact HHL sketch circuit.

    The circuit keeps the paper's pedagogical structure, but the Hamiltonian
    simulation and reciprocal rotation are intentionally simplified.
    """

    matrix = np.asarray(config.matrix, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("This sketch currently targets a 2x2 Hermitian example.")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("The system matrix must be Hermitian.")

    system = QuantumRegister(1, "sys")
    phase = QuantumRegister(config.phase_qubits, "phase")
    ancilla = QuantumRegister(1, "anc")
    classical = ClassicalRegister(2, "c")
    circuit = QuantumCircuit(system, phase, ancilla, classical, name="hhl_sketch")

    _prepare_b_state(circuit, system, config.b)
    _apply_phase_estimation(circuit, system[0], phase, config.evolution_time)
    _apply_reciprocal_rotation(circuit, phase, ancilla[0], config.reciprocal_scale)
    _uncompute_phase_estimation(circuit, system[0], phase, config.evolution_time)

    circuit.measure(ancilla[0], classical[0])
    circuit.measure(system[0], classical[1])
    return circuit


def example_walkthrough_circuit() -> QuantumCircuit:
    """Return a paper-inspired demo circuit."""

    example_matrix = np.array([[1.0, -1.0 / 3.0], [-1.0 / 3.0, 1.0]], dtype=float)
    example_b = np.array([1.0, 0.0], dtype=float)
    return build_hhl_sketch(HHLConfig(matrix=example_matrix, b=example_b))


if __name__ == "__main__":
    demo_circuit = example_walkthrough_circuit()
    print(demo_circuit.draw(output="text"))