"""Compare classical solve time vs. HHL-style circuit runtime on IBM Quantum.

Notes:
- The quantum side implements the tutorial's 2x2 HHL walkthrough circuit.
- Larger sizes (e.g., 8x8) are benchmarked only on the classical side.
- Execution time calculates the physical microwave duration to exclude the 2.0s IBM Runtime metric overhead.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def normalize_state(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=complex)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Input state must be non-zero.")
    return vector / norm


@dataclass(frozen=True)
class HHLExample:
    matrix: np.ndarray
    b: np.ndarray
    t: float
    c_scale: float = 2 / 3


def paper_example() -> HHLExample:
    matrix = np.array([[1.0, -1.0 / 3.0], [-1.0 / 3.0, 1.0]], dtype=complex)
    b = np.array([0.0, 1.0], dtype=complex)
    return HHLExample(matrix=matrix, b=b, t=3.0 * np.pi / 4.0)


def example_unitaries(example: HHLExample):
    # Instead of an arbitrary UnitaryGate (which transpiles exponentially),
    # we define the matrix as a sparse Pauli Operator.
    # The paper's matrix [[1, -1/3], [-1/3, 1]] decomposes exactly to: I - 1/3*X
    hamiltonian = SparsePauliOp.from_list([("I", 1.0), ("X", -1.0 / 3.0)])
    
    # PauliEvolutionGate simulates e^(iHt) algorithmically, enabling true O(log N) depth
    unitary = PauliEvolutionGate(hamiltonian, time=example.t)
    unitary_squared = PauliEvolutionGate(hamiltonian, time=2.0 * example.t)
    
    return unitary, unitary_squared


def initialize_registers():
    system = QuantumRegister(1, "sys")
    phase = QuantumRegister(2, "phase")
    ancilla = QuantumRegister(1, "anc")
    classical = ClassicalRegister(2, "c")
    circuit = QuantumCircuit(system, phase, ancilla, classical, name="hhl_walkthrough")
    return circuit, system, phase, ancilla, classical


def prepare_b_state(circuit: QuantumCircuit, system: QuantumRegister, b: np.ndarray) -> None:
    circuit.initialize(normalize_state(b), system[0])


def append_controlled_unitary(
    circuit: QuantumCircuit,
    control,
    target,
    unitary: PauliEvolutionGate,
    label: str,
) -> None:
    c_unitary = unitary.control(1)
    c_unitary.name = label
    circuit.append(c_unitary, [control, target])


def qpe_stage(
    circuit: QuantumCircuit,
    system,
    phase: QuantumRegister,
    unitary: PauliEvolutionGate,
    unitary_squared: PauliEvolutionGate,
) -> None:
    circuit.h(phase[1])
    circuit.h(phase[0])
    append_controlled_unitary(circuit, phase[1], system[0], unitary_squared, "U2")
    append_controlled_unitary(circuit, phase[0], system[0], unitary, "U")

    circuit.swap(phase[0], phase[1])
    circuit.h(phase[0])
    circuit.cp(-np.pi / 2, phase[1], phase[0])
    circuit.h(phase[1])


def inverse_qpe_stage(
    circuit: QuantumCircuit,
    system,
    phase: QuantumRegister,
    unitary: PauliEvolutionGate,
    unitary_squared: PauliEvolutionGate,
) -> None:
    circuit.h(phase[1])
    circuit.cp(np.pi / 2, phase[1], phase[0])
    circuit.h(phase[0])
    circuit.swap(phase[0], phase[1])

    append_controlled_unitary(circuit, phase[0], system[0], unitary.inverse(), "U^-1")
    append_controlled_unitary(circuit, phase[1], system[0], unitary_squared.inverse(), "U2^-1")

    circuit.h(phase[0])
    circuit.h(phase[1])


def ancilla_rotation_stage(
    circuit: QuantumCircuit,
    phase: QuantumRegister,
    ancilla,
    c_scale: float,
) -> None:
    rotations = {
        "01": 2.0 * np.arcsin(np.clip(c_scale / 1.0, -1.0, 1.0)),
        "10": 2.0 * np.arcsin(np.clip(c_scale / 2.0, -1.0, 1.0)),
    }

    for bitstring, angle in rotations.items():
        if bitstring == "01":
            circuit.x(phase[1])
        elif bitstring == "10":
            circuit.x(phase[0])

        circuit.append(RYGate(angle).control(2), [phase[0], phase[1], ancilla[0]])

        if bitstring == "01":
            circuit.x(phase[1])
        elif bitstring == "10":
            circuit.x(phase[0])


def build_hhl_circuit() -> QuantumCircuit:
    example = paper_example()
    unitary, unitary_squared = example_unitaries(example)
    circuit, system, phase, ancilla, classical = initialize_registers()

    prepare_b_state(circuit, system, example.b)
    qpe_stage(circuit, system, phase, unitary, unitary_squared)
    ancilla_rotation_stage(circuit, phase, ancilla, example.c_scale)
    inverse_qpe_stage(circuit, system, phase, unitary, unitary_squared)

    circuit.measure(ancilla[0], classical[0])
    circuit.measure(system[0], classical[1])
    return circuit


def generate_spd_matrix(size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(size, size))
    spd = mat.T @ mat + size * np.eye(size)
    return spd


def classical_benchmark(sizes: Iterable[int]) -> dict[int, float]:
    results: dict[int, float] = {}
    for size in sizes:
        a = generate_spd_matrix(size, seed=42)
        b = np.ones(size)
        start = time.perf_counter()
        _ = np.linalg.solve(a, b)
        results[size] = time.perf_counter() - start
    return results


def select_backend(service: QiskitRuntimeService, backend_name: str | None) -> str:
    if backend_name:
        return backend_name
    backend = service.least_busy(simulator=False, operational=True, min_num_qubits=4)
    return backend.name


def extract_execution_seconds(job) -> float | None:
    try:
        metrics = job.metrics()
    except Exception:
        return None

    usage = metrics.get("usage")
    if isinstance(usage, dict) and "seconds" in usage:
        return float(usage["seconds"])

    for key in ("execution_time", "estimated_execution_time", "duration"):
        value = metrics.get(key)
        if value is not None:
            return float(value)

    return None


def calculate_qpu_time(backend, isa_circuit, shots: int) -> float | None:
    """Calculates the physical microwave execution time without API overhead."""
    try:
        # Schedule the circuit to precise hardware clock cycles ("alap" scheduling)
        timing_pm = generate_preset_pass_manager(
            target=backend.target,
            optimization_level=0,
            scheduling_method="alap"
        )
        scheduled_circuit = timing_pm.run(isa_circuit)
        
        duration_dt = scheduled_circuit.duration
        if duration_dt is None or backend.dt is None:
            return None
            
        single_shot_seconds = duration_dt * backend.dt
        
        # Pull backend rep_delay (default to 250us if not found)
        rep_delay = 250e-6
        if hasattr(backend, "options") and hasattr(backend.options, "rep_delay"):
            rep_delay = backend.options.rep_delay
            
        total_time = (single_shot_seconds + rep_delay) * shots
        return total_time
    except Exception:
        return None


def quantum_benchmark(backend_name: str | None, shots: int) -> tuple[float, float | None, float | None]:
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ.get("QISKIT_IBM_TOKEN"),
        instance=os.environ.get("QISKIT_IBM_INSTANCE"),
    )
    backend_name = select_backend(service, backend_name)
    backend = service.backend(backend_name)

    circuit = build_hhl_circuit()
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pass_manager.run(circuit)

    # Calculate exact physical time
    qpu_seconds = calculate_qpu_time(backend, isa_circuit, shots)

    # V2 Sampler setup
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots

    start_wall = time.perf_counter()
    job = sampler.run([isa_circuit])
    _ = job.result()
    wall_seconds = time.perf_counter() - start_wall

    exec_seconds = extract_execution_seconds(job)
    return wall_seconds, exec_seconds, qpu_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classical solve vs. HHL runtime on IBM Quantum")
    parser.add_argument("--backend", default=None, help="IBM Quantum backend name (default: least busy)")
    parser.add_argument("--shots", type=int, default=1024, help="Shots for the quantum run")
    parser.add_argument("--sizes", default="2,4,8", help="Comma-separated classical sizes")
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    classical = classical_benchmark(sizes)

    print("Classical runtimes (NumPy linalg.solve):")
    for size in sizes:
        print(f"  {size}x{size}: {classical[size]:.6f} s")

    print("\nQuantum runtime (HHL 2x2 walkthrough):")
    wall_seconds, exec_seconds, qpu_seconds = quantum_benchmark(args.backend, args.shots)
    
    if exec_seconds is not None:
        print(f"  Execution time (IBM API metric w/ overhead): {exec_seconds:.6f} s")
    
    if qpu_seconds is not None:
        print(f"  Calculated pure QPU time (gates + reset):    {qpu_seconds:.6f} s")
    else:
        print("  Calculated pure QPU time:                    Unavailable")
        
    print(f"  Wall time (includes network queue):          {wall_seconds:.6f} s")


if __name__ == "__main__":
    main()