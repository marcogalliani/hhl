from hhl_2x2_sketch import *

if __name__ == "__main__":
    # Define the linear system
    A = np.array([[1.0, -1.0 / 3.0], [-1.0 / 3.0, 1.0]])
    b_vec = np.array([1.0, 0.0])
    
    # Classical reference
    x_cl = np.linalg.solve(A, b_vec)
    x_cl_norm = x_cl / np.linalg.norm(x_cl)

    #""""
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["QISKIT_IBM_TOKEN"]
    )
    backend = service.least_busy(operational=True, min_num_qubits=4)
    solver = HHLAlgorithm(backend=backend)
    #"""
    # solver = HHLAlgorithm()
    print(solver.build_circuit(A, b_vec).draw(output="text"))

    x_hhl, p_success = solver.solve(A, b_vec)
    
    # ---
    # Solution comparison
    # ---
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

    # ---
    # Observable x^T M x
    # ---
    """
    print("\n=== Observable ⟨x|M|x⟩ ===")
    M = np.array([[2.0, 0.5], [0.5, 1.0]])
    print(f"  M = {M.tolist()}")
    
    obs_quantum = solver.compute_observable(M, A, b_vec, shots=5000)
    obs_classical = float(x_cl_norm.T @ M @ x_cl_norm)
    print(f"  Classical ⟨x_cl|M|x_cl⟩  : {obs_classical:.6f}")
    print(f"  Quantum   ⟨x_hhl|M|x_hhl⟩: {obs_quantum:.6f}")
    print(f"  Difference                : {abs(obs_classical - obs_quantum):.2e}")
    """

    # ---
    # Shot-based simulation
    # ---
    def _print_sim(label: str, sim: SimulationResult) -> None:
        print(f"\n=== {label} ===")
        print(f"  {'outcome (sys anc)':<20} {'P(outcome)':>10}  {'P(outcome|anc=1)':>16}")
        for outcome, prob in sim.raw.items():
            cond = f"{sim.conditional[outcome]:>16.4f}" if outcome in sim.conditional else f"{'—':>16}"
            print(f"  {outcome:<20} {prob:>10.4f}  {cond}")

    _print_sim(f"Simulations ({solver.backend or 'local'})", solver.simulate(A, b_vec, shots=5000))