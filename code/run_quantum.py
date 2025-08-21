#!/usr/bin/env python
import argparse, sys
from pathlib import Path

def run_encoding(sequence="ATCG", outdir="images"):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit.visualization import circuit_drawer
    except Exception as e:
        print("Qiskit not installed. Please: pip install qiskit qiskit-aer")
        sys.exit(1)

    dna_map = {'A':'00','T':'01','C':'10','G':'11'}
    qc = QuantumCircuit(len(sequence), name="DNA_Encode")
    for i, base in enumerate(sequence):
        bits = dna_map.get(base, '00')
        if bits[1] == '1':
            qc.x(i)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    try:
        fig = circuit_drawer(qc, output='mpl')
        fig.savefig(Path(outdir)/'quantum_encoding_circuit.png', dpi=200, bbox_inches='tight')
        print(f"Saved circuit to {Path(outdir)/'quantum_encoding_circuit.png'}")
    except Exception as e:
        print("Could not render circuit diagram (matplotlib backend issue). Proceeding.")

    sim = AerSimulator()
    qc.save_statevector()
    result = sim.run(qc).result()
    sv = result.get_statevector()
    print("Qubits:", qc.num_qubits)
    print("Statevector (first 8 amps):", sv[:8])

def run_grover(n=3, outdir="images"):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer.primitives import Sampler
        from qiskit_algorithms import Grover, AmplificationProblem
    except Exception as e:
        print("Missing qiskit or qiskit-algorithms. Install: pip install qiskit qiskit-aer qiskit-algorithms")
        sys.exit(1)

    oracle = QuantumCircuit(n, name='oracle')
    # Mark |111> as solution (toy oracle): use multi-controlled Z via three CZs (approx toy)
    oracle.cz(0,1)
    oracle.cz(1,2)
    oracle.cz(0,2)
    problem = AmplificationProblem(oracle.to_gate())
    grover = Grover(sampler=Sampler())
    result = grover.solve(problem)
    print("Top measurement (candidate mutation state):", result.top_measurement)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quantum DNA encoding & Grover demo")
    p.add_argument("--sequence", type=str, default="ATCG", help="DNA sequence to encode")
    p.add_argument("--grover", action="store_true", help="Run Grover demo instead of encoding")
    p.add_argument("--n", type=int, default=3, help="Grover qubits if --grover")
    p.add_argument("--outdir", type=str, default="images", help="Output directory for figures")
    args = p.parse_args()
    if args.grover:
        run_grover(n=args.n, outdir=args.outdir)
    else:
        run_encoding(sequence=args.sequence, outdir=args.outdir)
