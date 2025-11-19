import qiskit
print(qiskit.version.get_version_info())

# Tested with: qiskit==2.2.1
import numpy as np
import re
from typing import Sequence

from qiskit.quantum_info import Kraus, SuperOp, DensityMatrix

# ------------------------
# 1) Build V/C sequence & transition matrix from text
# ------------------------
def vc_sequence(text: str, include_y_as_vowel: bool = False) -> Sequence[str]:
    vowels = set("aeiou" + ("y" if include_y_as_vowel else ""))
    seq = []
    for ch in re.findall(r"[a-zA-Z]", text):
        seq.append('V' if ch.lower() in vowels else 'C')
    return seq

def transition_matrix(seq: Sequence[str]) -> np.ndarray:
    """
    Returns a 2x2 row-stochastic matrix P with states [V=0, C=1]:
    P[i,j] = P(state_i -> state_j)
    """
    if len(seq) < 2:
        raise ValueError("Sequence too short to compute transitions.")
    idx = {'V': 0, 'C': 1}
    counts = np.zeros((2, 2), dtype=float)
    for a, b in zip(seq, seq[1:]):
        counts[idx[a], idx[b]] += 1.0

    # Handle any zero rows robustly (e.g., pathological tiny inputs)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

    # If a row had no outgoing transitions, default to uniform for that row
    for r in range(2):
        if np.isclose(P[r].sum(), 0.0):
            P[r] = np.array([0.5, 0.5])
    return P

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalized to sum to 1."""
    w, v = np.linalg.eig(P.T)
    i = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, i])
    pi = np.maximum(pi, 0)  # guard tiny negative noise
    s = pi.sum()
    return pi / s if s > 0 else np.array([0.5, 0.5])

# ------------------------
# 2) Convert 2x2 Markov chain to a Kraus channel and simulate
# ------------------------
def markov_to_kraus(P: np.ndarray) -> Kraus:
    """
    Map classical 2-state Markov chain P to a CPTP Kraus channel on 1 qubit
    with computational basis |0>=V, |1>=C.
    """
    p_VV, p_VC = float(P[0, 0]), float(P[0, 1])
    p_CV, p_CC = float(P[1, 0]), float(P[1, 1])

    # Kraus operators (ensure complex dtype)
    K_VV = np.array([[np.sqrt(p_VV), 0.0],
                     [0.0,            0.0]], dtype=complex)
    K_VC = np.array([[0.0,            0.0],
                     [np.sqrt(p_VC),  0.0]], dtype=complex)
    K_CV = np.array([[0.0,            np.sqrt(p_CV)],
                     [0.0,            0.0]], dtype=complex)
    K_CC = np.array([[0.0,            0.0],
                     [0.0,            np.sqrt(p_CC)]], dtype=complex)

    return Kraus([K_VV, K_VC, K_CV, K_CC])

def evolve_markov_channel(P, steps: int, start_state: str = 'V') -> dict:
    """
    Repeatedly apply the channel to a 1-qubit density matrix.
    Returns probabilities {'0': Pr(V), '1': Pr(C)} after 'steps' iterations.
    """
    channel = markov_to_kraus(P)
    superop = SuperOp(channel)  # QuantumChannel

    # Start in |0> (V) or |1> (C)
    rho = DensityMatrix.from_label('0' if start_state.upper() == 'V' else '1')

    for _ in range(steps):
        rho = rho.evolve(superop)  # apply channel to the state

    # Now rho is still a DensityMatrix, so this works:
    probs = rho.probabilities_dict()  # {'0': pV, '1': pC}
    return {k: float(v) for k, v in probs.items()}

# ------------------------
# 3) Example usage
# ------------------------
if __name__ == "__main__":
    text = "To be, or not to be, that is the question."  # Replace with a longer corpus for stability
    # text = "A E I o"
    seq = vc_sequence(text, include_y_as_vowel=False)
    P = transition_matrix(seq)

    print("Transition matrix P (rows: from V/C, cols: to V/C):\n", np.round(P, 3))
    pi = stationary_distribution(P)
    print("Stationary distribution [V, C]:", np.round(pi, 3))

    out = evolve_markov_channel(P, steps=25, start_state='V')
    print("Distribution after 25 steps:", {k: round(v, 3) for k, v in out.items()})