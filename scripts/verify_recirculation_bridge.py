"""Finite-dimensional checks for research/recirculation_generalization.md.

No dataset, training entry point, model class, or W&B run is created. Matrices
represent frozen Jacobians. All results concern linear algebra, not convergence
of a learned neural network. Run with the repository's NumPy environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-15))


def cosine(a, b):
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / scale) if scale else float('nan')


def rank(a):
    return int(np.linalg.matrix_rank(a, tol=1e-10))


def checks(seed=20261021):
    rng = np.random.default_rng(seed)
    # Supervised, non-square forward map: R^8 -> R^6 -> R^5 -> R^3.
    a1 = rng.normal(size=(6, 8)) / np.sqrt(8)
    a2 = rng.normal(size=(5, 6)) / np.sqrt(6)
    a3 = rng.normal(size=(3, 5)) / np.sqrt(5)
    prefixes = [a1, a2 @ a1]
    suffixes = [a3 @ a2, a3]
    forward = a3 @ a2 @ a1
    errors = rng.normal(size=(3, 64))
    bridge = forward.T
    results = {}

    # Raw adjoint-bridge recirculation is a positive-semidefinite
    # preconditioning of each hidden BP signal in a linear independent network.
    oracle = []
    for prefix, suffix in zip(prefixes, suffixes):
        bp = suffix.T @ errors
        recirc = prefix @ bridge @ errors
        gram = prefix @ prefix.T
        np.testing.assert_allclose(recirc, gram @ bp, atol=1e-12)
        corrected = np.linalg.solve(gram, recirc)
        np.testing.assert_allclose(corrected, bp, atol=1e-10)
        assert np.sum(bp * recirc) > 0
        wrong = prefix @ (-bridge) @ errors
        assert rank(-bridge) == 3 and np.sum(bp * wrong) < 0
        oracle.append(dict(bp_rank=rank(bp), recirculated_rank=rank(recirc),
                           raw_cosine=cosine(bp, recirc),
                           corrected_relative_error=relative_error(corrected, bp),
                           full_rank_wrong_sign_cosine=cosine(bp, wrong)))
    results['adjoint_bridge'] = oracle

    # The same scalar alignment numerator is shared by every hidden layer.
    random_bridge = rng.normal(size=(8, 3))
    per_layer_inner = [float(np.sum((q.T @ errors) * (p @ random_bridge @ errors)))
                       for p, q in zip(prefixes, suffixes)]
    loop_inner = float(np.sum(errors * (forward @ random_bridge @ errors)))
    np.testing.assert_allclose(per_layer_inner, loop_inner, atol=1e-12)
    results['shared_alignment_numerator'] = dict(per_layer=per_layer_inner, loop=loop_inner)

    # Forward probes construct a right inverse without reading a Jacobian
    # transpose: F (Xi (F Xi)^dagger) = I if F Xi has full row rank.
    right_probes = rng.normal(size=(8, 4))
    right_responses = forward @ right_probes
    right_bridge = right_probes @ np.linalg.pinv(right_responses)
    np.testing.assert_allclose(forward @ right_bridge, np.eye(3), atol=1e-12)
    right_stats = []
    for p, q in zip(prefixes, suffixes):
        bp = q.T @ errors
        signal = p @ right_bridge @ errors
        np.testing.assert_allclose(np.sum(bp * signal), np.sum(errors**2), atol=1e-10)
        right_stats.append(dict(cosine=cosine(bp, signal), signal_rank=rank(signal)))
    results['forward_probe_right_inverse'] = dict(probes=4,
        loop_identity_relative_error=relative_error(forward @ right_bridge, np.eye(3)),
        hidden_stats=right_stats)

    # Confirm that a small simultaneous parameter update lowers squared loss.
    x = rng.normal(size=(8, 1))
    target = rng.normal(size=(3, 1))
    values = [x, a1 @ x, a2 @ a1 @ x, forward @ x]
    residual = target - values[-1]
    signals = [p @ right_bridge @ residual for p in prefixes] + [residual]
    updates = [v @ h.T for v, h in zip(signals, values[:-1])]
    bp_signals = [q.T @ residual for q in suffixes] + [residual]
    true_gradients = [-v @ h.T for v, h in zip(bp_signals, values[:-1])]
    directional_derivative = sum(float(np.sum(g * d)) for g, d in zip(true_gradients, updates))
    assert directional_derivative < 0
    step = 1e-7 / max(1.0, sum(np.linalg.norm(d) for d in updates))
    changed = [a + step*d for a, d in zip((a1, a2, a3), updates)]
    before = .5 * float(np.sum(residual**2))
    after = .5 * float(np.sum((target-changed[2] @ changed[1] @ changed[0] @ x)**2))
    assert after < before
    results['independent_parameter_descent'] = dict(directional_derivative=directional_derivative,
        before_loss=before, after_loss=after, step=step)

    # Activation alignment need not survive a noncommuting local parameter metric.
    b = np.array([1.0, -.1])
    c = np.array([[1.0, 2.0], [2.0, 5.0]])
    local_metric = np.diag([.01, 1.0])
    assert np.linalg.eigvalsh(c).min() > 0
    results['nonlinear_parameter_metric_caution'] = dict(
        activation_inner_product=float(b @ c @ b),
        parameter_inner_product=float(b @ local_metric @ c @ b))
    assert results['nonlinear_parameter_metric_caution']['activation_inner_product'] > 0
    assert results['nonlinear_parameter_metric_caution']['parameter_inner_product'] < 0

    # A task subspace needs a right inverse only in its output coordinates.
    task_basis = np.linalg.qr(rng.normal(size=(3, 2)))[0]
    task_probes = rng.normal(size=(8, 2))
    reduced_responses = task_basis.T @ forward @ task_probes
    task_bridge = task_probes @ np.linalg.pinv(reduced_responses) @ task_basis.T
    np.testing.assert_allclose(task_basis.T @ forward @ task_bridge @ task_basis,
                               np.eye(2), atol=1e-12)
    task_errors = task_basis @ rng.normal(size=(2, 64))
    for p, q in zip(prefixes, suffixes):
        np.testing.assert_allclose(np.sum((q.T @ task_errors) * (p @ task_bridge @ task_errors)),
                                  np.sum(task_errors**2), atol=1e-9)
    results['task_subspace_right_inverse'] = dict(task_rank=2, probes=2,
        bridge_rank=rank(task_bridge), projected_loop_relative_error=relative_error(
            task_basis.T @ forward @ task_bridge @ task_basis, np.eye(2)))

    # A common raw input bridge generically cannot equal BP at every layer.
    stacked_p = np.vstack(prefixes)
    stacked_b = np.vstack([q.T for q in suffixes])
    best_raw = np.linalg.pinv(stacked_p) @ stacked_b
    residual = relative_error(stacked_p @ best_raw, stacked_b)
    assert residual > 1e-3
    results['common_raw_bridge'] = dict(least_squares_relative_error=residual,
                                       stacked_prefix_rank=rank(stacked_p))

    # High cosine need not preserve the task's three independent error modes.
    bp = np.diag([10.0, 1.0, 1.0])
    lowrank = np.diag([10.0, 0.0, 0.0])
    assert cosine(bp, lowrank) > .99 and rank(lowrank) == 1
    results['cosine_hides_rank_loss'] = dict(cosine=cosine(bp, lowrank),
                                            bp_rank=rank(bp), signal_rank=rank(lowrank))

    # A wide hidden layer has BP components unreachable from the input.
    prefix = np.vstack([np.eye(2), np.zeros((3, 2))])  # 5 x 2
    suffix = rng.normal(size=(3, 5)) / np.sqrt(5)
    forward_wide = suffix @ prefix
    projection = prefix @ np.linalg.pinv(prefix)
    bp_map = suffix.T
    unreachable = (np.eye(5) - projection) @ bp_map
    corrected_map = np.linalg.pinv(prefix @ prefix.T) @ prefix @ forward_wide.T
    np.testing.assert_allclose(corrected_map, projection @ bp_map, atol=1e-12)
    np.testing.assert_allclose(corrected_map + unreachable, bp_map, atol=1e-12)
    u, singular, vt = np.linalg.svd(unreachable, full_matrices=False)
    repaired_rank2 = corrected_map + (u[:, :2] * singular[:2]) @ vt[:2]
    assert rank(unreachable) == 3
    np.testing.assert_allclose(relative_error(repaired_rank2, bp_map),
                               singular[2] / np.linalg.norm(bp_map), atol=1e-12)
    results['minimal_repair'] = dict(bp_map_rank=rank(bp_map),
        reachable_map_rank=rank(corrected_map), required_bypass_rank=rank(unreachable),
        rank2_repair_relative_error=relative_error(repaired_rank2, bp_map),
        rank3_repair_relative_error=relative_error(corrected_map+unreachable, bp_map))

    # Matched forward probes cancel their empirical covariance exactly.
    # Even non-isotropic finite probes need not estimate J^T accurately to
    # recover the BP signals, provided hidden probe matrices have full row rank.
    probes = rng.normal(size=(8, 7))
    responses = forward @ probes
    empirical_bridge = probes @ responses.T / probes.shape[1]
    matched = []
    for prefix, suffix in zip(prefixes, suffixes):
        d = prefix @ probes
        covariance = d @ d.T / probes.shape[1]
        signal = prefix @ empirical_bridge @ errors
        corrected = np.linalg.solve(covariance, signal)
        bp = suffix.T @ errors
        np.testing.assert_allclose(corrected, bp, atol=1e-9)
        matched.append(relative_error(corrected, bp))
    results['matched_probes'] = dict(probe_count=7,
        bridge_relative_error_to_exact_adjoint=relative_error(empirical_bridge, bridge),
        corrected_relative_errors=matched)

    # Fewer probes return the projection into their reachable hidden subspace.
    sketch = []
    for k in (1, 2, 3, 4, 5, 6):
        probes = rng.normal(size=(8, k))
        responses = forward @ probes
        d = prefixes[0] @ probes
        gram = d @ d.T / k
        s = d @ responses.T @ errors / k
        corrected = np.linalg.pinv(gram) @ s
        bp = suffixes[0].T @ errors
        expected = d @ np.linalg.pinv(d) @ bp
        np.testing.assert_allclose(corrected, expected, atol=1e-9)
        sketch.append(dict(probes=k, reachable_rank=rank(d),
            projected_signal_rank=rank(corrected), bp_relative_error=relative_error(corrected, bp)))
    results['probe_rank_sweep'] = sketch

    # A fixed rank-deficient input bridge cannot recover its discarded modes
    # merely by repeated residual correction with frozen forward weights.
    r = rng.normal(size=(8, 1)) @ rng.normal(size=(1, 3))
    transform = np.eye(3) - .01 * forward @ r
    blocks = [prefixes[0] @ r @ np.linalg.matrix_power(transform, k) for k in range(8)]
    temporal_rank = rank(np.vstack(blocks))
    assert temporal_rank == 1
    results['fixed_loop_cannot_restore_lost_rank'] = dict(bridge_rank=rank(r),
        cycles=8, stacked_temporal_signal_rank=temporal_rank)

    # Rank and numerical conditioning differ, including for a raw signal whose
    # norm ratio and cosine are almost one because weak modes carry little energy.
    operator = np.diag([1.0, 1.0, 1e-8])
    results['rank_vs_conditioning'] = dict(algebraic_rank=int(np.linalg.matrix_rank(operator)),
        condition_number=float(np.linalg.cond(operator)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    results = checks(args.seed)
    rendered = json.dumps(results, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
