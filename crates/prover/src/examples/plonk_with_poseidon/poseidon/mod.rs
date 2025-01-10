use std::ops::{Add, AddAssign, Mul, Sub};
use num_traits::One;
use crate::constraint_framework::{EvalAtRow, RelationEntry, PREPROCESSED_TRACE_IDX};
use crate::core::fields::FieldExpOps;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::vcs::poseidon31_ref::{FIRST_FOUR_ROUND_RC, LAST_FOUR_ROUNDS_RC, PARTIAL_ROUNDS_RC};
use crate::examples::poseidon::PoseidonElements;
use crate::examples::xor::gkr_lookups::accumulation::DynMle::Base;

const N_STATE: usize = 16;
const N_HALF_FULL_ROUNDS: usize = 4;
const N_PARTIAL_ROUNDS: usize = 14;

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t12.clone() + t12.clone() + t3.clone();
    let t5 = t02.clone() + t02.clone() + t2.clone();
    let t6 = t3.clone() + t5.clone();
    let t7 = t2.clone() + t4.clone();
    [t6, t5, t7, t4]
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i].clone(),
            state[4 * i + 1].clone(),
            state[4 * i + 2].clone(),
            state[4 * i + 3].clone(),
        ]);
    }
    for j in 0..4 {
        let s =
            state[j].clone() + state[j + 4].clone() + state[j + 8].clone() + state[j + 12].clone();
        for i in 0..4 {
            state[4 * i + j] += s.clone();
        }
    }
}


// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);

    state[0] += state[0].clone() + state[0].clone() + sum.clone();
    state.iter_mut().enumerate().skip(1).for_each(|(i, s)| {
        // TODO(andrew): Change to rotations.
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

pub fn eval_poseidon_constraints<E: EvalAtRow>(eval: &mut E, lookup_elements: &PoseidonElements) {
    let addr_1 = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
    let addr_2 = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
    let addr_3 = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
    let addr_4 = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);

    let sel_1 = eval.next_trace_mask();
    let sel_2 = eval.next_trace_mask();
    let sel_3 = eval.next_trace_mask();
    let sel_4 = eval.next_trace_mask();

    let mut state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

    // Require state lookup.
    let initial_state = state.clone();

    // 4 full rounds.
    (0..N_HALF_FULL_ROUNDS).for_each(|round| {
        (0..N_STATE).for_each(|i| {
            state[i] += FIRST_FOUR_ROUND_RC[round][i];
        });
        state = std::array::from_fn(|i| pow5(state[i].clone()));
        apply_external_round_matrix(&mut state);
        state.iter_mut().for_each(|s| {
            let m = eval.next_trace_mask();
            eval.add_constraint(s.clone() - m.clone());
            *s = m;
        });
    });

    // Partial rounds.
    (0..N_PARTIAL_ROUNDS).for_each(|round| {
        state[0] += PARTIAL_ROUNDS_RC[round];
        state[0] = pow5(state[0].clone());
        apply_internal_round_matrix(&mut state);
        let m = eval.next_trace_mask();
        eval.add_constraint(state[0].clone() - m.clone());
        state[0] = m;
    });

    // 4 full rounds.
    (0..N_HALF_FULL_ROUNDS).for_each(|round| {
        (0..N_STATE).for_each(|i| {
            state[i] += LAST_FOUR_ROUNDS_RC[round][i];
        });
        state = std::array::from_fn(|i| pow5(state[i].clone()));
        apply_external_round_matrix(&mut state);
        state.iter_mut().for_each(|s| {
            let m = eval.next_trace_mask();
            eval.add_constraint(s.clone() - m.clone());
            *s = m;
        });
    });

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_1, sel_1.clone()]
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_2, sel_2.clone()]
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_3, sel_3.clone()]
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_4, sel_4.clone()]
    ));

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[sel_1.clone() + BaseField::from(i), initial_state[i].clone()]
        ));
    }

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[sel_2.clone() + BaseField::from(i), initial_state[i + 8].clone()]
        ))
    }

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[sel_3.clone() + BaseField::from(i), state[i].clone()]
        ))
    }

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[sel_4.clone() + BaseField::from(i), state[i + 8].clone()]
        ))
    }

    // TODO: use higher degrees batching
    eval.finalize_logup_in_pairs();
}
