use std::collections::HashMap;
use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
use num_traits::One;
use tracing::{span, Level};

use crate::constraint_framework::{EvalAtRow, RelationEntry, PREPROCESSED_TRACE_IDX};
use crate::core::backend::simd::m31::{PackedBaseField, PackedM31, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::vcs::poseidon31_ref::{
    FIRST_FOUR_ROUND_RC, LAST_FOUR_ROUNDS_RC, MAT_DIAG16_M_1, PARTIAL_ROUNDS_RC,
};
use crate::core::ColumnVec;
use crate::examples::plonk::PlonkLookupElements;

const N_STATE: usize = 16;
const N_HALF_FULL_ROUNDS: usize = 4;
const N_PARTIAL_ROUNDS: usize = 14;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
const N_COLUMNS: usize = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS + 4;

pub const CONSTANT_1: [M31; 8] = [M31::from_u32_unchecked(0); 8];
pub const CONSTANT_2: [M31; 8] = [
    M31::from_u32_unchecked(0x7412ba68),
    M31::from_u32_unchecked(0x70d9f6f6),
    M31::from_u32_unchecked(0x045f05e1),
    M31::from_u32_unchecked(0x056a10e6),
    M31::from_u32_unchecked(0x4aeaffae),
    M31::from_u32_unchecked(0x6b01895d),
    M31::from_u32_unchecked(0x20f34b05),
    M31::from_u32_unchecked(0x7d6e4f58),
];
pub const CONSTANT_3: [M31; 8] = [
    M31::from_u32_unchecked(0x16d7f425),
    M31::from_u32_unchecked(0x5bed6a76),
    M31::from_u32_unchecked(0x50fb5a7d),
    M31::from_u32_unchecked(0x4f86ae48),
    M31::from_u32_unchecked(0x72d1a80d),
    M31::from_u32_unchecked(0x710419a6),
    M31::from_u32_unchecked(0x28679230),
    M31::from_u32_unchecked(0x249e2073),
];

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

pub fn eval_poseidon_constraints<E: EvalAtRow>(
    eval: &mut E,
    lookup_elements: &PlonkLookupElements,
) {
    let [addr_1] = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
    let [addr_2] = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
    let [addr_3] = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
    let [addr_4] = eval.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);

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
    (0..N_HALF_FULL_ROUNDS - 1).for_each(|round| {
        (0..N_STATE).for_each(|i| {
            state[i] += LAST_FOUR_ROUNDS_RC[round][i];
        });
        state = std::array::from_fn(|i| pow5(state[i].clone()));
        apply_external_round_matrix(&mut state);
        state.iter_mut().for_each(|s| {
            let m = eval.next_trace_mask();
            eval.add_constraint(s.clone() - m.clone());
            *s = m;
        })
    });

    // 4 full rounds.
    {
        (0..N_STATE).for_each(|i| {
            state[i] += LAST_FOUR_ROUNDS_RC[3][i];
        });
        state = std::array::from_fn(|i| pow5(state[i].clone()));
        apply_external_round_matrix(&mut state);
        state
            .iter_mut()
            .zip(initial_state.iter())
            .take(8)
            .for_each(|(s, i)| {
                let m = eval.next_trace_mask();
                eval.add_constraint(m.clone() - s.clone() - i.clone());
                *s = m;
            });
        state.iter_mut().skip(8).for_each(|s| {
            let m = eval.next_trace_mask();
            eval.add_constraint(s.clone() - m.clone());
            *s = m;
        })
    }

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_1, sel_1.clone()],
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_2, sel_2.clone()],
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_3, sel_3.clone()],
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[addr_4, sel_4.clone()],
    ));

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[
                sel_1.clone() + E::F::from(BaseField::from(i).into()),
                initial_state[i].clone(),
            ],
        ));
    }

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[
                sel_2.clone() + E::F::from(BaseField::from(i).into()),
                initial_state[i + 8].clone(),
            ],
        ))
    }

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[
                sel_3.clone() + E::F::from(BaseField::from(i).into()),
                state[i].clone(),
            ],
        ))
    }

    for i in 0..8 {
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &[
                sel_4.clone() + E::F::from(BaseField::from(i).into()),
                state[i + 8].clone(),
            ],
        ))
    }

    // TODO: use higher degrees batching
    eval.finalize_logup_in_pairs();
}

pub struct PoseidonDataFlow(pub HashMap<usize, [M31; 8]>);

pub struct PoseidonControlFlow {
    pub sel_1: Vec<usize>,
    pub sel_2: Vec<usize>,
    pub sel_3: Vec<usize>,
    pub sel_4: Vec<usize>,
}

pub struct PoseidonPrescribedFlow {
    pub addr_1: Vec<usize>,
    pub addr_2: Vec<usize>,
    pub addr_3: Vec<usize>,
    pub addr_4: Vec<usize>,
}

pub struct PoseidonMetadata {
    pub prescribed_flow: PoseidonPrescribedFlow,
    pub control_flow: PoseidonControlFlow,
    pub data_flow: PoseidonDataFlow,
    pub constant_1_addr: usize,
    pub constant_1_sel: usize,
    pub constant_2_addr: usize,
    pub constant_2_sel: usize,
    pub constant_3_addr: usize,
    pub constant_3_sel: usize,
}

pub fn gen_trace(
    metadata: &mut PoseidonMetadata,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let prescribed_flow = &mut metadata.prescribed_flow;
    let control_flow = &mut metadata.control_flow;
    let data_flow = &mut metadata.data_flow;

    // check the length
    let len = prescribed_flow.addr_1.len();
    assert_eq!(len, prescribed_flow.addr_2.len());
    assert_eq!(len, prescribed_flow.addr_3.len());
    assert_eq!(len, prescribed_flow.addr_4.len());
    assert_eq!(len, control_flow.sel_1.len());
    assert_eq!(len, control_flow.sel_2.len());
    assert_eq!(len, control_flow.sel_3.len());
    assert_eq!(len, control_flow.sel_4.len());

    // check the constants are given
    assert!(data_flow.0.contains_key(&metadata.constant_1_sel));
    assert!(data_flow.0.contains_key(&metadata.constant_2_sel));
    assert!(data_flow.0.contains_key(&metadata.constant_3_sel));

    // compute the circuit size
    let log_size = len.next_power_of_two().ilog2();

    prescribed_flow
        .addr_1
        .resize(1 << log_size, metadata.constant_1_addr);
    prescribed_flow
        .addr_2
        .resize(1 << log_size, metadata.constant_1_addr);
    prescribed_flow
        .addr_3
        .resize(1 << log_size, metadata.constant_2_addr);
    prescribed_flow
        .addr_4
        .resize(1 << log_size, metadata.constant_3_addr);

    control_flow
        .sel_1
        .resize(1 << log_size, metadata.constant_1_sel);
    control_flow
        .sel_2
        .resize(1 << log_size, metadata.constant_1_sel);
    control_flow
        .sel_3
        .resize(1 << log_size, metadata.constant_2_sel);
    control_flow
        .sel_4
        .resize(1 << log_size, metadata.constant_3_addr);

    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);

    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();

    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        let mut col_index = 0;

        // Fill sel_1, sel_2, sel_3, sel_4
        trace[col_index].data[vec_index] = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_1[vec_index * N_LANES + i] as u32)
        }));
        col_index += 1;
        trace[col_index].data[vec_index] = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_2[vec_index * N_LANES + i] as u32)
        }));
        col_index += 1;
        trace[col_index].data[vec_index] = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_3[vec_index * N_LANES + i] as u32)
        }));
        col_index += 1;
        trace[col_index].data[vec_index] = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_4[vec_index * N_LANES + i] as u32)
        }));
        col_index += 1;

        // Initial state.
        let input_left: [[M31; 8]; N_LANES] = std::array::from_fn(|i| {
            *data_flow
                .0
                .get(&control_flow.sel_1[vec_index * N_LANES + i])
                .unwrap()
        });
        let input_right: [[M31; 8]; N_LANES] = std::array::from_fn(|i| {
            *data_flow
                .0
                .get(&control_flow.sel_2[vec_index * N_LANES + i])
                .unwrap()
        });

        let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
            if state_i < 8 {
                PackedBaseField::from_array(std::array::from_fn(|i| input_left[i][state_i]))
            } else {
                PackedBaseField::from_array(std::array::from_fn(|i| input_right[i][state_i - 8]))
            }
        });
        let initial_state_first_half = *state.first_chunk::<8>().unwrap();
        state.iter().copied().for_each(|s| {
            trace[col_index].data[vec_index] = s;
            col_index += 1;
        });

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(FIRST_FOUR_ROUND_RC[round][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i]));
            apply_external_round_matrix(&mut state);
            state.iter().copied().for_each(|s| {
                trace[col_index].data[vec_index] = s;
                col_index += 1;
            });
        });

        // Partial rounds.
        (0..N_PARTIAL_ROUNDS).for_each(|round| {
            state[0] += PackedBaseField::broadcast(PARTIAL_ROUNDS_RC[round]);
            state[0] = pow5(state[0]);
            apply_internal_round_matrix(&mut state);
            trace[col_index].data[vec_index] = state[0];
            col_index += 1;
        });

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS - 1).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[round][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i]));
            apply_external_round_matrix(&mut state);
            state.iter().copied().for_each(|s| {
                trace[col_index].data[vec_index] = s;
                col_index += 1;
            });
        });

        {
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[3][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i]));
            apply_external_round_matrix(&mut state);
            for i in 0..8 {
                state[i] = state[i] + initial_state_first_half[i];
            }
            state.iter().copied().for_each(|s| {
                trace[col_index].data[vec_index] = s;
                col_index += 1;
            });
        }
        assert_eq!(col_index, N_COLUMNS);
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    trace
}

pub fn check_trace(trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>) {
    assert_eq!(trace.len(), N_COLUMNS);

    let log_size = trace[0].length.ilog2();

    fn apply_4x4_mds_matrix(
        x0: PackedM31,
        x1: PackedM31,
        x2: PackedM31,
        x3: PackedM31,
    ) -> [PackedM31; 4] {
        let t0 = x0 + x1;
        let t1 = x2 + x3;
        let t2 = x1.double() + t1;
        let t3 = x3.double() + t0;
        let t4 = t1.double().double() + t3;
        let t5 = t0.double().double() + t2;
        let t6 = t3 + t5;
        let t7 = t2 + t4;

        [t6, t5, t7, t4]
    }

    fn apply_16x16_mds_matrix(state: &mut [PackedM31; 16]) {
        let p1 = apply_4x4_mds_matrix(state[0], state[1], state[2], state[3]);
        let p2 = apply_4x4_mds_matrix(state[4], state[5], state[6], state[7]);
        let p3 = apply_4x4_mds_matrix(state[8], state[9], state[10], state[11]);
        let p4 = apply_4x4_mds_matrix(state[12], state[13], state[14], state[15]);

        let t = [
            p1[0] + p2[0] + p3[0] + p4[0],
            p1[1] + p2[1] + p3[1] + p4[1],
            p1[2] + p2[2] + p3[2] + p4[2],
            p1[3] + p2[3] + p3[3] + p4[3],
        ];

        for i in 0..4 {
            state[i] = p1[i] + t[i];
            state[i + 4] = p2[i] + t[i];
            state[i + 8] = p3[i] + t[i];
            state[i + 12] = p4[i] + t[i];
        }
    }

    fn pow5(v: PackedM31) -> PackedM31 {
        let t = v * v;
        t * t * v
    }

    for vec_index in 0..1 << (log_size - LOG_N_LANES) {
        let mut col_index = 4;
        let mut state: [_; N_STATE] = std::array::from_fn(|i| trace[col_index + i].data[vec_index]);
        col_index += N_STATE;

        // Require state lookup.
        let initial_state = state.clone();

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(FIRST_FOUR_ROUND_RC[round][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            apply_16x16_mds_matrix(&mut state);

            for i in 0..16 {
                assert_eq!(
                    state[i].into_simd(),
                    trace[col_index + i].data[vec_index].into_simd()
                );
            }
            col_index += 16;
        });

        // Partial rounds.
        (0..N_PARTIAL_ROUNDS).for_each(|round| {
            state[0] += PackedBaseField::broadcast(PARTIAL_ROUNDS_RC[round]);
            state[0] = pow5(state[0].clone());

            let mut sum = state[0];
            for i in 1..16 {
                sum += state[i];
            }
            for i in 0..16 {
                state[i] = sum + state[i] * PackedM31::broadcast(MAT_DIAG16_M_1[i]);
            }

            assert_eq!(
                state[0].into_simd(),
                trace[col_index].data[vec_index].into_simd()
            );
            col_index += 1;
        });

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS - 1).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[round][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            apply_16x16_mds_matrix(&mut state);

            for i in 0..16 {
                assert_eq!(
                    state[i].into_simd(),
                    trace[col_index + i].data[vec_index].into_simd()
                );
            }
            col_index += 16;
        });

        // 4 full rounds.
        {
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[3][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            apply_16x16_mds_matrix(&mut state);

            for i in 0..8 {
                assert_eq!(
                    (state[i] + initial_state[i]).into_simd(),
                    trace[col_index + i].data[vec_index].into_simd()
                );
            }
            col_index += 8;

            for i in 0..8 {
                assert_eq!(
                    state[i + 8].into_simd(),
                    trace[col_index + i].data[vec_index].into_simd()
                );
            }
            col_index += 8;
        }
        assert_eq!(col_index, N_COLUMNS);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::examples::plonk_with_poseidon::poseidon::{
        check_trace, gen_trace, PoseidonControlFlow, PoseidonDataFlow, PoseidonMetadata,
        PoseidonPrescribedFlow, CONSTANT_1, CONSTANT_2, CONSTANT_3,
    };

    fn get_test_metadata() -> PoseidonMetadata {
        let mut data_flow = PoseidonDataFlow(HashMap::new());
        data_flow.0.insert(123001, CONSTANT_1);
        data_flow.0.insert(123002, CONSTANT_2);
        data_flow.0.insert(123003, CONSTANT_3);

        let prescribed_flow = PoseidonPrescribedFlow {
            addr_1: vec![456001; 16],
            addr_2: vec![456001; 16],
            addr_3: vec![456002; 16],
            addr_4: vec![456003; 16],
        };

        let control_flow = PoseidonControlFlow {
            sel_1: vec![123001; 16],
            sel_2: vec![123001; 16],
            sel_3: vec![123002; 16],
            sel_4: vec![123003; 16],
        };

        PoseidonMetadata {
            prescribed_flow,
            control_flow,
            data_flow,
            constant_1_addr: 456001,
            constant_1_sel: 123001,
            constant_2_addr: 456002,
            constant_2_sel: 123002,
            constant_3_addr: 456003,
            constant_3_sel: 123003,
        }
    }

    #[test]
    fn test_trace() {
        let mut metadata = get_test_metadata();
        let trace = gen_trace(&mut metadata);
        check_trace(&trace);
    }
}
