use std::collections::HashMap;
use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
use num_traits::{One, Zero};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::{span, Level};

use crate::constraint_framework::logup::LogupTraceGenerator;
use crate::constraint_framework::preprocessed_columns::{gen_is_first, PreprocessedColumn};
use crate::constraint_framework::{
    assert_constraints, EvalAtRow, FrameworkComponent, FrameworkEval, Relation, RelationEntry,
    TraceLocationAllocator,
};
use crate::core::backend::simd::m31::{PackedBaseField, PackedM31, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{BackendForChannel, Col, Column};
use crate::core::channel::MerkleChannel;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::FieldExpOps;
use crate::core::pcs::{CommitmentSchemeProver, PcsConfig};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, StarkProof};
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::poseidon31_ref::{
    FIRST_FOUR_ROUND_RC, LAST_FOUR_ROUNDS_RC, MAT_DIAG16_M_1, PARTIAL_ROUNDS_RC,
};
use crate::core::ColumnVec;
use crate::examples::plonk_with_poseidon::plonk::PlonkWithAcceleratorLookupElements;

const N_STATE: usize = 16;
const N_HALF_FULL_ROUNDS: usize = 4;
const N_PARTIAL_ROUNDS: usize = 14;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
const N_COLUMNS: usize = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS + 4;
const LOG_EXPAND: u32 = 2;

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

pub type PoseidonAcceleratorComponent = FrameworkComponent<PoseidonAcceleratorEval>;

#[derive(Clone)]
pub struct PoseidonAcceleratorEval {
    pub log_n_rows: u32,
    pub lookup_elements: PlonkWithAcceleratorLookupElements,
    pub total_sum: SecureField,
}

impl FrameworkEval for PoseidonAcceleratorEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_EXPAND
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_poseidon_constraints(&mut eval, &self.lookup_elements);
        eval
    }
}

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
    lookup_elements: &PlonkWithAcceleratorLookupElements,
) {
    let addr_1 = eval.get_preprocessed_column(PreprocessedColumn::Poseidon(0));
    let addr_2 = eval.get_preprocessed_column(PreprocessedColumn::Poseidon(1));
    let addr_3 = eval.get_preprocessed_column(PreprocessedColumn::Poseidon(2));
    let addr_4 = eval.get_preprocessed_column(PreprocessedColumn::Poseidon(3));

    let sel_1 = eval.next_trace_mask();
    let sel_2 = eval.next_trace_mask();
    let sel_3 = eval.next_trace_mask();
    let sel_4 = eval.next_trace_mask();

    let mut state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

    // Require state lookup.
    let initial_state = state.clone();

    apply_external_round_matrix(&mut state);

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

        let m = eval.next_trace_mask();
        eval.add_constraint(state[0].clone() - m.clone());
        state[0] = m;

        apply_internal_round_matrix(&mut state);
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

    // the last full round
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

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[
            -sel_1,
            initial_state[0].clone(),
            initial_state[1].clone(),
            initial_state[2].clone(),
            initial_state[3].clone(),
            initial_state[4].clone(),
            initial_state[5].clone(),
            initial_state[6].clone(),
            initial_state[7].clone(),
        ],
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[
            -sel_2,
            initial_state[8].clone(),
            initial_state[9].clone(),
            initial_state[10].clone(),
            initial_state[11].clone(),
            initial_state[12].clone(),
            initial_state[13].clone(),
            initial_state[14].clone(),
            initial_state[15].clone(),
        ],
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[
            -sel_3,
            state[0].clone(),
            state[1].clone(),
            state[2].clone(),
            state[3].clone(),
            state[4].clone(),
            state[5].clone(),
            state[6].clone(),
            state[7].clone(),
        ],
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[
            -sel_4,
            state[8].clone(),
            state[9].clone(),
            state[10].clone(),
            state[11].clone(),
            state[12].clone(),
            state[13].clone(),
            state[14].clone(),
            state[15].clone(),
        ],
    ));

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
}

pub fn gen_trace(
    metadata: &PoseidonMetadata,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let prescribed_flow = &metadata.prescribed_flow;
    let control_flow = &metadata.control_flow;
    let data_flow = &metadata.data_flow;

    // check the length
    let len = prescribed_flow.addr_1.len();
    assert_eq!(len, prescribed_flow.addr_2.len());
    assert_eq!(len, prescribed_flow.addr_3.len());
    assert_eq!(len, prescribed_flow.addr_4.len());
    assert_eq!(len, control_flow.sel_1.len());
    assert_eq!(len, control_flow.sel_2.len());
    assert_eq!(len, control_flow.sel_3.len());
    assert_eq!(len, control_flow.sel_4.len());

    // compute the circuit size
    assert!(len.is_power_of_two());
    let log_size = len.ilog2();

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

        apply_external_round_matrix(&mut state);

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
            trace[col_index].data[vec_index] = state[0];
            col_index += 1;
            apply_internal_round_matrix(&mut state);
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

        apply_16x16_mds_matrix(&mut state);

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

            assert_eq!(
                state[0].into_simd(),
                trace[col_index].data[vec_index].into_simd()
            );
            col_index += 1;

            let mut sum = state[0];
            for i in 1..16 {
                sum += state[i];
            }
            for i in 0..16 {
                state[i] = sum + state[i] * PackedM31::broadcast(MAT_DIAG16_M_1[i]);
            }
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

pub fn gen_interaction_trace(
    metadata: &PoseidonMetadata,
    lookup_elements: &PlonkWithAcceleratorLookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let prescribed_flow = &metadata.prescribed_flow;
    let control_flow = &metadata.control_flow;
    let data_flow = &metadata.data_flow;

    // check the length
    let len = prescribed_flow.addr_1.len();
    assert_eq!(len, prescribed_flow.addr_2.len());
    assert_eq!(len, prescribed_flow.addr_3.len());
    assert_eq!(len, prescribed_flow.addr_4.len());
    assert_eq!(len, control_flow.sel_1.len());
    assert_eq!(len, control_flow.sel_2.len());
    assert_eq!(len, control_flow.sel_3.len());
    assert_eq!(len, control_flow.sel_4.len());

    // compute the circuit size
    assert!(len.is_power_of_two());

    let log_size = len.ilog2();
    assert!(log_size >= LOG_N_LANES);

    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let addr_1 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(prescribed_flow.addr_1[vec_row * N_LANES + i] as u32)
        }));
        let sel_1 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_1[vec_row * N_LANES + i] as u32)
        }));
        let denom0: PackedSecureField = lookup_elements.combine(&[addr_1, sel_1]);

        let addr_2 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(prescribed_flow.addr_2[vec_row * N_LANES + i] as u32)
        }));
        let sel_2 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_2[vec_row * N_LANES + i] as u32)
        }));
        let denom1: PackedSecureField = lookup_elements.combine(&[addr_2, sel_2]);

        // (1 / denom1) + (1 / denom1) = (denom1 + denom0) / (denom0 * denom1).
        col_gen.write_frac(vec_row, denom1 + denom0, denom0 * denom1);
    }
    col_gen.finalize_col();

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let addr_3 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(prescribed_flow.addr_3[vec_row * N_LANES + i] as u32)
        }));
        let sel_3 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_3[vec_row * N_LANES + i] as u32)
        }));
        let denom0: PackedSecureField = lookup_elements.combine(&[addr_3, sel_3]);

        let addr_4 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(prescribed_flow.addr_4[vec_row * N_LANES + i] as u32)
        }));
        let sel_4 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_4[vec_row * N_LANES + i] as u32)
        }));
        let denom1: PackedSecureField = lookup_elements.combine(&[addr_4, sel_4]);

        // (1 / denom1) + (1 / denom1) = (denom1 + denom0) / (denom0 * denom1).
        col_gen.write_frac(vec_row, denom1 + denom0, denom0 * denom1);
    }
    col_gen.finalize_col();

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let sel_1 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_1[vec_row * N_LANES + i] as u32)
        }));
        let sel_2 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_2[vec_row * N_LANES + i] as u32)
        }));
        let src_1: [[M31; 8]; N_LANES] = std::array::from_fn(|i| {
            let r = data_flow
                .0
                .get(&control_flow.sel_1[vec_row * N_LANES + i])
                .unwrap();
            r.clone()
        });
        let src_2: [[M31; 8]; N_LANES] = std::array::from_fn(|i| {
            let r = data_flow
                .0
                .get(&control_flow.sel_2[vec_row * N_LANES + i])
                .unwrap();
            r.clone()
        });
        let mut denom0_arr = Vec::with_capacity(9);
        denom0_arr.push(-sel_1);
        for j in 0..8 {
            denom0_arr.push(PackedM31::from_array(std::array::from_fn(|i| src_1[i][j])));
        }
        let denom0: PackedSecureField = lookup_elements.combine(&denom0_arr);
        let mut denom1_arr = Vec::with_capacity(9);
        denom1_arr.push(-sel_2);
        for j in 0..8 {
            denom1_arr.push(PackedM31::from_array(std::array::from_fn(|i| src_2[i][j])));
        }
        let denom1: PackedSecureField = lookup_elements.combine(&denom1_arr);
        // (1 / denom1) + (1 / denom1) = (denom1 + denom0) / (denom0 * denom1).
        col_gen.write_frac(vec_row, denom1 + denom0, denom0 * denom1);
    }
    col_gen.finalize_col();

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let sel_3 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_3[vec_row * N_LANES + i] as u32)
        }));
        let sel_4 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(control_flow.sel_4[vec_row * N_LANES + i] as u32)
        }));
        let src_3: [[M31; 8]; N_LANES] = std::array::from_fn(|i| {
            let r = data_flow
                .0
                .get(&control_flow.sel_3[vec_row * N_LANES + i])
                .unwrap();
            r.clone()
        });
        let src_4: [[M31; 8]; N_LANES] = std::array::from_fn(|i| {
            let r = data_flow
                .0
                .get(&control_flow.sel_4[vec_row * N_LANES + i])
                .unwrap();
            r.clone()
        });
        let mut denom0_arr = Vec::with_capacity(9);
        denom0_arr.push(-sel_3);
        for j in 0..8 {
            denom0_arr.push(PackedM31::from_array(std::array::from_fn(|i| src_3[i][j])));
        }
        let denom0: PackedSecureField = lookup_elements.combine(&denom0_arr);
        let mut denom1_arr = Vec::with_capacity(9);
        denom1_arr.push(-sel_4);
        for j in 0..8 {
            denom1_arr.push(PackedM31::from_array(std::array::from_fn(|i| src_4[i][j])));
        }
        let denom1: PackedSecureField = lookup_elements.combine(&denom1_arr);
        // (1 / denom1) + (1 / denom1) = (denom1 + denom0) / (denom0 * denom1).
        col_gen.write_frac(vec_row, denom1 + denom0, denom0 * denom1);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

pub fn check_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    interaction: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    constant: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    lookup_elements: &PlonkWithAcceleratorLookupElements,
    total_sum: SecureField,
) {
    assert_eq!(trace.len(), N_COLUMNS);
    assert_eq!(interaction.len(), (4 + 4) / 2 * 4);

    let mut addr_sel_sum = QM31::zero();

    let log_size = trace[0].length.ilog2();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let addr_1 = constant[0].data[vec_row];
        let addr_2 = constant[1].data[vec_row];
        let addr_3 = constant[2].data[vec_row];
        let addr_4 = constant[3].data[vec_row];

        let sel_1 = trace[0].data[vec_row];
        let sel_2 = trace[1].data[vec_row];
        let sel_3 = trace[2].data[vec_row];
        let sel_4 = trace[3].data[vec_row];

        let v1: PackedSecureField = lookup_elements.combine(&[addr_1, sel_1]);
        addr_sel_sum += v1.inverse().pointwise_sum();

        let v2: PackedSecureField = lookup_elements.combine(&[addr_2, sel_2]);
        addr_sel_sum += v2.inverse().pointwise_sum();

        let v3: PackedSecureField = lookup_elements.combine(&[addr_3, sel_3]);
        addr_sel_sum += v3.inverse().pointwise_sum();

        let v4: PackedSecureField = lookup_elements.combine(&[addr_4, sel_4]);
        addr_sel_sum += v4.inverse().pointwise_sum();
    }

    let mut input_sum = QM31::zero();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let sel_1 = trace[0].data[vec_row];
        let sel_2 = trace[1].data[vec_row];

        let input_state_1: [PackedBaseField; 8] =
            std::array::from_fn(|i| trace[4 + i].data[vec_row]);
        let input_state_2: [PackedBaseField; 8] =
            std::array::from_fn(|i| trace[12 + i].data[vec_row]);

        let v1: PackedSecureField = lookup_elements.combine(&[
            -sel_1,
            input_state_1[0],
            input_state_1[1],
            input_state_1[2],
            input_state_1[3],
            input_state_1[4],
            input_state_1[5],
            input_state_1[6],
            input_state_1[7],
        ]);
        input_sum += v1.inverse().pointwise_sum();

        let v2: PackedSecureField = lookup_elements.combine(&[
            -sel_2,
            input_state_2[0],
            input_state_2[1],
            input_state_2[2],
            input_state_2[3],
            input_state_2[4],
            input_state_2[5],
            input_state_2[6],
            input_state_2[7],
        ]);
        input_sum += v2.inverse().pointwise_sum();
    }

    let mut output_sum = QM31::zero();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let sel_3 = trace[2].data[vec_row];
        let sel_4 = trace[3].data[vec_row];

        let output_state_1: [PackedBaseField; 8] =
            std::array::from_fn(|i| trace[146 + i].data[vec_row]);
        let output_state_2: [PackedBaseField; 8] =
            std::array::from_fn(|i| trace[154 + i].data[vec_row]);

        let v1: PackedSecureField = lookup_elements.combine(&[
            -sel_3,
            output_state_1[0],
            output_state_1[1],
            output_state_1[2],
            output_state_1[3],
            output_state_1[4],
            output_state_1[5],
            output_state_1[6],
            output_state_1[7],
        ]);
        output_sum += v1.inverse().pointwise_sum();

        let v2: PackedSecureField = lookup_elements.combine(&[
            -sel_4,
            output_state_2[0],
            output_state_2[1],
            output_state_2[2],
            output_state_2[3],
            output_state_2[4],
            output_state_2[5],
            output_state_2[6],
            output_state_2[7],
        ]);
        output_sum += v2.inverse().pointwise_sum();
    }

    assert_eq!(total_sum, addr_sel_sum + input_sum + output_sum);
}

pub fn gen_constant_trace(
    metadata: &PoseidonMetadata,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let prescribed_flow = &metadata.prescribed_flow;
    let log_n_rows = prescribed_flow.addr_1.len().ilog2();

    let mut res = [
        prescribed_flow.addr_1.clone(),
        prescribed_flow.addr_2.clone(),
        prescribed_flow.addr_3.clone(),
        prescribed_flow.addr_4.clone(),
    ]
    .into_iter()
    .map(|col| {
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            CanonicCoset::new(log_n_rows).circle_domain(),
            col.iter()
                .map(|x| BaseField::from_u32_unchecked(*x as u32))
                .collect(),
        )
    })
    .collect_vec();
    res.push(gen_is_first(log_n_rows));
    res
}

pub fn prove_poseidon_accelerator<MC: MerkleChannel>(
    log_n_rows: u32,
    config: PcsConfig,
    metadata: &mut PoseidonMetadata,
) -> (PoseidonAcceleratorComponent, StarkProof<MC::H>)
where
    SimdBackend: BackendForChannel<MC>,
{
    // Prepare a fibonacci circuit.
    assert!(log_n_rows >= LOG_N_LANES);

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + LOG_EXPAND)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<_, MC>::new(config, &twiddles);

    let trace = gen_trace(metadata);
    check_trace(&trace);
    let constant_trace = gen_constant_trace(metadata);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(constant_trace.clone());
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (interaction_trace, total_sum) = gen_interaction_trace(metadata, &lookup_elements);
    check_interaction_trace(
        &trace,
        &interaction_trace,
        &constant_trace,
        &lookup_elements,
        total_sum,
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace);
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonAcceleratorComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonAcceleratorEval {
            log_n_rows,
            lookup_elements,
            total_sum,
        },
        (total_sum, None),
    );

    // Sanity check. Remove for production.
    let trace_polys = commitment_scheme
        .trees
        .as_ref()
        .map(|t| t.polynomials.iter().cloned().collect_vec());
    assert_constraints(
        &trace_polys,
        CanonicCoset::new(log_n_rows),
        |eval| {
            component.evaluate(eval);
        },
        (total_sum, None),
    );

    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

#[allow(unused)]
pub fn prove_test_poseidon_accelerator(
    log_n_rows: u32,
    config: PcsConfig,
) -> (
    PoseidonAcceleratorComponent,
    StarkProof<Blake2sMerkleHasher>,
) {
    // Additional test constants
    const TEST_1: [BaseField; 8] = [
        BaseField::from_u32_unchecked(0),
        BaseField::from_u32_unchecked(1),
        BaseField::from_u32_unchecked(2),
        BaseField::from_u32_unchecked(3),
        BaseField::from_u32_unchecked(4),
        BaseField::from_u32_unchecked(5),
        BaseField::from_u32_unchecked(6),
        BaseField::from_u32_unchecked(7),
    ];
    const TEST_2: [BaseField; 8] = [
        BaseField::from_u32_unchecked(8),
        BaseField::from_u32_unchecked(9),
        BaseField::from_u32_unchecked(10),
        BaseField::from_u32_unchecked(11),
        BaseField::from_u32_unchecked(12),
        BaseField::from_u32_unchecked(13),
        BaseField::from_u32_unchecked(14),
        BaseField::from_u32_unchecked(15),
    ];

    const TEST_3: [BaseField; 8] = [
        BaseField::from_u32_unchecked(0x0f8b2223),
        BaseField::from_u32_unchecked(0x4681926c),
        BaseField::from_u32_unchecked(0x62bf39da),
        BaseField::from_u32_unchecked(0x2c775855),
        BaseField::from_u32_unchecked(0x0621c328),
        BaseField::from_u32_unchecked(0x6c092e66),
        BaseField::from_u32_unchecked(0x1ebf9d29),
        BaseField::from_u32_unchecked(0x2d015c8e),
    ];

    const TEST_4: [BaseField; 8] = [
        BaseField::from_u32_unchecked(0x6447f97c),
        BaseField::from_u32_unchecked(0x4b6cc939),
        BaseField::from_u32_unchecked(0x0e395f63),
        BaseField::from_u32_unchecked(0x0bf7f688),
        BaseField::from_u32_unchecked(0x62ed4765),
        BaseField::from_u32_unchecked(0x7bfd5e1b),
        BaseField::from_u32_unchecked(0x4bafb4b0),
        BaseField::from_u32_unchecked(0x4cc30530),
    ];

    // Prepare a fibonacci circuit.
    assert!(log_n_rows >= LOG_N_LANES);
    let n_rows = (1 << log_n_rows) as usize;

    let mut data_flow = PoseidonDataFlow(HashMap::new());
    data_flow.0.insert(123001, CONSTANT_1);
    data_flow.0.insert(123002, CONSTANT_2);
    data_flow.0.insert(123003, CONSTANT_3);
    data_flow.0.insert(256001, TEST_1);
    data_flow.0.insert(256002, TEST_2);
    data_flow.0.insert(256003, TEST_3);
    data_flow.0.insert(256004, TEST_4);

    let prescribed_flow = PoseidonPrescribedFlow {
        addr_1: vec![456001; n_rows],
        addr_2: vec![456001; n_rows],
        addr_3: vec![456002; n_rows],
        addr_4: vec![456003; n_rows],
    };

    let (sel_1, sel_2, sel_3, sel_4) = {
        let mut sel_1 = Vec::with_capacity(n_rows);
        let mut sel_2 = Vec::with_capacity(n_rows);
        let mut sel_3 = Vec::with_capacity(n_rows);
        let mut sel_4 = Vec::with_capacity(n_rows);

        let mut prng = SmallRng::seed_from_u64(0);
        for _ in 0..n_rows {
            if prng.gen::<bool>() == true {
                sel_1.push(123001);
                sel_2.push(123001);
                sel_3.push(123002);
                sel_4.push(123003);
            } else {
                sel_1.push(256001);
                sel_2.push(256002);
                sel_3.push(256003);
                sel_4.push(256004);
            }
        }
        (sel_1, sel_2, sel_3, sel_4)
    };

    let control_flow = PoseidonControlFlow {
        sel_1,
        sel_2,
        sel_3,
        sel_4,
    };

    let mut metadata = PoseidonMetadata {
        prescribed_flow,
        control_flow,
        data_flow,
    };

    prove_poseidon_accelerator::<Blake2sMerkleChannel>(log_n_rows, config, &mut metadata)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::env;

    use itertools::Itertools;

    use crate::constraint_framework::assert_constraints;
    use crate::core::air::Component;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fri::FriConfig;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::prover::verify;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::examples::plonk_with_poseidon::plonk::PlonkWithAcceleratorLookupElements;
    use crate::examples::plonk_with_poseidon::poseidon::{
        check_interaction_trace, check_trace, eval_poseidon_constraints, gen_constant_trace,
        gen_interaction_trace, gen_trace, prove_test_poseidon_accelerator, PoseidonControlFlow,
        PoseidonDataFlow, PoseidonMetadata, PoseidonPrescribedFlow, CONSTANT_1, CONSTANT_2,
        CONSTANT_3,
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
        }
    }

    #[test]
    fn test_poseidon_trace() {
        let metadata = get_test_metadata();
        let trace = gen_trace(&metadata);
        check_trace(&trace);

        let log_n_rows = metadata.prescribed_flow.addr_1.len().ilog2();

        let mut channel = Blake2sChannel::default();
        let lookup_elements = PlonkWithAcceleratorLookupElements::draw(&mut channel);

        let constant = gen_constant_trace(&metadata);

        let (interaction, total_sum) = gen_interaction_trace(&metadata, &lookup_elements);
        check_interaction_trace(&trace, &interaction, &constant, &lookup_elements, total_sum);

        let traces = TreeVec::new(vec![constant, trace, interaction]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());
        assert_constraints(
            &trace_polys,
            CanonicCoset::new(log_n_rows),
            |mut eval| {
                eval_poseidon_constraints(&mut eval, &lookup_elements);
            },
            (total_sum, None),
        );
    }

    #[test_log::test]
    fn test_simd_poseidon_prove() {
        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 4, 64),
        };

        // Prove.
        let (component, proof) = prove_test_poseidon_accelerator(log_n_instances, config);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

        // Decommit.
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Trace columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup element.
        let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(&[&component], channel, commitment_scheme, proof).unwrap();
    }
}
