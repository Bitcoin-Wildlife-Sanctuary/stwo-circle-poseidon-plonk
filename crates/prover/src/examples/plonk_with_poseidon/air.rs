use std::cmp::max;

use itertools::{chain, Itertools};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use tracing::{span, Level};

use crate::constraint_framework::preprocessed_columns::{gen_is_first, PreprocessedColumn};
use crate::constraint_framework::{
    Relation, TraceLocationAllocator, INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
    PREPROCESSED_TRACE_IDX,
};
use crate::core::air::{Component, ComponentProver};
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::BackendForChannel;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::FieldExpOps;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, verify, StarkProof, VerificationError};
use crate::core::vcs::ops::MerkleHasher;
use crate::examples::plonk_with_poseidon::plonk::{
    PlonkWithAcceleratorCircuitTrace, PlonkWithAcceleratorComponent, PlonkWithAcceleratorEval,
    PlonkWithAcceleratorLookupElements,
};
use crate::examples::plonk_with_poseidon::poseidon::{
    check_interaction_trace, check_trace, PoseidonAcceleratorComponent, PoseidonAcceleratorEval,
    PoseidonFlow,
};
use crate::examples::plonk_with_poseidon::{plonk, poseidon};

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithPoseidonStatement0 {
    pub log_size_plonk: u32,
    pub log_size_poseidon: u32,
}

impl PlonkWithPoseidonStatement0 {
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut sizes = TreeVec::new(vec![vec![], vec![], vec![]]);

        let log_size_plonk = self.log_size_plonk;
        let log_size_poseidon = self.log_size_poseidon;

        sizes[PREPROCESSED_TRACE_IDX].extend_from_slice(&[log_size_plonk; 10]);
        sizes[PREPROCESSED_TRACE_IDX].extend_from_slice(&[log_size_poseidon; 5]);

        sizes[ORIGINAL_TRACE_IDX].extend_from_slice(&[log_size_plonk; 12]);
        sizes[ORIGINAL_TRACE_IDX].extend_from_slice(&[log_size_poseidon; 166]);

        sizes[INTERACTION_TRACE_IDX].extend_from_slice(&[log_size_plonk; 8]);
        sizes[INTERACTION_TRACE_IDX].extend_from_slice(&[log_size_poseidon; 16]);

        sizes
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size_plonk as u64);
        channel.mix_u64(self.log_size_poseidon as u64);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithPoseidonStatement1 {
    pub plonk_total_sum: SecureField,
    pub poseidon_total_sum: SecureField,
}

impl PlonkWithPoseidonStatement1 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.plonk_total_sum, self.poseidon_total_sum]);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithPoseidonProof<H: MerkleHasher> {
    pub stmt0: PlonkWithPoseidonStatement0,
    pub stmt1: PlonkWithPoseidonStatement1,
    pub stark_proof: StarkProof<H>,
}

pub struct PlonkWithPoseidonComponents {
    pub plonk: PlonkWithAcceleratorComponent,
    pub poseidon: PoseidonAcceleratorComponent,
}

impl PlonkWithPoseidonComponents {
    pub fn new(
        stmt0: &PlonkWithPoseidonStatement0,
        lookup_elements: &PlonkWithAcceleratorLookupElements,
        stmt1: &PlonkWithPoseidonStatement1,
    ) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::new_with_preproccessed_columns(
            &chain!(
                [
                    PreprocessedColumn::Plonk(0),
                    PreprocessedColumn::Plonk(1),
                    PreprocessedColumn::Plonk(2),
                    PreprocessedColumn::Plonk(3),
                    PreprocessedColumn::Plonk(4),
                    PreprocessedColumn::Plonk(5),
                    PreprocessedColumn::Plonk(6),
                    PreprocessedColumn::Plonk(7),
                    PreprocessedColumn::Plonk(8),
                    PreprocessedColumn::IsFirst(stmt0.log_size_plonk as u32),
                ],
                [
                    PreprocessedColumn::Poseidon(0),
                    PreprocessedColumn::Poseidon(1),
                    PreprocessedColumn::Poseidon(2),
                    PreprocessedColumn::Poseidon(3),
                    PreprocessedColumn::IsFirst(stmt0.log_size_poseidon as u32),
                ]
            )
            .collect_vec()[..],
        );

        Self {
            plonk: PlonkWithAcceleratorComponent::new(
                tree_span_provider,
                PlonkWithAcceleratorEval {
                    log_n_rows: stmt0.log_size_plonk as u32,
                    lookup_elements: lookup_elements.clone(),
                    total_sum: stmt1.plonk_total_sum,
                },
                (stmt1.plonk_total_sum, None),
            ),
            poseidon: PoseidonAcceleratorComponent::new(
                tree_span_provider,
                PoseidonAcceleratorEval {
                    log_n_rows: stmt0.log_size_poseidon as u32,
                    lookup_elements: lookup_elements.clone(),
                    total_sum: stmt1.poseidon_total_sum,
                },
                (stmt1.poseidon_total_sum, None),
            ),
        }
    }

    pub fn components(&self) -> Vec<&dyn Component> {
        vec![
            &self.plonk as &dyn Component,
            &self.poseidon as &dyn Component,
        ]
    }

    pub fn component_provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![
            &self.plonk as &dyn ComponentProver<SimdBackend>,
            &self.poseidon as &dyn ComponentProver<SimdBackend>,
        ]
    }
}

pub fn prove_plonk_with_poseidon<MC: MerkleChannel>(
    log_size_plonk: u32,
    log_size_poseidon: u32,
    config: PcsConfig,
    circuit: &PlonkWithAcceleratorCircuitTrace,
    flow: &mut PoseidonFlow,
) -> PlonkWithPoseidonProof<MC::H>
where
    SimdBackend: BackendForChannel<MC>,
{
    assert!(log_size_plonk >= LOG_N_LANES);
    assert!(log_size_poseidon >= LOG_N_LANES);
    assert_eq!(circuit.mult_c.length, 1 << log_size_plonk);
    assert_eq!(flow.0.len(), 1 << log_size_poseidon);

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let log_max_rows = max(log_size_plonk + 1, log_size_poseidon + 2);
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_max_rows + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::new(config, &twiddles);

    // Preprocessed trace
    let is_first = gen_is_first(log_size_plonk);
    let mut plonk_constant_trace = [
        circuit.a_wire.clone(),
        circuit.b_wire.clone(),
        circuit.c_wire.clone(),
        circuit.op.clone(),
        circuit.mult_a.clone(),
        circuit.mult_b.clone(),
        circuit.mult_c.clone(),
        circuit.mult_poseidon.clone(),
        circuit.enforce_c_m31.clone(),
    ]
    .into_iter()
    .map(|eval| {
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            CanonicCoset::new(log_size_plonk).circle_domain(),
            eval.clone(),
        )
    })
    .collect_vec();
    plonk_constant_trace.push(is_first);

    let poseidon_trace = poseidon::gen_trace(flow);
    check_trace(&poseidon_trace);
    let poseidon_constant_trace = poseidon::gen_constant_trace(flow);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![
        plonk_constant_trace,
        poseidon_constant_trace.clone(),
    ]);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let plonk_trace = plonk::gen_trace(log_size_plonk, &circuit);

    // Statement0.
    let stmt0 = PlonkWithPoseidonStatement0 {
        log_size_plonk,
        log_size_poseidon,
    };
    stmt0.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![plonk_trace, poseidon_trace.clone()]);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (plonk_interaction_trace, plonk_total_sum) =
        plonk::gen_interaction_trace(log_size_plonk, &circuit, &lookup_elements);
    let (poseidon_interaction_trace, poseidon_total_sum) =
        poseidon::gen_interaction_trace(flow, &lookup_elements);
    check_interaction_trace(
        &poseidon_trace,
        &poseidon_interaction_trace,
        &poseidon_constant_trace,
        &lookup_elements,
        poseidon_total_sum,
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![plonk_interaction_trace, poseidon_interaction_trace]);
    // Statement1.
    let stmt1 = PlonkWithPoseidonStatement1 {
        plonk_total_sum,
        poseidon_total_sum,
    };
    stmt1.mix_into(channel);
    tree_builder.commit(channel);
    span.exit();

    assert_eq!(
        commitment_scheme
            .polynomials()
            .as_cols_ref()
            .map_cols(|c| c.log_size())
            .0,
        stmt0.log_sizes().0
    );

    // Prove constraints.
    let components = PlonkWithPoseidonComponents::new(&stmt0, &lookup_elements, &stmt1);
    let stark_proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();

    PlonkWithPoseidonProof {
        stmt0,
        stmt1,
        stark_proof,
    }
}

#[allow(unused)]
pub fn verify_plonk_with_poseidon<MC: MerkleChannel>(
    PlonkWithPoseidonProof {
        stmt0,
        stmt1,
        stark_proof,
    }: PlonkWithPoseidonProof<MC::H>,
    config: PcsConfig,
    inputs: &[(usize, QM31)],
) -> Result<(), VerificationError> {
    let channel = &mut MC::C::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(config);

    let log_sizes = stmt0.log_sizes();

    // Preprocessed trace.
    commitment_scheme.commit(stark_proof.commitments[0], &log_sizes[0], channel);

    // Trace.
    stmt0.mix_into(channel);
    commitment_scheme.commit(stark_proof.commitments[1], &log_sizes[1], channel);

    // Draw interaction elements.
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    stmt1.mix_into(channel);
    commitment_scheme.commit(stark_proof.commitments[2], &log_sizes[2], channel);

    let components = PlonkWithPoseidonComponents::new(&stmt0, &lookup_elements, &stmt1);

    let mut input_sum = SecureField::zero();
    for &(i, v) in inputs.iter() {
        let sum: SecureField =
            lookup_elements.combine(&[M31::from(i), v.0 .0, v.0 .1, v.1 .0, v.1 .1]);
        input_sum += sum.inverse();
    }

    let total_sum = stmt1.plonk_total_sum + input_sum + stmt1.poseidon_total_sum;
    assert_eq!(total_sum, SecureField::zero());

    verify(
        &components.components(),
        channel,
        commitment_scheme,
        stark_proof,
    )
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;
    use std::ops::Neg;

    use num_traits::{One, Zero};

    use crate::core::air::Component;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::QM31;
    use crate::core::fri::FriConfig;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use crate::core::prover::verify;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::vcs::poseidon31_merkle::{Poseidon31MerkleChannel, Poseidon31MerkleHasher};
    use crate::examples::plonk_with_poseidon::air::{
        prove_plonk_with_poseidon, verify_plonk_with_poseidon, PlonkWithPoseidonProof,
    };
    use crate::examples::plonk_with_poseidon::plonk::{
        prove_plonk_with_accelerator, PlonkWithAcceleratorCircuitTrace,
        PlonkWithAcceleratorLookupElements,
    };
    use crate::examples::plonk_with_poseidon::poseidon::{
        prove_poseidon_accelerator, PoseidonEntry, PoseidonFlow, CONSTANT_1, CONSTANT_2, CONSTANT_3,
    };

    fn generate_test_circuit() -> (PlonkWithAcceleratorCircuitTrace, PoseidonFlow) {
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

        let mut variables = vec![];
        let mut push_variable = |v: QM31| {
            let idx = variables.len();
            variables.push(v);
            idx
        };

        let zero_var = push_variable(QM31::zero());
        let one_var = push_variable(QM31::one());

        let mut mult_a = vec![];
        let mut mult_b = vec![];
        let mut mult_c = vec![];
        let mut mult_poseidon = vec![];
        let mut a_wire = vec![];
        let mut b_wire = vec![];
        let mut c_wire = vec![];
        let mut op = vec![];

        // allocate 0
        mult_a.push(1);
        mult_b.push(1);
        mult_c.push(-3);
        mult_poseidon.push(0);
        a_wire.push(zero_var);
        b_wire.push(zero_var);
        c_wire.push(zero_var);
        op.push(M31::one());

        // allocate 1
        mult_a.push(1);
        mult_b.push(1);
        mult_c.push(-2); // intentionally reduced by 1 to request for an input
        mult_poseidon.push(0);
        a_wire.push(one_var);
        b_wire.push(zero_var);
        c_wire.push(one_var);
        op.push(M31::one());

        let mut hash_idx = vec![];

        // allocate TEST_1 to TEST_4
        for g in [TEST_1, TEST_2, TEST_3, TEST_4] {
            let a = QM31::from_m31(g[0], g[1], g[2], g[3]);
            let b = QM31::from_m31(g[4], g[5], g[6], g[7]);
            let c = a * b;
            let a_var = push_variable(a);
            let b_var = push_variable(b);
            let c_var = push_variable(c);

            mult_a.push(0);
            mult_b.push(0);
            mult_c.push(0);
            mult_poseidon.push(16);
            a_wire.push(a_var);
            b_wire.push(b_var);
            c_wire.push(c_var);
            op.push(M31::zero());

            hash_idx.push(c_var);
        }

        assert_eq!(hash_idx.len(), 4);

        // allocate the hash_idx addresses
        let mut addr_hash_idx = vec![];
        for idx in hash_idx.iter() {
            let idx_var = push_variable(QM31::from(M31::from_u32_unchecked(*idx as u32)));

            mult_a.push(1);
            mult_b.push(1);
            mult_c.push(-17);
            mult_poseidon.push(0);
            a_wire.push(idx_var);
            b_wire.push(one_var);
            c_wire.push(idx_var);
            op.push(M31::zero());

            addr_hash_idx.push(idx_var);
            mult_c[1] -= 1;
        }

        let mut constant_idx = vec![];
        // allocate CONSTANT_1 to CONSTANT_3
        for (i, g) in [CONSTANT_1, CONSTANT_2, CONSTANT_3].iter().enumerate() {
            let a = QM31::from_m31(g[0], g[1], g[2], g[3]);
            let b = QM31::from_m31(g[4], g[5], g[6], g[7]);
            let c = a * b;
            let a_var = push_variable(a);
            let b_var = push_variable(b);
            let c_var = push_variable(c);

            mult_a.push(0);
            mult_b.push(0);
            mult_c.push(0);
            if i == 0 {
                mult_poseidon.push(32);
            } else {
                mult_poseidon.push(16);
            }
            a_wire.push(a_var);
            b_wire.push(b_var);
            c_wire.push(c_var);
            op.push(M31::zero());

            constant_idx.push(c_var);
        }

        // allocate the constant_idx addresses
        let mut addr_constant_idx = vec![];
        for (i, idx) in constant_idx.iter().enumerate() {
            let idx_var = push_variable(QM31::from(M31::from_u32_unchecked(*idx as u32)));

            mult_a.push(1);
            mult_b.push(1);
            if i == 0 {
                mult_c.push(-33);
            } else {
                mult_c.push(-17);
            }
            mult_poseidon.push(0);
            a_wire.push(idx_var);
            b_wire.push(one_var);
            c_wire.push(idx_var);
            op.push(M31::zero());

            addr_constant_idx.push(idx_var);
            mult_c[1] -= 1;
        }

        // assume that Poseidon has 32 gates,
        // 16 of them will be dealing with CONSTANT_1, _2, _3
        // 16 of them will be dealing with TEST_1, _2, _3, _4
        let len = mult_a.len();
        let padded_len = len.next_power_of_two();

        for _ in len..padded_len {
            mult_a.push(0);
            mult_b.push(0);
            mult_c.push(0);
            mult_poseidon.push(0);
            a_wire.push(0);
            b_wire.push(0);
            c_wire.push(0);
            op.push(M31::zero());
        }

        let mut counts = HashMap::<usize, isize>::new();
        assert_eq!(a_wire.len(), mult_a.len());
        for (&i, v) in a_wire.iter().zip(mult_a.iter()) {
            let p = counts.get(&i).copied().unwrap_or_default();
            counts.insert(i, p + v);
        }
        assert_eq!(b_wire.len(), mult_b.len());
        for (&i, v) in b_wire.iter().zip(mult_b.iter()) {
            let p = counts.get(&i).copied().unwrap_or_default();
            counts.insert(i, p + v);
        }
        assert_eq!(c_wire.len(), mult_c.len());
        for (&i, v) in c_wire.iter().zip(mult_c.iter()) {
            let p = counts.get(&i).copied().unwrap_or_default();
            counts.insert(i, p + v);
        }

        for (&k, &v) in counts.iter() {
            if !v.is_zero() {
                if addr_hash_idx.contains(&k) && v == -16 {
                    continue;
                } else if k == addr_constant_idx[0] && v == -32 {
                    continue;
                } else if (k == addr_constant_idx[1] || k == addr_constant_idx[2]) && v == -16 {
                    continue;
                } else {
                    assert!(
                        k == 1 && v == -1,
                        "The logUp sum is not expected: {} {}",
                        k,
                        v
                    );
                }
            }
        }

        let log_n_rows = padded_len.ilog2();
        let range = 0..(1 << log_n_rows);
        let isize_to_m31 = |v: isize| {
            if v.is_negative() {
                M31::from((-v) as u32).neg()
            } else {
                M31::from(v as u32)
            }
        };
        let circuit = PlonkWithAcceleratorCircuitTrace {
            mult_a: range.clone().map(|i| isize_to_m31(mult_a[i])).collect(),
            mult_b: range.clone().map(|i| isize_to_m31(mult_b[i])).collect(),
            mult_c: range.clone().map(|i| isize_to_m31(mult_c[i])).collect(),
            mult_poseidon: range
                .clone()
                .map(|i| isize_to_m31(mult_poseidon[i]))
                .collect(),
            enforce_c_m31: range.clone().map(|_| 0.into()).collect(),
            a_wire: range.clone().map(|i| a_wire[i].into()).collect(),
            b_wire: range.clone().map(|i| b_wire[i].into()).collect(),
            c_wire: range.clone().map(|i| c_wire[i].into()).collect(),
            op: range.clone().map(|i| op[i].into()).collect(),
            a_val_0: range.clone().map(|i| variables[a_wire[i]].0 .0).collect(),
            a_val_1: range.clone().map(|i| variables[a_wire[i]].0 .1).collect(),
            a_val_2: range.clone().map(|i| variables[a_wire[i]].1 .0).collect(),
            a_val_3: range.clone().map(|i| variables[a_wire[i]].1 .1).collect(),
            b_val_0: range.clone().map(|i| variables[b_wire[i]].0 .0).collect(),
            b_val_1: range.clone().map(|i| variables[b_wire[i]].0 .1).collect(),
            b_val_2: range.clone().map(|i| variables[b_wire[i]].1 .0).collect(),
            b_val_3: range.clone().map(|i| variables[b_wire[i]].1 .1).collect(),
            c_val_0: range.clone().map(|i| variables[c_wire[i]].0 .0).collect(),
            c_val_1: range.clone().map(|i| variables[c_wire[i]].0 .1).collect(),
            c_val_2: range.clone().map(|i| variables[c_wire[i]].1 .0).collect(),
            c_val_3: range.clone().map(|i| variables[c_wire[i]].1 .1).collect(),
        };

        let mut flow = PoseidonFlow::default();
        for i in 0..32 {
            if i % 2 == 0 {
                flow.0.push((
                    PoseidonEntry {
                        addr: addr_hash_idx[0],
                        sel: hash_idx[0],
                        hash: TEST_1,
                    },
                    PoseidonEntry {
                        addr: addr_hash_idx[1],
                        sel: hash_idx[1],
                        hash: TEST_2,
                    },
                    PoseidonEntry {
                        addr: addr_hash_idx[2],
                        sel: hash_idx[2],
                        hash: TEST_3,
                    },
                    PoseidonEntry {
                        addr: addr_hash_idx[3],
                        sel: hash_idx[3],
                        hash: TEST_4,
                    },
                ));
            } else {
                flow.0.push((
                    PoseidonEntry {
                        addr: addr_constant_idx[0],
                        sel: constant_idx[0],
                        hash: CONSTANT_1,
                    },
                    PoseidonEntry {
                        addr: addr_constant_idx[0],
                        sel: constant_idx[0],
                        hash: CONSTANT_1,
                    },
                    PoseidonEntry {
                        addr: addr_constant_idx[1],
                        sel: constant_idx[1],
                        hash: CONSTANT_2,
                    },
                    PoseidonEntry {
                        addr: addr_constant_idx[2],
                        sel: constant_idx[2],
                        hash: CONSTANT_3,
                    },
                ));
            }
        }

        for r in flow.0.iter() {
            for &v in [r.0.addr, r.1.addr, r.2.addr, r.3.addr].iter() {
                let p = counts.get(&v).unwrap();
                counts.insert(v, *p + 1);
            }
        }
        for (&k, &v) in counts.iter() {
            if !v.is_zero() {
                assert!(
                    k == 1 && v == -1,
                    "The logUp sum is not expected: {} {}",
                    k,
                    v
                );
            }
        }

        let mut counts_poseidon = HashMap::<usize, isize>::new();
        for (i, &v) in mult_poseidon.iter().enumerate() {
            counts_poseidon.insert(c_wire[i], v);
        }
        for r in flow.0.iter() {
            for &v in [r.0.sel, r.1.sel, r.2.sel, r.3.sel].iter() {
                if v != 0 {
                    let p = counts_poseidon.get(&v).unwrap();
                    counts_poseidon.insert(v, *p - 1);
                }
            }
        }
        for (_, &v) in counts_poseidon.iter() {
            assert!(v.is_zero());
        }

        (circuit, flow)
    }

    #[test]
    fn test_individual_proofs() {
        let (plonk, mut poseidon) = generate_test_circuit();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(2, 4, 64),
        };

        let (plonk_component, plonk_proof) = prove_plonk_with_accelerator::<Blake2sMerkleChannel>(
            plonk.mult_c.length.ilog2(),
            config,
            &plonk,
        );

        let (poseidon_component, poseidon_proof) = prove_poseidon_accelerator::<Blake2sMerkleChannel>(
            poseidon.0.len().ilog2(),
            config,
            &mut poseidon,
        );

        // If the two proofs are not generated together, their lookup parameters would be different,
        // but we can still verify them independently.

        // Verify the Plonk proof
        {
            // TODO: Create Air instance independently.
            let channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Decommit.
            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = plonk_component.trace_log_degree_bounds();

            // Preprocessed columns.
            commitment_scheme.commit(plonk_proof.commitments[0], &sizes[0], channel);

            // Trace columns.
            commitment_scheme.commit(plonk_proof.commitments[1], &sizes[1], channel);
            // Draw lookup element.
            let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);
            assert_eq!(lookup_elements, plonk_component.lookup_elements);
            // Interaction columns.
            commitment_scheme.commit(plonk_proof.commitments[2], &sizes[2], channel);

            verify(&[&plonk_component], channel, commitment_scheme, plonk_proof).unwrap();
        }

        // Verify the Poseidon proof
        {
            // TODO: Create Air instance independently.
            let channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Decommit.
            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = poseidon_component.trace_log_degree_bounds();

            // Preprocessed columns.
            commitment_scheme.commit(poseidon_proof.commitments[0], &sizes[0], channel);
            // Trace columns.
            commitment_scheme.commit(poseidon_proof.commitments[1], &sizes[1], channel);
            // Draw lookup element.
            let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);
            assert_eq!(lookup_elements, poseidon_component.lookup_elements);
            // Interaction columns.
            commitment_scheme.commit(poseidon_proof.commitments[2], &sizes[2], channel);

            verify(
                &[&poseidon_component],
                channel,
                commitment_scheme,
                poseidon_proof,
            )
            .unwrap();
        }
    }

    #[test]
    fn test_joint_proof() {
        let (plonk, mut poseidon) = generate_test_circuit();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(2, 4, 64),
        };

        let proof = prove_plonk_with_poseidon::<Blake2sMerkleChannel>(
            plonk.mult_c.length.ilog2(),
            poseidon.0.len().ilog2(),
            config,
            &plonk,
            &mut poseidon,
        );
        verify_plonk_with_poseidon::<Blake2sMerkleChannel>(proof, config, &[(1, QM31::one())])
            .unwrap();
    }

    #[test]
    fn test_joint_proof_serialize_poseidon31() {
        let (plonk, mut poseidon) = generate_test_circuit();
        let config = PcsConfig {
            pow_bits: 20,
            fri_config: FriConfig::new(0, 5, 16),
        };

        let proof = prove_plonk_with_poseidon::<Poseidon31MerkleChannel>(
            plonk.mult_c.length.ilog2(),
            poseidon.0.len().ilog2(),
            config,
            &plonk,
            &mut poseidon,
        );

        let encoded = bincode::serialize(&proof).unwrap();

        let mut file = File::create("joint_proof.bin").unwrap();
        file.write_all(&encoded).unwrap();

        let decoded: PlonkWithPoseidonProof<Poseidon31MerkleHasher> =
            bincode::deserialize(&encoded).unwrap();
        verify_plonk_with_poseidon::<Poseidon31MerkleChannel>(decoded, config, &[(1, QM31::one())])
            .unwrap();
    }
}
