use std::ops::{Add, AddAssign, Mul, Neg, Sub};

use itertools::Itertools;
use num_traits::One;
use tracing::{span, Level};

use crate::constraint_framework::logup::LogupTraceGenerator;
use crate::constraint_framework::{
    assert_constraints, EvalAtRow, FrameworkComponent, FrameworkEval, Relation, RelationEntry,
    TraceLocationAllocator,
};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{BackendForChannel, Column};
use crate::core::channel::MerkleChannel;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentSchemeProver, PcsConfig};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, StarkProof};
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::ColumnVec;
use crate::examples::plonk::Plonk;
use crate::relation;

pub type PlonkWithoutAcceleratorComponent = FrameworkComponent<PlonkWithoutAcceleratorEval>;
relation!(PlonkWithoutAcceleratorLookupElements, 2);

#[derive(Clone)]
pub struct PlonkWithoutAcceleratorEval {
    pub log_n_rows: u32,
    pub lookup_elements: PlonkWithoutAcceleratorLookupElements,
    pub total_sum: SecureField,
}

impl FrameworkEval for PlonkWithoutAcceleratorEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 2
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let a_wire = eval.get_preprocessed_column(Plonk::new("a_wire".to_string()).id());
        let b_wire = eval.get_preprocessed_column(Plonk::new("b_wire".to_string()).id());
        // Note: c_wire could also be implicit: (self.eval.point() - M31_CIRCLE_GEN.into_ef()).x.
        //   A constant column is easier though.
        let c_wire = eval.get_preprocessed_column(Plonk::new("c_wire".to_string()).id());
        let op1 = eval.get_preprocessed_column(Plonk::new("op1".to_string()).id());
        let op2 = eval.get_preprocessed_column(Plonk::new("op2".to_string()).id());
        let op3 = eval.get_preprocessed_column(Plonk::new("op3".to_string()).id());
        let op4 = eval.get_preprocessed_column(Plonk::new("op4".to_string()).id());
        let mult_c = eval.get_preprocessed_column(Plonk::new("mult_c".to_string()).id());

        let one_minus_op3 = E::F::one() - op3.clone();
        let one_minus_op4 = E::F::one() - op4.clone();

        let is_arith = E::EF::from(one_minus_op3.clone()) * one_minus_op4.clone();
        let is_pow5 = E::EF::from(op2.clone());
        let is_m4 = E::EF::from(op3.clone()) * one_minus_op4.clone();
        let is_hadamard_product = E::EF::from(one_minus_op3.clone()) * op4.clone();
        let is_grand_sum = E::EF::from(op3.clone()) * op4.clone();

        let a_val_0 = eval.next_trace_mask();
        let a_val_1 = eval.next_trace_mask();
        let a_val_2 = eval.next_trace_mask();
        let a_val_3 = eval.next_trace_mask();

        let b_val_0 = eval.next_trace_mask();
        let b_val_1 = eval.next_trace_mask();
        let b_val_2 = eval.next_trace_mask();
        let b_val_3 = eval.next_trace_mask();

        let c_val_0 = eval.next_trace_mask();
        let c_val_1 = eval.next_trace_mask();
        let c_val_2 = eval.next_trace_mask();
        let c_val_3 = eval.next_trace_mask();

        let a_val = E::EF::from(a_val_0.clone())
            + a_val_1.clone() * SecureField::from_u32_unchecked(0, 1, 0, 0)
            + a_val_2.clone() * SecureField::from_u32_unchecked(0, 0, 1, 0)
            + a_val_3.clone() * SecureField::from_u32_unchecked(0, 0, 0, 1);

        let b_val = E::EF::from(b_val_0.clone())
            + b_val_1.clone() * SecureField::from_u32_unchecked(0, 1, 0, 0)
            + b_val_2.clone() * SecureField::from_u32_unchecked(0, 0, 1, 0)
            + b_val_3.clone() * SecureField::from_u32_unchecked(0, 0, 0, 1);

        let c_val = E::EF::from(c_val_0.clone())
            + c_val_1.clone() * SecureField::from_u32_unchecked(0, 1, 0, 0)
            + c_val_2.clone() * SecureField::from_u32_unchecked(0, 0, 1, 0)
            + c_val_3.clone() * SecureField::from_u32_unchecked(0, 0, 0, 1);

        eval.add_constraint(
            (E::EF::from(a_val_0.clone()) * a_val_0.clone() * a_val_0.clone() * a_val_0.clone()
                - E::EF::from(b_val_0.clone()))
                * is_pow5.clone(),
        );
        eval.add_constraint(
            (E::EF::from(a_val_1.clone()) * a_val_1.clone() * a_val_1.clone() * a_val_1.clone()
                - E::EF::from(b_val_1.clone()))
                * is_pow5.clone(),
        );
        eval.add_constraint(
            (E::EF::from(a_val_2.clone()) * a_val_2.clone() * a_val_2.clone() * a_val_2.clone()
                - E::EF::from(b_val_2.clone()))
                * is_pow5.clone(),
        );
        eval.add_constraint(
            (E::EF::from(a_val_3.clone()) * a_val_3.clone() * a_val_3.clone() * a_val_3.clone()
                - E::EF::from(b_val_3.clone()))
                * is_pow5.clone(),
        );

        #[inline(always)]
        /// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
        fn apply_m4<F>(x: [F; 4]) -> [F; 4]
        where
            F: Clone
                + AddAssign<F>
                + Add<F, Output = F>
                + Sub<F, Output = F>
                + Mul<BaseField, Output = F>,
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

        let m4_result = apply_m4([
            E::EF::from(a_val_0.clone()) * b_val_0.clone(),
            E::EF::from(a_val_1.clone()) * b_val_1.clone(),
            E::EF::from(a_val_2.clone()) * b_val_2.clone(),
            E::EF::from(a_val_3.clone()) * b_val_3.clone(),
        ]);

        let grand_sum = a_val_0.clone()
            + a_val_1.clone()
            + a_val_2.clone()
            + a_val_3.clone()
            + b_val_0.clone()
            + b_val_1.clone()
            + b_val_2.clone()
            + b_val_3.clone();

        eval.add_constraint(
            c_val.clone()
                - is_arith * E::EF::from(op1.clone()) * (a_val.clone() + b_val.clone())
                - E::EF::from(E::F::one() - op1) * a_val.clone() * b_val.clone()
                - is_m4.clone() * m4_result[0].clone()
                - is_m4.clone()
                    * m4_result[1].clone()
                    * SecureField::from_u32_unchecked(0, 1, 0, 0)
                - is_m4.clone()
                    * m4_result[2].clone()
                    * SecureField::from_u32_unchecked(0, 0, 1, 0)
                - is_m4.clone()
                    * m4_result[3].clone()
                    * SecureField::from_u32_unchecked(0, 0, 0, 1)
                - is_hadamard_product.clone() * (a_val_0.clone() * b_val_0.clone())
                - is_hadamard_product.clone()
                    * (a_val_1.clone() * b_val_1.clone())
                    * SecureField::from_u32_unchecked(0, 1, 0, 0)
                - is_hadamard_product.clone()
                    * (a_val_2.clone() * b_val_2.clone())
                    * SecureField::from_u32_unchecked(0, 0, 1, 0)
                - is_hadamard_product.clone()
                    * (a_val_3.clone() * b_val_3.clone())
                    * SecureField::from_u32_unchecked(0, 0, 0, 1)
                - is_grand_sum.clone() * grand_sum * SecureField::from_u32_unchecked(1, 1, 1, 1),
        );

        eval.add_to_relation_ef(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[a_val.clone(), E::EF::from(a_wire.clone())],
        ));
        eval.add_to_relation_ef(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[b_val.clone(), E::EF::from(b_wire.clone())],
        ));
        eval.add_to_relation_ef(RelationEntry::new(
            &self.lookup_elements,
            mult_c.into(),
            &[c_val.clone(), E::EF::from(c_wire.clone())],
        ));

        eval.finalize_logup_batched(&vec![0, 0, 0]);
        eval
    }
}

#[derive(Clone)]
pub struct PlonkWithoutAcceleratorCircuitTrace {
    pub a_wire: BaseColumn,
    pub b_wire: BaseColumn,
    pub c_wire: BaseColumn,
    pub op1: BaseColumn,
    pub op2: BaseColumn,
    pub op3: BaseColumn,
    pub op4: BaseColumn,
    pub mult_c: BaseColumn,
    pub a_val_0: BaseColumn,
    pub a_val_1: BaseColumn,
    pub a_val_2: BaseColumn,
    pub a_val_3: BaseColumn,
    pub b_val_0: BaseColumn,
    pub b_val_1: BaseColumn,
    pub b_val_2: BaseColumn,
    pub b_val_3: BaseColumn,
    pub c_val_0: BaseColumn,
    pub c_val_1: BaseColumn,
    pub c_val_2: BaseColumn,
    pub c_val_3: BaseColumn,
}
pub fn gen_trace(
    log_size: u32,
    circuit: &PlonkWithoutAcceleratorCircuitTrace,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Generation").entered();
    [
        &circuit.a_val_0,
        &circuit.a_val_1,
        &circuit.a_val_2,
        &circuit.a_val_3,
        &circuit.b_val_0,
        &circuit.b_val_1,
        &circuit.b_val_2,
        &circuit.b_val_3,
        &circuit.c_val_0,
        &circuit.c_val_1,
        &circuit.c_val_2,
        &circuit.c_val_3,
    ]
    .into_iter()
    .map(|eval| CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), eval.clone()))
    .collect()
}

pub fn gen_interaction_trace(
    log_size: u32,
    circuit: &PlonkWithoutAcceleratorCircuitTrace,
    lookup_elements: &PlonkWithoutAcceleratorLookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let q0: PackedSecureField = <PlonkWithoutAcceleratorLookupElements as Relation<
            PackedBaseField,
            PackedSecureField,
        >>::combine_ef(
            &lookup_elements,
            &[
                PackedSecureField::from_packed_m31s([
                    circuit.a_val_0.data[vec_row],
                    circuit.a_val_1.data[vec_row],
                    circuit.a_val_2.data[vec_row],
                    circuit.a_val_3.data[vec_row],
                ]),
                PackedSecureField::from(circuit.a_wire.data[vec_row]),
            ],
        );
        let q1: PackedSecureField = <PlonkWithoutAcceleratorLookupElements as Relation<
            PackedBaseField,
            PackedSecureField,
        >>::combine_ef(
            &lookup_elements,
            &[
                PackedSecureField::from_packed_m31s([
                    circuit.b_val_0.data[vec_row],
                    circuit.b_val_1.data[vec_row],
                    circuit.b_val_2.data[vec_row],
                    circuit.b_val_3.data[vec_row],
                ]),
                PackedSecureField::from(circuit.b_wire.data[vec_row]),
            ],
        );

        let pab = q0 + q1;
        let qab = q0 * q1;

        let pc = circuit.mult_c.data[vec_row];
        let qc: PackedSecureField = <PlonkWithoutAcceleratorLookupElements as Relation<
            PackedBaseField,
            PackedSecureField,
        >>::combine_ef(
            &lookup_elements,
            &[
                PackedSecureField::from_packed_m31s([
                    circuit.c_val_0.data[vec_row],
                    circuit.c_val_1.data[vec_row],
                    circuit.c_val_2.data[vec_row],
                    circuit.c_val_3.data[vec_row],
                ]),
                PackedSecureField::from(circuit.c_wire.data[vec_row]),
            ],
        );

        col_gen.write_frac(vec_row, pab * qc + qab * pc, qab * qc);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

pub fn prove_plonk_without_accelerator<MC: MerkleChannel>(
    log_n_rows: u32,
    config: PcsConfig,
    circuit: &PlonkWithoutAcceleratorCircuitTrace,
) -> (PlonkWithoutAcceleratorComponent, StarkProof<MC::H>)
where
    SimdBackend: BackendForChannel<MC>,
{
    assert!(log_n_rows >= LOG_N_LANES);

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + 2)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<_, MC>::new(config, &twiddles);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let constant_trace = [
        circuit.a_wire.clone(),
        circuit.b_wire.clone(),
        circuit.c_wire.clone(),
        circuit.op1.clone(),
        circuit.op2.clone(),
        circuit.op3.clone(),
        circuit.op4.clone(),
        circuit.mult_c.clone(),
    ]
    .into_iter()
    .map(|col| {
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            CanonicCoset::new(log_n_rows).circle_domain(),
            col,
        )
    })
    .collect_vec();
    tree_builder.extend_evals(constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let trace = gen_trace(log_n_rows, &circuit);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkWithoutAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, total_sum) = gen_interaction_trace(log_n_rows, &circuit, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();
    // Prove constraints.
    let component = PlonkWithoutAcceleratorComponent::new(
        &mut TraceLocationAllocator::default(),
        PlonkWithoutAcceleratorEval {
            log_n_rows,
            lookup_elements,
            total_sum,
        },
        total_sum,
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
        total_sum,
    );

    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

#[allow(unused)]
pub fn prove_fibonacci_plonk_without_accelerator(
    log_n_rows: u32,
    config: PcsConfig,
) -> (
    PlonkWithoutAcceleratorComponent,
    StarkProof<Blake2sMerkleHasher>,
) {
    assert!(log_n_rows >= LOG_N_LANES);

    // Prepare a fibonacci circuit.
    let mut fib_values = vec![BaseField::one(), BaseField::one()];
    for _ in 0..(1 << log_n_rows) {
        fib_values.push(fib_values[fib_values.len() - 1] + fib_values[fib_values.len() - 2]);
    }
    let range = 0..(1 << log_n_rows);
    let mut circuit = PlonkWithoutAcceleratorCircuitTrace {
        mult_c: range.clone().map(|_| M31::from(2).neg()).collect(),
        a_wire: range.clone().map(|i| i.into()).collect(),
        b_wire: range.clone().map(|i| (i + 1).into()).collect(),
        c_wire: range.clone().map(|i| (i + 2).into()).collect(),
        op1: range.clone().map(|_| 1.into()).collect(),
        op2: range.clone().map(|_| 0.into()).collect(),
        op3: range.clone().map(|_| 0.into()).collect(),
        op4: range.clone().map(|_| 0.into()).collect(),
        a_val_0: range.clone().map(|i| fib_values[i]).collect(),
        a_val_1: range.clone().map(|_| 0.into()).collect(),
        a_val_2: range.clone().map(|_| 0.into()).collect(),
        a_val_3: range.clone().map(|_| 0.into()).collect(),
        b_val_0: range.clone().map(|i| fib_values[i + 1]).collect(),
        b_val_1: range.clone().map(|_| 0.into()).collect(),
        b_val_2: range.clone().map(|_| 0.into()).collect(),
        b_val_3: range.clone().map(|_| 0.into()).collect(),
        c_val_0: range.clone().map(|i| fib_values[i + 2]).collect(),
        c_val_1: range.clone().map(|_| 0.into()).collect(),
        c_val_2: range.clone().map(|_| 0.into()).collect(),
        c_val_3: range.clone().map(|_| 0.into()).collect(),
    };
    circuit.mult_c.set((1 << log_n_rows) - 1, 0.into());
    circuit.mult_c.set((1 << log_n_rows) - 2, 1.into());

    prove_plonk_without_accelerator::<Blake2sMerkleChannel>(log_n_rows, config, &circuit)
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::core::air::Component;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fri::FriConfig;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use crate::core::prover::verify;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::examples::plonk_without_poseidon::plonk::{
        prove_fibonacci_plonk_without_accelerator, PlonkWithoutAcceleratorLookupElements,
    };

    #[test_log::test]
    fn test_simd_plonk_without_accelerator_prove() {
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
        let (component, proof) = prove_fibonacci_plonk_without_accelerator(log_n_instances, config);

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
        let lookup_elements = PlonkWithoutAcceleratorLookupElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(&[&component], channel, commitment_scheme, proof).unwrap();
    }
}
