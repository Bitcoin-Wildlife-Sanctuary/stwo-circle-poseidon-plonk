use std::ops::Neg;

use itertools::Itertools;
use num_traits::One;
use tracing::{span, Level};

use stwo_constraint_framework::logup::LogupTraceGenerator;
use stwo_constraint_framework::{
    assert_constraints_on_polys, EvalAtRow, FrameworkComponent, FrameworkEval, Relation,
    RelationEntry, TraceLocationAllocator,
};
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo_prover::core::backend::simd::qm31::PackedSecureField;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::{BackendForChannel, Column};
use stwo_prover::core::channel::MerkleChannel;
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::pcs::{CommitmentSchemeProver, PcsConfig};
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::prover::{prove, StarkProof};
use stwo_prover::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo_prover::core::ColumnVec;
use crate::plonk::Plonk;
use stwo_constraint_framework::relation;

pub type PlonkWithAcceleratorComponent = FrameworkComponent<PlonkWithAcceleratorEval>;
relation!(PlonkWithAcceleratorLookupElements, 3);

#[derive(Clone)]
pub struct PlonkWithAcceleratorEval {
    pub log_n_rows: u32,
    pub lookup_elements: PlonkWithAcceleratorLookupElements,
    pub total_sum: SecureField,
}

impl FrameworkEval for PlonkWithAcceleratorEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let a_wire = eval.get_preprocessed_column(Plonk::new("a_wire".to_string()).id());
        let b_wire = eval.get_preprocessed_column(Plonk::new("b_wire".to_string()).id());
        // Note: c_wire could also be implicit: (self.eval.point() - M31_CIRCLE_GEN.into_ef()).x.
        //   A constant column is easier though.
        let c_wire = eval.get_preprocessed_column(Plonk::new("c_wire".to_string()).id());
        let op = eval.get_preprocessed_column(Plonk::new("op".to_string()).id());
        let mult_a = eval.get_preprocessed_column(Plonk::new("mult_a".to_string()).id());
        let mult_b = eval.get_preprocessed_column(Plonk::new("mult_b".to_string()).id());
        let mult_c = eval.get_preprocessed_column(Plonk::new("mult_c".to_string()).id());
        let poseidon_wire =
            eval.get_preprocessed_column(Plonk::new("poseidon_wire".to_string()).id());
        let mult_poseidon =
            eval.get_preprocessed_column(Plonk::new("mult_poseidon".to_string()).id());
        let enforce_c_m31 =
            eval.get_preprocessed_column(Plonk::new("enforce_c_m31".to_string()).id());

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

        eval.add_constraint(enforce_c_m31.clone() * c_val_1.clone());
        eval.add_constraint(enforce_c_m31.clone() * c_val_2.clone());
        eval.add_constraint(enforce_c_m31.clone() * c_val_3.clone());

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
            c_val.clone()
                - E::EF::from(op.clone()) * (a_val.clone() + b_val.clone())
                - E::EF::from(E::F::one() - op) * a_val.clone() * b_val.clone(),
        );

        eval.add_to_relation_ef(RelationEntry::new(
            &self.lookup_elements,
            mult_a.into(),
            &[a_val.clone(), E::EF::from(a_wire.clone())],
        ));
        eval.add_to_relation_ef(RelationEntry::new(
            &self.lookup_elements,
            mult_b.into(),
            &[b_val.clone(), E::EF::from(b_wire.clone())],
        ));

        eval.add_to_relation_ef(RelationEntry::new(
            &self.lookup_elements,
            mult_c.into(),
            &[c_val.clone(), E::EF::from(c_wire.clone())],
        ));

        eval.add_to_relation_ef(RelationEntry::new(
            &self.lookup_elements,
            (-mult_poseidon).into(),
            &[
                E::EF::from(poseidon_wire.clone()),
                a_val.clone(),
                b_val.clone(),
            ],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

#[derive(Clone)]
pub struct PlonkWithAcceleratorCircuitTrace {
    pub mult_a: BaseColumn,
    pub mult_b: BaseColumn,
    pub mult_c: BaseColumn,
    pub poseidon_wire: BaseColumn,
    pub mult_poseidon: BaseColumn,
    pub enforce_c_m31: BaseColumn,
    pub a_wire: BaseColumn,
    pub b_wire: BaseColumn,
    pub c_wire: BaseColumn,
    pub op: BaseColumn,
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
    circuit: &PlonkWithAcceleratorCircuitTrace,
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
    circuit: &PlonkWithAcceleratorCircuitTrace,
    lookup_elements: &PlonkWithAcceleratorLookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let p0 = circuit.mult_a.data[vec_row];
        let q0: PackedSecureField = <PlonkWithAcceleratorLookupElements as Relation<
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
        let p1 = circuit.mult_b.data[vec_row];
        let q1: PackedSecureField = <PlonkWithAcceleratorLookupElements as Relation<
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
        col_gen.write_frac(vec_row, q0 * p1 + q1 * p0, q0 * q1);
    }
    col_gen.finalize_col();

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let p0 = circuit.mult_c.data[vec_row];
        let q0: PackedSecureField = <PlonkWithAcceleratorLookupElements as Relation<
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

        let p1 = -circuit.mult_poseidon.data[vec_row];
        let q1: PackedSecureField = <PlonkWithAcceleratorLookupElements as Relation<
            PackedBaseField,
            PackedSecureField,
        >>::combine_ef(
            &lookup_elements,
            &[
                PackedSecureField::from(circuit.poseidon_wire.data[vec_row]),
                PackedSecureField::from_packed_m31s([
                    circuit.a_val_0.data[vec_row],
                    circuit.a_val_1.data[vec_row],
                    circuit.a_val_2.data[vec_row],
                    circuit.a_val_3.data[vec_row],
                ]),
                PackedSecureField::from_packed_m31s([
                    circuit.b_val_0.data[vec_row],
                    circuit.b_val_1.data[vec_row],
                    circuit.b_val_2.data[vec_row],
                    circuit.b_val_3.data[vec_row],
                ]),
            ],
        );
        col_gen.write_frac(vec_row, q0 * p1 + q1 * p0, q0 * q1);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

pub fn prove_plonk_with_accelerator<MC: MerkleChannel>(
    log_n_rows: u32,
    config: PcsConfig,
    circuit: &PlonkWithAcceleratorCircuitTrace,
) -> (PlonkWithAcceleratorComponent, StarkProof<MC::H>)
where
    SimdBackend: BackendForChannel<MC>,
{
    assert!(log_n_rows >= LOG_N_LANES);

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + 1)
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
        circuit.op.clone(),
        circuit.mult_a.clone(),
        circuit.mult_b.clone(),
        circuit.mult_c.clone(),
        circuit.poseidon_wire.clone(),
        circuit.mult_poseidon.clone(),
        circuit.enforce_c_m31.clone(),
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
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, total_sum) = gen_interaction_trace(log_n_rows, &circuit, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();
    // Prove constraints.
    let component = PlonkWithAcceleratorComponent::new(
        &mut TraceLocationAllocator::default(),
        PlonkWithAcceleratorEval {
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

    let component_eval = component.clone();
    assert_constraints_on_polys(
        &trace_polys,
        CanonicCoset::new(log_n_rows),
        |assert_eval| {
            component_eval.evaluate(assert_eval);
        },
        total_sum,
    );

    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

#[allow(unused)]
pub fn prove_fibonacci_plonk_with_accelerator(
    log_n_rows: u32,
    config: PcsConfig,
) -> (
    PlonkWithAcceleratorComponent,
    StarkProof<Blake2sMerkleHasher>,
) {
    assert!(log_n_rows >= LOG_N_LANES);

    // Prepare a fibonacci circuit.
    let mut fib_values = vec![BaseField::one(), BaseField::one()];
    for _ in 0..(1 << log_n_rows) {
        fib_values.push(fib_values[fib_values.len() - 1] + fib_values[fib_values.len() - 2]);
    }
    let range = 0..(1 << log_n_rows);
    let mut circuit = PlonkWithAcceleratorCircuitTrace {
        mult_a: range.clone().map(|_| 1.into()).collect(),
        mult_b: range.clone().map(|_| 1.into()).collect(),
        mult_c: range.clone().map(|_| M31::from(2).neg()).collect(),
        poseidon_wire: range.clone().map(|_| 0.into()).collect(),
        mult_poseidon: range.clone().map(|_| 0.into()).collect(),
        a_wire: range.clone().map(|i| i.into()).collect(),
        b_wire: range.clone().map(|i| (i + 1).into()).collect(),
        c_wire: range.clone().map(|i| (i + 2).into()).collect(),
        op: range.clone().map(|_| 1.into()).collect(),
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
        enforce_c_m31: range.clone().map(|_| 0.into()).collect(),
    };
    circuit.poseidon_wire.set(1, 1.into());
    circuit.mult_poseidon.set(1, 1.into());
    circuit.mult_c.set((1 << log_n_rows) - 1, 0.into());
    circuit.mult_c.set((1 << log_n_rows) - 2, 1.into());

    prove_plonk_with_accelerator::<Blake2sMerkleChannel>(log_n_rows, config, &circuit)
}

#[cfg(test)]
mod tests {
    use std::env;

    use stwo_prover::core::air::Component;
    use stwo_prover::core::channel::Blake2sChannel;
    use stwo_prover::core::fri::FriConfig;
    use stwo_prover::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use stwo_prover::core::prover::verify;
    use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::plonk_with_poseidon::plonk::{
        prove_fibonacci_plonk_with_accelerator, PlonkWithAcceleratorLookupElements,
    };

    #[test_log::test]
    fn test_simd_plonk_with_accelerator_prove() {
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
        let (component, proof) = prove_fibonacci_plonk_with_accelerator(log_n_instances, config);

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
