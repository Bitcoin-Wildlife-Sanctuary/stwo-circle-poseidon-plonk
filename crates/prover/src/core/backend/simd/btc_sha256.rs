use itertools::Itertools;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::btc_sha256_hash::BTCSha256Hash;
use crate::core::vcs::btc_sha256_merkle::BTCSha256MerkleHasher;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::parallel_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl ColumnOps<BTCSha256Hash> for SimdBackend {
    type Column = Vec<BTCSha256Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<BTCSha256MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<BTCSha256Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<BTCSha256Hash> {
        parallel_iter!(0..1 << log_size)
            .map(|i| {
                BTCSha256MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.at(i)).collect_vec(),
                )
            })
            .collect()
    }
}
