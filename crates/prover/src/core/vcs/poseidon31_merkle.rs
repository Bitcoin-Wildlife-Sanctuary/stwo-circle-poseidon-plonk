use std::cmp::min;

use num_traits::Zero;
use serde::{Deserialize, Serialize};

use crate::core::channel::{MerkleChannel, Poseidon31Channel};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::poseidon31_hash::Poseidon31Hash;
use crate::core::vcs::poseidon31_ref::{poseidon2_permute, Poseidon31CRH};

const ELEMENTS_IN_BLOCK: usize = 8;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Poseidon31MerkleHasher;

impl Poseidon31MerkleHasher {
    pub fn hash_column(column_values: &[BaseField]) -> Poseidon31Hash {
        assert!(!column_values.is_empty());

        let zero = M31::zero();
        let len = column_values.len();
        let num_chunk = len.div_ceil(ELEMENTS_IN_BLOCK);

        if num_chunk == 1 {
            let mut res = [zero; 16];
            for i in 0..min(8, len) {
                res[i] = column_values[i];
            }
            Poseidon31Hash(Poseidon31CRH::permute_get_rate(&res))
        } else {
            let mut res = [zero; 16];
            for i in 0..8 {
                res[i] = column_values[i];
            }
            let mut digest = Poseidon31CRH::permute_get_capacity(&res);

            for chunk in column_values
                .chunks_exact(ELEMENTS_IN_BLOCK)
                .skip(1)
                .take(num_chunk - 2)
            {
                let mut state = [zero; 16];
                state[..8].copy_from_slice(chunk);
                state[8..16].copy_from_slice(&digest);
                digest = Poseidon31CRH::permute_get_capacity(&state);
            }

            let remain = len % ELEMENTS_IN_BLOCK;
            if remain == 0 {
                let mut state = [zero; 16];
                state[..8].copy_from_slice(&column_values[len - 8..]);
                state[8..16].copy_from_slice(&digest);
                digest = Poseidon31CRH::permute_get_rate(&state);
            } else {
                let mut state = [zero; 16];
                state[..remain].copy_from_slice(&column_values[len - remain..]);
                state[8..16].copy_from_slice(&digest);
                digest = Poseidon31CRH::permute_get_rate(&state);
            }
            Poseidon31Hash(digest)
        }
    }
}

impl MerkleHasher for Poseidon31MerkleHasher {
    type Hash = Poseidon31Hash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let zero = M31::zero();

        if children_hashes.is_some() && column_values.is_empty() {
            let (left, right) = children_hashes.unwrap();
            let mut res = [zero; 16];
            for i in 0..ELEMENTS_IN_BLOCK {
                res[i] = left.0[i];
                res[i + ELEMENTS_IN_BLOCK] = right.0[i];
            }
            Poseidon31Hash(Poseidon31CRH::permute_get_rate(&res))
        } else if children_hashes.is_none() && !column_values.is_empty() {
            Self::hash_column(column_values)
        } else if children_hashes.is_none() && column_values.is_empty() {
            Poseidon31Hash([zero; 8])
        } else {
            let hash_tree = {
                let (left, right) = children_hashes.unwrap();
                let mut res = [zero; 16];
                for i in 0..ELEMENTS_IN_BLOCK {
                    res[i] = left.0[i];
                    res[i + ELEMENTS_IN_BLOCK] = right.0[i];
                }
                Poseidon31Hash(Poseidon31CRH::permute_get_capacity(&res))
            };
            let hash_column = Self::hash_column(column_values);
            let mut res = [zero; 16];
            for i in 0..ELEMENTS_IN_BLOCK {
                res[i] = hash_column.0[i];
                res[i + ELEMENTS_IN_BLOCK] = hash_tree.0[i];
            }
            Poseidon31Hash(Poseidon31CRH::permute_get_rate(&res))
        }
    }
}

#[derive(Default)]
pub struct Poseidon31MerkleChannel;

impl MerkleChannel for Poseidon31MerkleChannel {
    type C = Poseidon31Channel;
    type H = Poseidon31MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        let channel_digest = channel.digest();
        let mut state = [
            root.0[0],
            root.0[1],
            root.0[2],
            root.0[3],
            root.0[4],
            root.0[5],
            root.0[6],
            root.0[7],
            channel_digest[0],
            channel_digest[1],
            channel_digest[2],
            channel_digest[3],
            channel_digest[4],
            channel_digest[5],
            channel_digest[6],
            channel_digest[7],
        ];
        poseidon2_permute(&mut state);

        let new_digest = state.last_chunk::<8>().unwrap();
        channel.update_digest(*new_digest);
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::channel::{MerkleChannel, Poseidon31Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::poseidon31_hash::Poseidon31Hash;
    use crate::core::vcs::poseidon31_merkle::{Poseidon31MerkleChannel, Poseidon31MerkleHasher};
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Poseidon31MerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon31MerkleHasher>();
        decommitment.hash_witness[4] = Poseidon31Hash::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon31MerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon31MerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon31MerkleHasher>();
        decommitment.hash_witness.push(Poseidon31Hash::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon31MerkleHasher>();
        values.insert(3, BaseField::zero());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooManyQueriedValues
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon31MerkleHasher>();
        values.remove(3);

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooFewQueriedValues
        );
    }

    #[test]
    fn test_merkle_channel() {
        let mut channel = Poseidon31Channel::default();
        let (_queries, _decommitment, _values, verifier) =
            prepare_merkle::<Poseidon31MerkleHasher>();
        Poseidon31MerkleChannel::mix_root(&mut channel, verifier.root);
        assert_eq!(channel.channel_time.n_challenges, 1);
    }
}
