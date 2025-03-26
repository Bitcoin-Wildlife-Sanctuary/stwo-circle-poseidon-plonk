use blake2::Digest;
use serde::{Deserialize, Serialize};

use crate::core::channel::{MerkleChannel, Sha256Poseidon31Channel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::bitcoin_num_to_bytes;
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::poseidon31_merkle::Poseidon31MerkleHasher;
use crate::core::vcs::sha256_hash::{Sha256Hash, Sha256Hasher};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Sha256Poseidon31MerkleHasher;

impl MerkleHasher for Sha256Poseidon31MerkleHasher {
    type Hash = Sha256Hash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        // There are three possibilities:
        // - children only
        // - children and column elements
        // - column elements only
        //
        // They are handled as follows.
        // - H(left | right)
        // - H(H(left | right) | [column hash])
        // - [column hash]

        let hash_tree = if let Some(children_hashes) = children_hashes {
            let mut hash = [0u8; 32];
            let mut sha256 = sha2::Sha256::new();
            Digest::update(&mut sha256, children_hashes.0);
            Digest::update(&mut sha256, children_hashes.1);
            hash.copy_from_slice(sha256.finalize().as_slice());
            Some(hash)
        } else {
            None
        };

        if column_values.is_empty() {
            assert!(hash_tree.is_some(), "hash_node must not be empty");
            Sha256Hash(hash_tree.unwrap())
        } else {
            if column_values.len() <= 8 {
                let mut hash = [0u8; 32];
                if hash_tree.is_some() {
                    let mut sha256 = sha2::Sha256::new();
                    Digest::update(&mut sha256, bitcoin_num_to_bytes(column_values[0]));
                    Digest::update(&mut sha256, hash_tree.unwrap());
                    hash.copy_from_slice(sha256.finalize().as_slice());
                } else {
                    let mut sha256 = sha2::Sha256::new();
                    Digest::update(&mut sha256, bitcoin_num_to_bytes(column_values[0]));
                    hash.copy_from_slice(sha256.finalize().as_slice());
                };
                for i in 1..column_values.len() {
                    let mut sha256 = sha2::Sha256::new();
                    Digest::update(&mut sha256, bitcoin_num_to_bytes(column_values[i]));
                    Digest::update(&mut sha256, &hash);
                    hash.copy_from_slice(sha256.finalize().as_slice());
                }
                Sha256Hash(hash)
            } else {
                let mut hash = [0u8; 32];
                let data = Poseidon31MerkleHasher::hash_column_get_rate(column_values);
                if hash_tree.is_some() {
                    let mut sha256 = sha2::Sha256::new();
                    Digest::update(&mut sha256, bitcoin_num_to_bytes(data.0[0]));
                    Digest::update(&mut sha256, hash_tree.unwrap());
                    hash.copy_from_slice(sha256.finalize().as_slice());

                    for i in 1..8 {
                        let mut sha256 = sha2::Sha256::new();
                        Digest::update(&mut sha256, bitcoin_num_to_bytes(data.0[i]));
                        Digest::update(&mut sha256, &hash);
                        hash.copy_from_slice(sha256.finalize().as_slice());
                    }
                } else {
                    let mut sha256 = sha2::Sha256::new();
                    Digest::update(&mut sha256, bitcoin_num_to_bytes(data.0[0]));
                    hash.copy_from_slice(sha256.finalize().as_slice());

                    for i in 1..8 {
                        let mut sha256 = sha2::Sha256::new();
                        Digest::update(&mut sha256, bitcoin_num_to_bytes(data.0[i]));
                        Digest::update(&mut sha256, &hash);
                        hash.copy_from_slice(sha256.finalize().as_slice());
                    }
                }
                Sha256Hash(hash)
            }
        }
    }
}

#[derive(Default)]
pub struct Sha256Poseidon31MerkleChannel;

impl MerkleChannel for Sha256Poseidon31MerkleChannel {
    type C = Sha256Poseidon31Channel;
    type H = Sha256Poseidon31MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.update_digest(Sha256Hasher::concat_and_hash(&channel.digest(), &root));
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::channel::{MerkleChannel, Sha256Poseidon31Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::sha256_hash::Sha256Hash;
    use crate::core::vcs::sha256_poseidon31_merkle::{
        Sha256Poseidon31MerkleChannel, Sha256Poseidon31MerkleHasher,
    };
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();
        decommitment.hash_witness[4] = Sha256Hash::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();
        decommitment.hash_witness.push(Sha256Hash::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();
        values.insert(3, BaseField::zero());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooManyQueriedValues
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();
        values.remove(3);

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooFewQueriedValues
        );
    }

    #[test]
    fn test_merkle_channel() {
        let mut channel = Sha256Poseidon31Channel::default();
        let (_queries, _decommitment, _values, verifier) =
            prepare_merkle::<Sha256Poseidon31MerkleHasher>();
        Sha256Poseidon31MerkleChannel::mix_root(&mut channel, verifier.root);
        assert_eq!(channel.inner.channel_time.n_challenges, 1);
    }
}
