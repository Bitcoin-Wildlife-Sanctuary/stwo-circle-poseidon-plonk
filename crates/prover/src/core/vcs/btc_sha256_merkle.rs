use serde::{Deserialize, Serialize};
use sha2::Digest;
use crate::core::channel::{BTCSha256Channel, MerkleChannel};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::btc_sha256_hash::{BTCSha256Hash, BTCSha256Hasher};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct BTCSha256MerkleHasher;
impl MerkleHasher for BTCSha256MerkleHasher {
    type Hash = BTCSha256Hash;

    fn hash_node(children_hashes: Option<(Self::Hash, Self::Hash)>, column_values: &[BaseField]) -> Self::Hash {
        // There are three possibilities:
        // - children only
        // - children and column elements
        // - column elements only
        //
        // They are handled as follows.
        // - H(left | right) 
        // - H(H(left | right) | [column hash])
        // - [column hash]

        let hash_column = if column_values.is_empty() {
            None
        } else {
            let len = column_values.len();

            let mut hash = [0u8; 32];
            let mut sha256 = sha2::Sha256::new();
            Digest::update(&mut sha256, bitcoin_num_to_bytes(column_values[len - 1]));
            hash.copy_from_slice(sha256.finalize().as_slice());

            for i in 1..len {
                let mut sha256 = sha2::Sha256::new();
                Digest::update(&mut sha256, bitcoin_num_to_bytes(column_values[len - 1 - i]));
                Digest::update(&mut sha256, hash);
                hash.copy_from_slice(sha256.finalize().as_slice());
            }

            Some(hash)
        };
        
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

        let hash_result = match (hash_tree, hash_column) {
            (Some(hash_tree), Some(hash_column)) => {
                let mut sha256 = sha2::Sha256::new();
                Digest::update(&mut sha256, hash_tree);
                Digest::update(&mut sha256, hash_column);
                BTCSha256Hash::from(sha256.finalize().as_slice())
            },
            (Some(hash_tree), None) => {
                BTCSha256Hash(hash_tree)
            },
            (None, Some(hash_column)) => {
                BTCSha256Hash(hash_column)
            },
            (None, None) => unreachable!(),
        };
        hash_result
    }
}

pub fn bitcoin_num_to_bytes(v: M31) -> Vec<u8> {
    let mut bytes = Vec::new();

    let mut v = v.0;
    while v > 0 {
        bytes.push((v & 0xff) as u8);
        v >>= 8;
    }

    if bytes.last().is_some() && bytes.last().unwrap() & 0x80 != 0 {
        bytes.push(0);
    }

    bytes
}

#[derive(Default)]
pub struct BTCSha256MerkleChannel;

impl MerkleChannel for BTCSha256MerkleChannel {
    type C = BTCSha256Channel;
    type H = BTCSha256MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.update_digest(BTCSha256Hasher::concat_and_hash(
            &channel.digest(),
            &root,
        ));
    }
}


#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::channel::{BTCSha256Channel, MerkleChannel};
    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::btc_sha256_hash::BTCSha256Hash;
    use crate::core::vcs::btc_sha256_merkle::{BTCSha256MerkleChannel, BTCSha256MerkleHasher};
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();
        decommitment.hash_witness[4] = BTCSha256Hash::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();
        decommitment.hash_witness.push(BTCSha256Hash::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();
        values.insert(3, BaseField::zero());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooManyQueriedValues
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();
        values.remove(3);

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooFewQueriedValues
        );
    }

    #[test]
    fn test_merkle_channel() {
        let mut channel = BTCSha256Channel::default();
        let (_queries, _decommitment, _values, verifier) = prepare_merkle::<BTCSha256MerkleHasher>();
        BTCSha256MerkleChannel::mix_root(&mut channel, verifier.root);
        assert_eq!(channel.channel_time.n_challenges, 1);
    }
}
