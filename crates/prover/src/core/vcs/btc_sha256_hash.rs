use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::Digest;

// Wrapper for the sha256 hash type.
#[repr(align(32))]
#[derive(Clone, Copy, PartialEq, Default, Eq, Deserialize, Serialize)]
pub struct BTCSha256Hash(pub(crate) [u8; 32]);

impl From<BTCSha256Hash> for Vec<u8> {
    fn from(value: BTCSha256Hash) -> Self {
        Vec::from(value.0)
    }
}

impl From<Vec<u8>> for BTCSha256Hash {
    fn from(value: Vec<u8>) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting Vec<u8> to Sha256Hash type"),
        )
    }
}

impl From<&[u8]> for BTCSha256Hash {
    fn from(value: &[u8]) -> Self {
        Self(
            value
                .try_into()
                .expect("Failed converting &[u8] to Sha256Hash type"),
        )
    }
}

impl AsRef<[u8]> for BTCSha256Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<BTCSha256Hash> for [u8; 32] {
    fn from(val: BTCSha256Hash) -> Self {
        val.0
    }
}

impl fmt::Display for BTCSha256Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for BTCSha256Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <BTCSha256Hash as fmt::Display>::fmt(self, f)
    }
}

impl super::hash::Hash for BTCSha256Hash {}

// Wrapper for the sha256 Hashing functionalities.
#[derive(Clone, Debug, Default)]
pub struct BTCSha256Hasher {
    state: sha2::Sha256,
}

impl BTCSha256Hasher {
    pub fn new() -> Self {
        Self {
            state: sha2::Sha256::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.state.update(data);
    }

    pub fn finalize(self) -> BTCSha256Hash {
        BTCSha256Hash(self.state.finalize().into())
    }

    pub fn concat_and_hash(v1: &BTCSha256Hash, v2: &BTCSha256Hash) -> BTCSha256Hash {
        let mut hasher = Self::new();
        hasher.update(v1.as_ref());
        hasher.update(v2.as_ref());
        hasher.finalize()
    }

    pub fn hash(data: &[u8]) -> BTCSha256Hash {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

#[cfg(test)]
impl BTCSha256Hasher {
    fn finalize_reset(&mut self) -> BTCSha256Hash {
        BTCSha256Hash(self.state.finalize_reset().into())
    }
}

#[cfg(test)]
mod tests {
    use super::BTCSha256Hasher;

    #[test]
    fn single_hash_test() {
        let hash_a = BTCSha256Hasher::hash(b"a");
        assert_eq!(
            hash_a.to_string(),
            "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
        );
    }

    #[test]
    fn hash_state_test() {
        let mut state = BTCSha256Hasher::new();
        state.update(b"a");
        state.update(b"b");
        let hash = state.finalize_reset();
        let hash_empty = state.finalize();

        assert_eq!(hash.to_string(), BTCSha256Hasher::hash(b"ab").to_string());
        assert_eq!(hash_empty.to_string(), BTCSha256Hasher::hash(b"").to_string());
    }
}
