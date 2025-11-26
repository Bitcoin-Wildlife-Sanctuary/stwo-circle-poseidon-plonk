#![feature(portable_simd, iter_array_chunks, array_chunks)]
pub mod blake;
pub mod plonk;
pub mod poseidon;
pub mod state_machine;
pub mod wide_fibonacci;
pub mod xor;

pub mod plonk_with_poseidon;
pub mod plonk_without_poseidon;
