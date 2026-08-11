package gfp

import "math/big"

// P is a prime over which we form a basic field: 36u⁴+36u³+24u²+6u+1.
var P, _ = new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

// p2 is p, represented as little-endian 64-bit words.
var p2 = [4]uint64{0x3c208c16d87cfd47, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029}

// np is the negative inverse of p, mod 2^256.
var np = [4]uint64{0x87d20782e4866389, 0x9ede7d651eca6ac9, 0xd8afcbd01833da80, 0xf57a22b791888c6b}

// rN1 is R^-1 where R = 2^256 mod p.
var rN1 = &gfP{0xed84884a014afa37, 0xeb2022850278edf8, 0xcf63e9cfb74492d9, 0x2e67157159e5c639}

// r2 is R^2 where R = 2^256 mod p.
var r2 = &gfP{0xf32cfc5b538afa89, 0xb5e71911d44501fb, 0x47ab1eff0a417ff6, 0x06d89f71cab8351f}

// r3 is R^3 where R = 2^256 mod p.
var r3 = &gfP{0xb1cd6dafda1530df, 0x62f210e6a7283db6, 0xef7f0b0c0ada0afb, 0x20fd6e902d592544}
