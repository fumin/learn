package zk

import (
	"math/big"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/fumin/learn/zk/circom"
)

func mulWitness(witness []*big.Int, m circom.Matrix) []fr.Element {
	wmI := new(big.Int)
	wm := new(fr.Element)

	numConstraints := m.Rows
	evals := make([]fr.Element, numConstraints)
	for _, e := range m.COO {
		wmI.Mul(witness[e.Col], e.Val)
		wm.SetBigInt(wmI)
		evals[e.Row].Add(&evals[e.Row], wm)
	}
	return evals
}

// frToBigInts converts a slice of fr.Element coefficients, as returned by
// groth16.ComputeH, back to *big.Int, undoing the bit-reversal
// groth16.ComputeH leaves its output in.
func frToBigInts(xs []fr.Element) []*big.Int {
	out := make([]*big.Int, len(xs))
	for i := range xs {
		out[i] = new(big.Int)
		xs[i].BigInt(out[i])
	}
	bitReverse(out)
	return out
}

// bitReverse permutes s in place so that s[i] and s[reverseBits(i)] are
// swapped, where reverseBits reverses the log2(len(s))-bit binary
// representation of i. len(s) must be a power of two.
func bitReverse[T any](s []T) {
	n := len(s)
	if n <= 1 {
		return
	}
	numBits := bits.Len(uint(n)) - 1
	for i := range s {
		j := bits.Reverse(uint(i)) >> (bits.UintSize - numBits)
		if j > uint(i) {
			s[i], s[j] = s[j], s[i]
		}
	}
}

// lagrangeCoeffsAtTau returns [L_0(tau), L_1(tau), ..., L_{n-1}(tau)], the
// Lagrange basis polynomials for the n-th roots-of-unity domain, evaluated at tau.
// https://github.com/Consensys-Incorporated/gnark/blob/fd5c2443d59970eb1c3e4202fb8f10a23ef60632/backend/groth16/bn254/mpcsetup/phase2.go#L158
func lagrangeCoeffsAtTau(tau *big.Int, domain *fft.Domain) []*big.Int {
	powers := make([]fr.Element, domain.Cardinality)
	pw := big.NewInt(1)
	for i := range powers {
		powers[i].SetBigInt(pw)
		pw.Mul(pw, tau)
		pw.Mod(pw, bn254.ID.ScalarField())
	}

	// Unlike gnark's bespoke difFFTG1 (used by lagrangeCoeffsG1), which
	// leaves the CardinalityInv scaling to the caller, domain.FFTInverse
	// already applies it internally.
	domain.FFTInverse(powers, fft.DIF)
	bitReverse(powers)

	coeffs := make([]*big.Int, domain.Cardinality)
	for i := range coeffs {
		coeffs[i] = new(big.Int)
		powers[i].BigInt(coeffs[i])
	}
	return coeffs
}
