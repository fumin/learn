package bn128

import (
	"math/big"

	bn256 "github.com/ethereum/go-ethereum/crypto/bn256/cloudflare"
)

func G1Int(e *bn256.G1) []*big.Int {
	bs := e.Marshal()

	is := make([]*big.Int, 2)
	for i := range is {
		b := bs[i*32 : (i+1)*32]
		is[i] = new(big.Int).SetBytes(b)
	}
	return is
}

func G2Int(e *bn256.G2) []*big.Int {
	bs := e.Marshal()

	is := make([]*big.Int, 4)
	for i := range is {
		b := bs[i*32 : (i+1)*32]
		is[i] = new(big.Int).SetBytes(b)
	}

	// Reorder to py_ecc convertion.
	is[0], is[1] = is[1], is[0]
	is[2], is[3] = is[3], is[2]
	return is
}

// GTInt converts e to the flat coefficients of bn128's Fp12, i.e. the
// coefficients of 1, w, w^2, ..., w^11 where w^12-18w^6+82=0, matching the
// convention used by py_ecc.
//
// e.Marshal lays out its coefficients in cloudflare's tower basis
// {ωτ², ωτ, ω, τ², τ, 1} over Fp2=Fp[i] (i²=-1), with τ³=ξ=i+9 and ω²=τ.
// Each Fp2 coefficient is itself marshaled as (imaginary, real), since
// cloudflare's gfP2{x,y} represents the value x*i+y.
func GTInt(e *bn256.GT) []*big.Int {
	bs := e.Marshal()

	g := make([]*big.Int, 12)
	for i := range g {
		b := bs[i*32 : (i+1)*32]
		g[i] = new(big.Int).SetBytes(b)
	}

	// lo[k]/hi[k] are the w-exponents of the real/imaginary parts of the
	// k'th tower digit (ωτ², ωτ, ω, τ², τ, 1, in marshal order).
	lo := []int{5, 3, 1, 4, 2, 0}
	hi := []int{11, 9, 7, 10, 8, 6}

	c := make([]*big.Int, 12)
	for i := range c {
		c[i] = big.NewInt(0)
	}
	nine := big.NewInt(9)
	for k := range lo {
		img, ral := g[2*k], g[2*k+1]

		re := new(big.Int).Mul(nine, img)
		re.Sub(ral, re)
		re.Mod(re, fieldChar)
		c[lo[k]] = re

		c[hi[k]] = new(big.Int).Mod(img, fieldChar)
	}
	return c
}

func NewG1(is []*big.Int) *bn256.G1 {
	bs := make([]byte, 32*2)
	for i := range is {
		b := bs[i*32 : (i+1)*32]
		is[i].FillBytes(b)
	}
	e := new(bn256.G1)
	if _, err := e.Unmarshal(bs); err != nil {
		panic(err)
	}
	return e
}

func NewG2(is []*big.Int) *bn256.G2 {
	loc := []int{1, 0, 3, 2}
	bs := make([]byte, 32*4)
	for i := range is {
		b := bs[i*32 : (i+1)*32]
		is[loc[i]].FillBytes(b)
	}
	e := new(bn256.G2)
	if _, err := e.Unmarshal(bs); err != nil {
		panic(err)
	}
	return e
}

// NewGT returns the element GT represented by the coefficients of
// 1, w, w^2, ..., w^11.
func NewGT(is []*big.Int) *bn256.GT {
	lo := []int{5, 3, 1, 4, 2, 0}
	hi := []int{11, 9, 7, 10, 8, 6}

	nine := big.NewInt(9)
	bs := make([]byte, 32*12)
	for k := range lo {
		img := new(big.Int).Mod(is[hi[k]], fieldChar)
		ral := new(big.Int).Mul(nine, img)
		ral.Add(ral, is[lo[k]])
		ral.Mod(ral, fieldChar)

		img.FillBytes(bs[2*k*32 : (2*k+1)*32])
		ral.FillBytes(bs[(2*k+1)*32 : (2*k+2)*32])
	}

	e := new(bn256.GT)
	if _, err := e.Unmarshal(bs); err != nil {
		panic(err)
	}
	return e
}
