package gfp

import (
	"math/big"
)

// A Gfp is an element of the base field of the bn128 elliptic curve.
type Gfp gfP

func NewGfp(i int64) *Gfp {
	return (*Gfp)(newGFp(i))
}

func (x *Gfp) NewZero() *Gfp {
	return (*Gfp)(newGFp(0))
}

func (x *Gfp) NewOne() *Gfp {
	return (*Gfp)(newGFp(1))
}

func (x *Gfp) Equal(y *Gfp) bool {
	return *x == *y
}

func (x *Gfp) Set(y *Gfp) *Gfp {
	*x = *y
	return x
}

func (x *Gfp) Add(a, b *Gfp) *Gfp {
	gfpAdd((*gfP)(x), (*gfP)(a), (*gfP)(b))
	return x
}

func (x *Gfp) Sub(a, b *Gfp) *Gfp {
	gfpSub((*gfP)(x), (*gfP)(a), (*gfP)(b))
	return x
}

func (x *Gfp) Mul(a, b *Gfp) *Gfp {
	gfpMul((*gfP)(x), (*gfP)(a), (*gfP)(b))
	return x
}

func (x *Gfp) Inv(a *Gfp) *Gfp {
	(*gfP)(x).Invert((*gfP)(a))
	return x
}

func (x *Gfp) Div(a, b *Gfp) *Gfp {
	// Compute the inverse into a fresh element rather than the receiver,
	// since a (or b) may alias x, e.g. in the common in-place idiom x.Div(x, y).
	inv := x.NewOne()
	inv.Inv(b)
	return x.Mul(a, inv)
}

func (x *Gfp) String() string {
	return x.Coeffs()[0].String()
}

// SetCoeffs sets x to coefficients[0] modulo the field characteristic. As Gfp
// has degree 1, only a single coefficient is meaningful; it defaults to 0.
func (x *Gfp) SetCoeffs(coefficients ...*big.Int) *Gfp {
	c := new(big.Int)
	if len(coefficients) > 0 {
		c.Mod(coefficients[0], x.Characteristic())
	}

	buf := make([]byte, 32)
	c.FillBytes(buf)
	if err := (*gfP)(x).Unmarshal(buf); err != nil {
		panic(err)
	}
	montEncode((*gfP)(x), (*gfP)(x))
	return x
}

// Coeffs returns the single coefficient representing x.
func (x *Gfp) Coeffs() []*big.Int {
	decoded := &gfP{}
	montDecode(decoded, (*gfP)(x))
	buf := make([]byte, 32)
	decoded.Marshal(buf)
	return []*big.Int{new(big.Int).SetBytes(buf)}
}

func (x *Gfp) Degree() int { return 1 }

func (x *Gfp) Characteristic() *big.Int {
	return new(big.Int).Set(P)
}
