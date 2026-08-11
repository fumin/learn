package bn128

import (
	"math/big"

	"github.com/fumin/nag/field"
)

// A FiniteExt is a finite field extension.
type FiniteExt[K field.Finite[K]] field.Extension[K]

// NewZero returns the additive identity 0.
func (x *FiniteExt[K]) NewZero() *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(x).NewZero())
}

// NewOne returns the multiplicative identity 1.
func (x *FiniteExt[K]) NewOne() *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(x).NewOne())
}

// Equal reports whether x and y are equal.
func (x *FiniteExt[K]) Equal(y *FiniteExt[K]) bool {
	return (*field.Extension[K])(x).Equal((*field.Extension[K])(y))
}

// Set sets x to y.
func (x *FiniteExt[K]) Set(y *FiniteExt[K]) *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(x).Set((*field.Extension[K])(y)))
}

// Add sets z to the sum x+y and returns z.
func (z *FiniteExt[K]) Add(x, y *FiniteExt[K]) *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(z).Add((*field.Extension[K])(x), (*field.Extension[K])(y)))
}

// Sub sets z to the difference x-y and returns z.
func (z *FiniteExt[K]) Sub(x, y *FiniteExt[K]) *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(z).Sub((*field.Extension[K])(x), (*field.Extension[K])(y)))
}

// Mul sets z to the product x*y and returns z.
func (z *FiniteExt[K]) Mul(x, y *FiniteExt[K]) *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(z).Mul((*field.Extension[K])(x), (*field.Extension[K])(y)))
}

// Div sets z to the quotient x/y and returns z.
func (z *FiniteExt[K]) Div(x, y *FiniteExt[K]) *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(z).Div((*field.Extension[K])(x), (*field.Extension[K])(y)))
}

// Inv sets x to 1/y and returns x.
func (x *FiniteExt[K]) Inv(y *FiniteExt[K]) *FiniteExt[K] {
	return (*FiniteExt[K])((*field.Extension[K])(x).Inv((*field.Extension[K])(y)))
}

// String returns the coefficient representation of x.
func (x *FiniteExt[K]) String() string {
	return (*field.Extension[K])(x).String()
}

// SetCoeffs sets the coefficients of the polynomial representation of x.
func (x *FiniteExt[K]) SetCoeffs(cs ...*big.Int) *FiniteExt[K] {
	k := x.Poly.Field()
	coeffs := make([]K, len(cs))
	for i := range coeffs {
		coeffs[i] = k.NewZero()
		coeffs[i].SetCoeffs(cs[i])
	}
	return (*FiniteExt[K])((*field.Extension[K])(x).SetCoeffs(coeffs...))
}

// Coeffs returns the coefficients of x's polynomial representation.
func (x *FiniteExt[K]) Coeffs() []*big.Int {
	coeffs := (*field.Extension[K])(x).Coeffs()
	cs := make([]*big.Int, len(coeffs))
	for i := range cs {
		base := coeffs[i].Coeffs()
		if len(base) != 1 {
			panic(base)
		}
		cs[i] = new(big.Int).Set(base[0])
	}
	return cs
}

// Degree returns the [degree] of the field extension.
//
// [degree]: https://en.wikipedia.org/wiki/Degree_of_a_field_extension
func (x *FiniteExt[K]) Degree() int {
	return (*field.Extension[K])(x).Degree()
}

// Characteristic returns the [characteristic] of the field.
//
// [characteristic]: https://en.wikipedia.org/wiki/Characteristic_(algebra)
func (x *FiniteExt[K]) Characteristic() *big.Int {
	return x.Irr.LeadingTerm().Coefficient.Characteristic()
}
