package ecc

import (
	"fmt"
	"math/big"

	"github.com/fumin/nag"
	"github.com/fumin/nag/field"
)

const (
	symbolX nag.Symbol = 0
	symbolY nag.Symbol = 1
)

type CurvePoint[E any, K nag.Field[K]] interface {
	nag.Group[E]
	IsInfinity() bool
	Coords() []K
	SetCoords([]K) E
}

type ellipticCurve[K nag.Field[K]] struct {
	a       K
	x, y, z K
}

func newEllipticCurve[K nag.Field[K]](a K) *ellipticCurve[K] {
	return &ellipticCurve[K]{a: a.NewZero().Set(a), x: a.NewZero(), y: a.NewOne(), z: a.NewZero()}
}

func (c *ellipticCurve[K]) NewOne() *ellipticCurve[K] {
	k := c.a
	return &ellipticCurve[K]{a: k.NewZero().Set(c.a), x: k.NewZero(), y: k.NewOne(), z: k.NewZero()}
}

func (c *ellipticCurve[K]) Equal(d *ellipticCurve[K]) bool {
	return c.a.Equal(d.a) && c.x.Equal(d.x) && c.y.Equal(d.y) && c.z.Equal(c.z)
}

func (c *ellipticCurve[K]) Set(d *ellipticCurve[K]) *ellipticCurve[K] {
	c.a.Set(d.a)
	c.x.Set(d.x)
	c.y.Set(d.y)
	c.z.Set(d.z)
	return c
}

func (r *ellipticCurve[K]) Mul(p, q *ellipticCurve[K]) *ellipticCurve[K] {
	if p.IsInfinity() {
		return r.Set(q)
	}
	if q.IsInfinity() {
		return r.Set(p)
	}

	k := r.a
	lambda := k.NewZero()
	if q.x.Equal(p.x) {
		negQy := k.NewZero()
		negQy.Sub(negQy, q.y)
		if p.y.Equal(negQy) {
			r.x.Set(k.NewZero())
			r.y.Set(k.NewOne())
			r.z.Set(k.NewZero())
			return r
		}

		xp2 := k.NewZero().Mul(p.x, p.x)
		lambda.Add(lambda, xp2)
		lambda.Add(lambda, xp2)
		lambda.Add(lambda, xp2)
		lambda.Add(lambda, r.a)
		yp2 := k.NewZero().Set(p.y)
		yp2.Add(yp2, yp2)
		lambda.Div(lambda, yp2)
	} else {
		lambda.Sub(q.y, p.y)
		lambda.Div(lambda, k.NewZero().Sub(q.x, p.x))
	}

	px, qx := p.x, q.x
	r.x = k.NewZero()
	r.x.Mul(lambda, lambda)
	r.x.Sub(r.x, px)
	r.x.Sub(r.x, qx)

	py := p.y
	r.y = k.NewZero()
	r.y.Sub(px, r.x)
	r.y.Mul(r.y, lambda)
	r.y.Sub(r.y, py)

	r.z.Set(k.NewOne())
	return r
}

func (c *ellipticCurve[K]) Inv(a *ellipticCurve[K]) *ellipticCurve[K] {
	c.x.Set(a.x)
	c.y.Sub(a.y.NewZero(), a.y)
	c.z.Set(a.z)
	return c
}

func (c *ellipticCurve[K]) String() string {
	return "(" + c.x.String() + ", " + c.y.String() + ", " + c.z.String() + ")"
}

func (c *ellipticCurve[K]) Coords() []K {
	return []K{c.x, c.y}
}

func (c *ellipticCurve[K]) SetCoords(cs []K) *ellipticCurve[K] {
	c.x.Set(cs[0])
	c.y.Set(cs[1])
	c.z.Set(cs[0].NewOne())
	return c
}

func (c *ellipticCurve[K]) IsInfinity() bool {
	return c.z.Equal(c.z.NewZero())
}

// A WeierstrassA0 is an element of the curve y^2 = x^3+ax+b, where a=0.
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
// https://pkg.go.dev/golang.org/x/crypto/bn256
type WeierstrassA0[K nag.Field[K]] struct {
	// In Jacobian coordinates:
	// X = x/z^2
	// Y = y/z^3
	x, y, z K
}

func NewWeierstrassA0[K nag.Field[K]](xy ...K) *WeierstrassA0[K] {
	k := xy[0]
	return &WeierstrassA0[K]{x: k.NewZero().Set(xy[0]), y: k.NewZero().Set(xy[1]), z: k.NewOne()}
}

func (x *WeierstrassA0[K]) NewOne() *WeierstrassA0[K] {
	k := x.x
	infinity := &WeierstrassA0[K]{x: k.NewZero(), y: k.NewOne(), z: k.NewZero()}
	return infinity
}

func (x *WeierstrassA0[K]) Equal(y *WeierstrassA0[K]) bool {
	x.makeAffine()
	y.makeAffine()
	return x.x.Equal(y.x) && x.y.Equal(y.y) && x.z.Equal(y.z)
}

func (z *WeierstrassA0[K]) Set(x *WeierstrassA0[K]) *WeierstrassA0[K] {
	z.x.Set(x.x)
	z.y.Set(x.y)
	z.z.Set(x.z)
	return z
}

func (c *WeierstrassA0[K]) Mul(a, b *WeierstrassA0[K]) *WeierstrassA0[K] {
	// Allocate necessary memory.
	if a == c {
		a = c.NewOne().Set(a)
	}
	if b == c {
		b = c.NewOne().Set(b)
	}
	if a == b {
		b = c.NewOne().Set(b)
	}

	if a.IsInfinity() {
		return c.Set(b)
	}
	if b.IsInfinity() {
		return c.Set(a)
	}

	// See http://hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/addition/add-2007-bl.op3

	// Normalize the points by replacing a = [x1:y1:z1] and b = [x2:y2:z2]
	// by [u1:s1:z1·z2] and [u2:s2:z1·z2]
	// where u1 = x1·z2², s1 = y1·z2³ and u1 = x2·z1², s2 = y2·z1³
	k := c.x
	z1z1 := k.NewZero().Mul(a.z, a.z)
	z2z2 := k.NewZero().Mul(b.z, b.z)
	u1 := k.NewZero().Mul(a.x, z2z2)
	u2 := k.NewZero().Mul(b.x, z1z1)

	t := k.NewZero().Mul(b.z, z2z2)
	s1 := k.NewZero().Mul(a.y, t)

	t.Mul(a.z, z1z1)
	s2 := k.NewZero().Mul(b.y, t)

	// Compute x = (2h)²(s²-u1-u2)
	// where s = (s2-s1)/(u2-u1) is the slope of the line through
	// (u1,s1) and (u2,s2). The extra factor 2h = 2(u2-u1) comes from the value of z below.
	// This is also:
	// 4(s2-s1)² - 4h²(u1+u2) = 4(s2-s1)² - 4h³ - 4h²(2u1)
	//                        = r² - j - 2v
	// with the notations below.
	h := k.NewZero().Sub(u2, u1)
	xEqual := h.Equal(k.NewZero())

	t.Add(h, h)
	// i = 4h²
	i := k.NewZero().Mul(t, t)
	// j = 4h³
	j := k.NewZero().Mul(h, i)

	t.Sub(s2, s1)
	yEqual := t.Equal(k.NewZero())
	if xEqual && yEqual {
		return c.double(a)
	}
	r := k.NewZero().Add(t, t)

	v := k.NewZero().Mul(u1, i)

	// t4 = 4(s2-s1)²
	t4 := k.NewZero().Mul(r, r)
	t.Add(v, v)
	t6 := k.NewZero().Sub(t4, j)
	c.x.Sub(t6, t)

	// Set y = -(2h)³(s1 + s*(x/4h²-u1))
	// This is also
	// y = - 2·s1·j - (s2-s1)(2x - 2i·u1) = r(v-x) - 2·s1·j
	t.Sub(v, c.x)  // t7
	t4.Mul(s1, j)  // t8
	t6.Add(t4, t4) // t9
	t4.Mul(r, t)   // t10
	c.y.Sub(t4, t6)

	// Set z = 2(u2-u1)·z1·z2 = 2h·z1·z2
	t.Add(a.z, b.z) // t11
	t4.Mul(t, t)    // t12
	t.Sub(t4, z1z1) // t13
	t4.Sub(t, z2z2) // t14
	c.z.Mul(t4, h)

	return c
}

func (c *WeierstrassA0[K]) Inv(a *WeierstrassA0[K]) *WeierstrassA0[K] {
	c.x.Set(a.x)
	c.y.Sub(a.y.NewZero(), a.y)
	c.z.Set(a.z)
	return c
}

func (c *WeierstrassA0[K]) String() string {
	c.makeAffine()
	return "(" + c.x.String() + ", " + c.y.String() + ", " + c.z.String() + ")"
}

func (c *WeierstrassA0[K]) Coords() []K {
	c.makeAffine()
	return []K{c.x, c.y}
}

func (c *WeierstrassA0[K]) SetCoords(cs []K) *WeierstrassA0[K] {
	c.x.Set(cs[0])
	c.y.Set(cs[1])
	c.z.Set(cs[0].NewOne())
	return c
}

func (c *WeierstrassA0[K]) double(a *WeierstrassA0[K]) *WeierstrassA0[K] {
	// See http://hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/doubling/dbl-2009-l.op3
	k := c.x
	A := k.NewZero().Mul(a.x, a.x)
	B := k.NewZero().Mul(a.y, a.y)
	C := k.NewZero().Mul(B, B)

	t := k.NewZero().Add(a.x, B)
	t2 := k.NewZero().Mul(t, t)
	t.Sub(t2, A)
	t2.Sub(t, C)
	d := k.NewZero().Add(t2, t2)
	t.Add(A, A)
	e := k.NewZero().Add(t, A)
	f := k.NewZero().Mul(e, e)

	t.Add(d, d)
	c.x.Sub(f, t)

	t.Add(C, C)
	t2.Add(t, t)
	t.Add(t2, t2)
	c.y.Sub(d, c.x)
	t2.Mul(e, c.y)
	c.y.Sub(t2, t)

	t.Mul(a.y, a.z)
	c.z.Add(t, t)

	return c
}

func (c *WeierstrassA0[K]) makeAffine() *WeierstrassA0[K] {
	k := c.z
	if c.z.Equal(k.NewOne()) {
		return c
	}
	if c.IsInfinity() {
		return c.Set(c.NewOne())
	}

	zInv := k.NewZero().Inv(c.z)
	t := k.NewZero().Mul(c.y, zInv)
	zInv2 := k.NewZero().Mul(zInv, zInv)
	c.y.Mul(t, zInv2)
	t.Mul(c.x, zInv2)
	c.x.Set(t)
	c.z.Set(k.NewOne())

	return c
}

func (x *WeierstrassA0[K]) IsInfinity() bool {
	return x.z.Equal(x.z.NewZero())
}

func Coefficients(x *WeierstrassA0[*field.PrimeExt]) [][]*big.Int {
	x.makeAffine()
	cs := make([][]*big.Int, 0)
	cs = append(cs, x.x.Coeffs())
	cs = append(cs, x.y.Coeffs())
	return cs
}

func FrobeniusEndo[K field.Finite[K]](p []K, n int) []K {
	k := p[0].NewZero()
	exponent := new(big.Int)
	fp := make([]K, len(p))
	for i := range fp {
		fp[i] = k.NewZero()
		exponent.Exp(k.Characteristic(), big.NewInt(int64(n)), nil)
		fp[i].Set(nag.Pow(k.Set(p[i]), exponent))
	}
	return fp
}

func Trace[K field.Finite[K], E CurvePoint[E, K]](p E) E {
	k := p.Coords()[0]
	t, frob := p.NewOne(), p.NewOne()
	for i := range k.Degree() {
		frob.SetCoords(FrobeniusEndo(p.Coords(), i))
		t.Mul(t, frob)
	}
	return t
}

func Int10(s string) *big.Int {
	i, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("SetString error")
	}
	return i
}

func NumPoints(qi, efqi, ni int) *big.Int {
	efq, q, n := big.NewInt(int64(efqi)), big.NewInt(int64(qi)), big.NewInt(int64(ni))
	qn := new(big.Int).Exp(q, n, nil)
	qn.Add(qn, big.NewInt(1))
	qn.Sub(qn, numPointsAn(q, efq, n))
	return qn
}

// numPointsAn returns the number of points of an elliptic curve E/F_{q^n}.
// Exercise 5.13, J. H. Silverman. The Arithmetic of Elliptic Curves 2nd Ed.
func numPointsAn(q, efq, n *big.Int) *big.Int {
	switch {
	case n.Cmp(big.NewInt(0)) == 0:
		return big.NewInt(2)
	case n.Cmp(big.NewInt(1)) == 0:
		out := big.NewInt(1)
		out.Add(out, q)
		out.Sub(out, efq)
		return out
	default:
		n1 := new(big.Int).Sub(n, big.NewInt(1))
		a1an1 := new(big.Int).Mul(numPointsAn(q, efq, big.NewInt(1)), numPointsAn(q, efq, n1))
		n2 := new(big.Int).Sub(n, big.NewInt(2))
		qan := new(big.Int).Mul(q, numPointsAn(q, efq, n2))
		return a1an1.Sub(a1an1, qan)
	}
}

// setIth sets x to the i'th element in the field and returns x.
func setIth[K field.Finite[K]](x K, i *big.Int) K {
	i = new(big.Int).Set(i)
	p := x.Characteristic()
	r := new(big.Int)
	coeffs := make([]*big.Int, 0)
	for i.Sign() != 0 {
		i.QuoRem(i, p, r)
		coeffs = append(coeffs, new(big.Int).Set(r))
	}
	if len(coeffs) == 0 {
		coeffs = append(coeffs, big.NewInt(0))
	}
	return x.SetCoeffs(coeffs...)
}

func EvalPoly[K nag.Field[K]](p *nag.Polynomial[K], point []K) K {
	k := p.Field()
	out, term := k.NewZero(), k.NewZero()
	for c, m := range p.Terms() {
		term.Set(c)
		for _, v := range m {
			term.Mul(term, point[v])
		}

		out.Add(out, term)
	}
	return out
}

func ellipticPoly[K nag.Field[K]](a, b K) *nag.Polynomial[K] {
	k := a.NewOne()
	zero, one := k.NewZero(), k.NewOne()
	neg1 := k.NewZero().Sub(zero, one)
	return nag.NewPolynomial(k, nag.ElimOrder(),
		// y^2
		nag.PolynomialTerm[K]{Coefficient: one, Monomial: []nag.Symbol{symbolY, symbolY}},
		// x^3
		nag.PolynomialTerm[K]{Coefficient: neg1, Monomial: make([]nag.Symbol, 3)},
		// a*x
		nag.PolynomialTerm[K]{Coefficient: k.NewZero().Mul(neg1, a), Monomial: make([]nag.Symbol, 1)},
		// b
		nag.PolynomialTerm[K]{Coefficient: k.NewZero().Mul(neg1, b)})
}

func solveElliptic[K field.Finite[K]](poly *nag.Polynomial[K], n int) [][]K {
	k := poly.LeadingTerm().Coefficient
	order := new(big.Int).Exp(k.Characteristic(), big.NewInt(int64(k.Degree())), nil)
	sols := make([][]K, 0)
	for i := big.NewInt(0); i.Cmp(order) < 0; i.Add(i, big.NewInt(1)) {
		x := setIth(k.NewZero(), i)
		y2 := EvalPoly(poly, []K{x})
		y, ok := field.Sqrt(y2)
		if !ok {
			continue
		}

		sols = append(sols, []K{x, y})
		if !y.Equal(k.NewZero()) && k.Characteristic().Cmp(big.NewInt(2)) != 0 {
			yconj := k.NewZero()
			yconj.Sub(yconj, y)
			sols = append(sols, []K{x, yconj})
			if n > 0 && len(sols) >= n {
				break
			}
		}
	}
	return sols
}

func randEllipticPoints[K field.Finite[K]](a, b K, n int) [][]K {
	ep := ellipticPoly(a, b)

	// Remove the y^2.
	k := ep.Field()
	zero, one := k.NewZero(), k.NewOne()
	neg1 := k.NewZero().Sub(zero, one)
	ep.Mul(ep, nag.NewPolynomial(ep.Field(), ep.Order(), nag.PolynomialTerm[K]{Coefficient: neg1}))
	y2 := nag.NewPolynomial(ep.Field(), ep.Order(), nag.PolynomialTerm[K]{Coefficient: k.NewOne(), Monomial: []nag.Symbol{symbolY, symbolY}})
	ep.Add(y2, ep)

	return solveElliptic(ep, n)
}

// FitPoints fits a polynomial that runs through points.
func FitPoints[K nag.Field[K]](points [][]K) *nag.Polynomial[K] {
	k := points[0][0].NewZero()
	// Create the matrix m representing the contraints that the polynomial
	//
	//   y = an*x^n + an-1*x^(n-1) + ... + a0
	//
	// needs to satisfy.
	//
	// an, an-1, ..., a0 satisfies:
	//
	//   points[0].y = an*points[0].x^n + an-1*points[0].x^(n-1) + ... + a0
	//   points[1].y = an*points[1].x^n + an-1*points[1].x^(n-1) + ... + a0
	//   ...
	//
	m := make([][]K, len(points))
	for i := range m {
		x, y := points[i][0], points[i][1]

		m[i] = make([]K, len(points)+1)
		for j := range m[i] {
			m[i][j] = k.NewZero()

			if j == len(points) {
				m[i][j].Set(y)
			} else {
				mij := k.NewZero().Set(x)
				deg := big.NewInt(int64(len(points) - 1 - j))
				m[i][j].Set(nag.Pow(mij, deg))
			}
		}
	}

	nag.GaussElim(m)

	v := nag.NewPolynomial(k, nag.Deglex)
	for i := range m {
		row := m[i]
		c := row[len(row)-1]

		// Check that solution is at degree len(points)-1-i.
		deg := len(points) - 1 - i
		for j := range len(row) - 1 {
			var want K
			if len(points)-1-j == deg {
				want = k.NewOne()
			} else {
				want = k.NewZero()
			}
			if !row[j].Equal(want) {
				panic(fmt.Sprintf("m[%d][%d] = %v want %v %v", i, j, row[j], want, m))
			}
		}

		v.Add(v, nag.NewPolynomial(v.Field(), v.Order(), nag.PolynomialTerm[K]{Coefficient: c, Monomial: make([]nag.Symbol, deg)}))
	}
	return v
}

func linePQ[K nag.Field[K], E CurvePoint[E, K]](p, q E, a K) *nag.Polynomial[K] {
	k := p.Coords()[0].NewZero()
	zero, one, tmp := k.NewZero(), k.NewOne(), k.NewZero()
	// Use the elliptic curve double algorithm.
	if q.Equal(p) {
		x, y := p.Coords()[0], p.Coords()[1]
		x2 := nag.Pow(tmp.Set(x), big.NewInt(2))
		slope := k.NewZero().Set(x2)
		slope.Add(slope, x2)
		slope.Add(slope, x2)
		slope.Add(slope, a)
		slope.Div(slope, tmp.Add(y, y))
		slope.Sub(zero, slope)

		intercept := k.NewZero().Set(y)
		intercept.Add(intercept, tmp.Mul(slope, x))
		intercept.Sub(zero, intercept)

		line := nag.NewPolynomial(k, nag.Deglex,
			nag.PolynomialTerm[K]{Coefficient: one, Monomial: []nag.Symbol{symbolY}},
			nag.PolynomialTerm[K]{Coefficient: slope, Monomial: make([]nag.Symbol, 1)},
			nag.PolynomialTerm[K]{Coefficient: intercept})
		return line
	}

	points := [][]K{p.Coords(), q.Coords()}
	fit := FitPoints(points)

	neg1 := nag.NewPolynomial(k, fit.Order(), nag.PolynomialTerm[K]{Coefficient: tmp.Sub(zero, one)})
	fit.Mul(neg1, fit)
	line := nag.NewPolynomial(k, fit.Order(), nag.PolynomialTerm[K]{Coefficient: one, Monomial: []nag.Symbol{symbolY}})
	line.Add(line, fit)
	return line
}

func verticalLine[K nag.Field[K], E CurvePoint[E, K]](p E) *nag.Polynomial[K] {
	k := p.Coords()[0].NewZero()
	zero, one, tmp := k.NewZero(), k.NewOne(), k.NewZero()
	vertical := nag.NewPolynomial(k, nag.Deglex,
		nag.PolynomialTerm[K]{Coefficient: one, Monomial: make([]nag.Symbol, 1)},
		nag.PolynomialTerm[K]{Coefficient: tmp.Sub(zero, p.Coords()[0])})
	return vertical
}

type divisorFunc[K nag.Field[K]] struct {
	a, b       K
	Num, Denom *nag.Polynomial[K]
}

func newDivisorFunc[K nag.Field[K]](a, b K, num, denom *nag.Polynomial[K]) *divisorFunc[K] {
	return &divisorFunc[K]{a: a, b: b, Num: num, Denom: denom}
}

func NewDivisorFuncOne[K nag.Field[K]](a, b K) *divisorFunc[K] {
	k := a.NewZero()
	x := &divisorFunc[K]{a: k.NewZero().Set(a), b: k.NewZero().Set(b)}
	x.Num = nag.NewPolynomial(k, nag.Deglex, nag.PolynomialTerm[K]{Coefficient: k.NewOne()})
	x.Denom = nag.NewPolynomial(k, nag.Deglex, nag.PolynomialTerm[K]{Coefficient: k.NewOne()})
	return x
}

func (z *divisorFunc[K]) NewOne() *divisorFunc[K] {
	k := z.Num.Field()
	x := &divisorFunc[K]{a: k.NewZero().Set(z.a), b: k.NewZero().Set(z.b)}
	x.Num = nag.NewPolynomial(k, z.Num.Order(), nag.PolynomialTerm[K]{Coefficient: k.NewOne()})
	x.Denom = nag.NewPolynomial(k, z.Num.Order(), nag.PolynomialTerm[K]{Coefficient: k.NewOne()})
	return x
}

func (z *divisorFunc[K]) Equal(x *divisorFunc[K]) bool {
	return z.a.Equal(x.a) && z.b.Equal(x.b) && z.Num.Equal(x.Num) && z.Denom.Equal(x.Denom)
}

func (z *divisorFunc[K]) Set(x *divisorFunc[K]) *divisorFunc[K] {
	z.a.Set(x.a)
	z.b.Set(x.b)
	z.Num.Set(x.Num)
	z.Denom.Set(x.Denom)
	return z
}

func (z *divisorFunc[K]) Mul(x, y *divisorFunc[K]) *divisorFunc[K] {
	k := x.Num.Field()
	zero, one := k.NewZero(), k.NewOne()
	neg1 := k.NewZero().Sub(zero, one)
	elliptic := ellipticPoly(z.a, z.b)
	commute := nag.NewPolynomial(elliptic.Field(), elliptic.Order(),
		// x*y
		nag.PolynomialTerm[K]{Coefficient: one, Monomial: []nag.Symbol{symbolX, symbolY}},
		// -y*x
		nag.PolynomialTerm[K]{Coefficient: neg1, Monomial: []nag.Symbol{symbolY, symbolX}})
	ellipticBasis := []*nag.Polynomial[K]{elliptic, commute}

	// Group operation on divisors is defined as the multiplication of their defining functions.
	z.Num.Mul(x.Num, y.Num)
	z.Denom.Mul(x.Denom, y.Denom)

	// Simplify terms using the elliptic curve equation.
	_, z.Num = nag.Divide(nil, z.Num, ellipticBasis)
	_, z.Denom = nag.Divide(nil, z.Denom, ellipticBasis)
	// Reduce fraction.
	nd := reduceFraction([2]*nag.Polynomial[K]{z.Num, z.Denom})
	z.Num, z.Denom = nd[0], nd[1]

	return z
}

func (z *divisorFunc[K]) Inv(x *divisorFunc[K]) *divisorFunc[K] {
	xnum := nag.NewPolynomial(x.Num.Field(), x.Num.Order()).Set(x.Num)
	z.Num.Set(x.Denom)
	z.Denom.Set(xnum)
	return z
}

func (z *divisorFunc[K]) String() string {
	return fmt.Sprintf("(%v/%v)", z.Num, z.Denom)
}

func EvalDivisorFunc[K nag.Field[K], E CurvePoint[E, K]](f *divisorFunc[K], p E) K {
	num := EvalPoly(f.Num, p.Coords())
	denom := EvalPoly(f.Denom, p.Coords())
	return num.Div(num, denom)
}

func divisorFuncRP[K nag.Field[K], E CurvePoint[E, K]](r *big.Int, p E, a, b K) *divisorFunc[K] {
	k := p.Coords()[0].NewZero()
	one := k.NewOne()

	p2 := nag.Pow(p.NewOne().Set(p), big.NewInt(2))
	frp := newDivisorFunc(a, b, linePQ(p, p2.Inv(p2), a),
		verticalLine(nag.Pow(p.NewOne().Set(p), big.NewInt(2))))
	lv := frp.NewOne()
	r1 := new(big.Int).Sub(r, big.NewInt(1))
	for m := big.NewInt(2); m.Cmp(r) < 0; m.Add(m, big.NewInt(1)) {
		if m.Cmp(r1) == 0 {
			// Since [r]P = O, this means [r-1]P = -P.
			// Therefore, linePQ(P, [r-1]P) is the vertical line.
			lv.Num.Set(verticalLine(p))
			// verticalLine([r]P) = verticalLine(O) is defined to be 1.
			lv.Denom.Set(nag.NewPolynomial(lv.Denom.Field(), lv.Denom.Order(), nag.PolynomialTerm[K]{Coefficient: one}))
		} else {
			mp := nag.Pow(p2.Set(p), new(big.Int).Set(m))
			lv.Num.Set(linePQ(p, mp, a))
			m1 := new(big.Int).Add(m, big.NewInt(1))
			lv.Denom.Set(verticalLine(nag.Pow(p2.Set(p), m1)))
		}

		frp.Mul(frp, lv)
	}
	return frp
}

func weilPairingNaive[K field.Finite[K], E CurvePoint[E, K]](order *big.Int, p, q E, a, b K, r E) K {
	f := divisorFuncRP(order, p, a, b)
	g := divisorFuncRP(order, q, a, b)

	pr := p.NewOne().Mul(p, r)
	lv := newDivisorFunc(a, b, linePQ(p, r, a), verticalLine(pr))

	lvq := EvalDivisorFunc(lv, q)
	fq := EvalDivisorFunc(f, q)
	fq.Div(fq, nag.Pow(lvq, new(big.Int).Set(order)))

	fq.Mul(fq, EvalDivisorFunc(g, r))
	fq.Div(fq, EvalDivisorFunc(g, pr))
	return fq
}

func tatePairingNaive[K field.Finite[K], E CurvePoint[E, K]](order *big.Int, p, q E, a, b K, r E) K {
	f := divisorFuncRP(order, p, a, b)

	qr := p.NewOne().Mul(q, r)
	fqr := EvalDivisorFunc(f, qr)
	fr := EvalDivisorFunc(f, r)
	fqr.Div(fqr, fr)

	qk1r := new(big.Int).Exp(fqr.Characteristic(), big.NewInt(int64(fqr.Degree())), nil)
	qk1r.Sub(qk1r, big.NewInt(1))
	qk1r.Div(qk1r, order)
	fqr = nag.Pow(fqr, qk1r)

	return fqr
}

func FitDivisorFunc[K nag.Field[K], E CurvePoint[E, K]](lv *divisorFunc[K], r, p E, a K, one *nag.Polynomial[K], tmp E) (*divisorFunc[K], E) {
	if tmp.Mul(r, p).IsInfinity() {
		// If R + P = O, then the line between R and P is vertical.
		lv.Num.Set(verticalLine(p))
		r.Mul(r, p)
		lv.Denom.Set(one)
	} else {
		lv.Num.Set(linePQ(r, p, a))
		r.Mul(r, p)
		lv.Denom.Set(verticalLine(r))
	}
	return lv, r
}

func Miller[K nag.Field[K], E CurvePoint[E, K]](order *big.Int, p, q E, a, b K) (K, E) {
	bits := make([]int64, 0)
	bit := new(big.Int)
	od := new(big.Int).Set(order)
	for od.Sign() != 0 {
		od.QuoRem(od, big.NewInt(2), bit)
		bits = append(bits, bit.Int64())
	}

	tmp := p.NewOne()
	lv := NewDivisorFuncOne(a, b)
	one := nag.NewPolynomial(lv.Num.Field(), lv.Num.Order()).Set(lv.Num)
	r := p.NewOne().Set(p)
	f := a.NewOne()
	for i := len(bits) - 2; i >= 0; i-- {
		lv, r = FitDivisorFunc(lv, r, r, a, one, tmp)

		lvEval := EvalDivisorFunc(lv, q)
		f.Mul(f, f)
		f.Mul(f, lvEval)

		if bits[i] == 1 {
			lv, r = FitDivisorFunc(lv, r, p, a, one, tmp)

			lvEval := EvalDivisorFunc(lv, q)
			f.Mul(f, lvEval)
		}
	}
	return f, r
}

func tatePairing[K field.Finite[K], E CurvePoint[E, K]](r *big.Int, p, q E, a, b K) K {
	f, _ := Miller(r, p, q, a, b)

	qk1r := new(big.Int).Exp(f.Characteristic(), big.NewInt(int64(f.Degree())), nil)
	qk1r.Sub(qk1r, big.NewInt(1))
	qk1r.Div(qk1r, r)
	f = nag.Pow(f, qk1r)
	return f
}

func weilPairing[K field.Finite[K], E CurvePoint[E, K]](r *big.Int, p, q E, a, b K) K {
	f, _ := Miller(r, p, q, a, b)
	g, _ := Miller(r, q, p, a, b)
	f.Div(f, g)

	neg1rI := new(big.Int).Exp(big.NewInt(-1), r, nil)
	if neg1rI.Int64() == -1 {
		f.Sub(f.NewZero(), f)
	}

	return f
}

func AtePairing[K field.Finite[K], E CurvePoint[E, K]](order, trace *big.Int, q, p E, a, b K) K {
	tr := new(big.Int).Sub(trace, big.NewInt(1))
	negative := tr.Sign() < 0
	tr.Abs(tr)

	f, r := Miller(tr, q, p, a, b)

	if negative {
		vee := EvalPoly(verticalLine(r), p.Coords())
		f.Mul(f, vee)
		f.Inv(f)
	}

	qk1r := new(big.Int).Exp(f.Characteristic(), big.NewInt(int64(f.Degree())), nil)
	qk1r.Sub(qk1r, big.NewInt(1))
	qk1r.Div(qk1r, order)
	f = nag.Pow(f, qk1r)
	return f
}
