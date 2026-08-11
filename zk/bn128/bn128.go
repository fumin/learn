package bn128

import (
	"math/big"

	"github.com/fumin/learn/zk/bn128/gfp"
	"github.com/fumin/learn/zk/ecc"
	"github.com/fumin/nag"
	"github.com/fumin/nag/field"
)

var (
	fieldChar = new(big.Int).Set(gfp.P)
	fp        = gfp.NewGfp(1)
	G1        = ecc.NewWeierstrassA0[*gfp.Gfp](
		fp.NewZero().SetCoeffs(big.NewInt(1)),
		fp.NewZero().SetCoeffs(big.NewInt(2)))
	b          = fp.NewZero().SetCoeffs(big.NewInt(3))
	curveOrder = ecc.Int10("21888242871839275222246405745257275088548364400416034343698204186575808495617")

	// fp12 irreducible polynomial is w^12-18w^6+82 = (w^6-9)^2 + 1.
	fp12 = (*FiniteExt[*gfp.Gfp])(&field.Extension[*gfp.Gfp]{
		Irr: nag.NewPolynomial(gfp.NewGfp(0), nag.Deglex,
			nag.PolynomialTerm[*gfp.Gfp]{Coefficient: gfp.NewGfp(82)},
			nag.PolynomialTerm[*gfp.Gfp]{Coefficient: gfp.NewGfp(-18), Monomial: make([]nag.Symbol, 6)},
			nag.PolynomialTerm[*gfp.Gfp]{Coefficient: gfp.NewGfp(1), Monomial: make([]nag.Symbol, 12)}),
		Poly: nag.NewPolynomial(gfp.NewGfp(0), nag.Deglex),
	})
	w = fp12.NewZero().SetCoeffs(big.NewInt(0), big.NewInt(1))

	// fp2 irreducible polynomial is u^2+1, thus u = w^6 - 9.
	fp2 = (*FiniteExt[*gfp.Gfp])(&field.Extension[*gfp.Gfp]{
		Irr: nag.NewPolynomial(gfp.NewGfp(0), nag.Deglex,
			nag.PolynomialTerm[*gfp.Gfp]{Coefficient: gfp.NewGfp(1)},
			nag.PolynomialTerm[*gfp.Gfp]{Coefficient: gfp.NewGfp(1), Monomial: make([]nag.Symbol, 2)}),
		Poly: nag.NewPolynomial(gfp.NewGfp(0), nag.Deglex),
	})
	u = fp12.NewZero().Sub(
		nag.Pow(fp12.NewZero().Set(w), big.NewInt(6)),
		fp12.NewZero().SetCoeffs(big.NewInt(9)))
	// twist curve is y^2 = x^3 + (a/w^4)x + b/w^6, thus b2 = b / (u+9).
	b2 = fp2.NewZero().Div(
		fp2.NewZero().SetCoeffs(b.Coeffs()...),
		fp2.NewZero().Add(
			fp2.NewZero().SetCoeffs(big.NewInt(0), big.NewInt(1)),
			fp2.NewZero().SetCoeffs(big.NewInt(9))))
	G2 = ecc.NewWeierstrassA0[*FiniteExt[*gfp.Gfp]](
		fp2.NewZero().SetCoeffs(
			ecc.Int10("10857046999023057135944570762232829481370756359578518086990519993285655852781"),
			ecc.Int10("11559732032986387107991004021392285783925812861821192530917403151452391805634"),
		),
		fp2.NewZero().SetCoeffs(
			ecc.Int10("8495653923123431417604973247489272438418190587263600148770280649306958101930"),
			ecc.Int10("4082367875863433681332203403145435568316851327593401208105741076214120093531"),
		))
	g12 = twist(u, w, ecc.NewWeierstrassA0(fp12.NewZero(), fp12.NewZero()), G2)
)

// Pairing computes the optimal Ate pairing e(q, p) of q in G2 and p in G1.
func Pairing[K1 field.Finite[K1], K2 field.Finite[K2], E1 ecc.CurvePoint[E1, K1], E2 ecc.CurvePoint[E2, K2]](q E2, p E1) *FiniteExt[*gfp.Gfp] {
	q12 := twist(u, w, g12.NewOne(), q)
	p12 := fp12PointFromfp1(g12.NewOne(), p)
	return optimalAte(q12, p12)
}

// optimalAte computes the optimal Ate Miller loop of q in G2 and p in G1,
// both already embedded into Fp12. See
// https://github.com/ethereum/py_ecc/blob/v8.0.0/py_ecc/bn128/bn128_pairing.py.
func optimalAte[K field.Finite[K], E ecc.CurvePoint[E, K]](q, p E) K {
	// ateLoopCount=6x+2, where x=4965661367192848881 is the BN parameter.
	// This shorter loop (compared to the trace-based Ate pairing loop)
	// together with the two Frobenius correction steps at the end of
	// millerLoop define the "optimal Ate pairing" for BN curves.
	// See Vercauteren, "Optimal Pairings".
	ateLoopCount := ecc.Int10("29793968203157093288")

	k := q.Coords()[0]
	a := k.NewZero()
	bT := k.NewZero().SetCoeffs(b.Coeffs()...)
	f, r := ecc.Miller(ateLoopCount, q, p, a, bT)

	// Two extra line evals needed, compared to the general ate-pairing.
	// This is because ateLoopCount != t-1, where t is the curve's
	// trace of Frobenius.
	tmp := q.NewOne()
	lv := ecc.NewDivisorFuncOne(a, bT)
	one := nag.NewPolynomial(lv.Num.Field(), lv.Num.Order()).Set(lv.Num)
	mulLine := func(f K, r, linePoint, evalPoint E) (K, E) {
		lv, r = ecc.FitDivisorFunc(lv, r, linePoint, a, one, tmp)
		lvEval := ecc.EvalDivisorFunc(lv, evalPoint)
		f.Mul(f, lvEval)
		return f, r
	}
	// Line eval for FrobeniusEndo(q, 1).
	q1 := q.NewOne().SetCoords(ecc.FrobeniusEndo(q.Coords(), 1))
	f, r = mulLine(f, r, q1, p)
	// Line eval for -FrobeniusEndo(q, 2)
	q2Coords := ecc.FrobeniusEndo(q.Coords(), 2)
	negQ2 := q.NewOne().SetCoords([]K{q2Coords[0], k.NewZero().Sub(k.NewZero(), q2Coords[1])})
	f, r = mulLine(f, r, negQ2, p)

	// Final exponentiation.
	xpn := new(big.Int).Exp(k.Characteristic(), big.NewInt(int64(k.Degree())), nil)
	xpn.Sub(xpn, big.NewInt(1))
	xpn.Div(xpn, curveOrder)
	return nag.Pow(f, xpn)
}

func twist2To12[K2 field.Finite[K2], K12 field.Finite[K12]](u K12, x K2) K12 {
	xc := x.Coeffs()
	a := u.NewZero().SetCoeffs(xc[0])
	b := u.NewZero().SetCoeffs(xc[1])
	return a.Add(a, b.Mul(b, u))
}

func twist[K2 field.Finite[K2], K12 field.Finite[K12], G2 ecc.CurvePoint[G2, K2], GT ecc.CurvePoint[GT, K12]](u, w K12, pt GT, p2 G2) GT {
	coords := p2.Coords()
	tcoords := make([]K12, len(coords))
	for i := range tcoords {
		tcoords[i] = twist2To12(u, coords[i])
	}

	// Perform the twist isomorphism (x, y) -> (x*w^2, y&w^3)
	tcoords[0].Mul(tcoords[0], nag.Pow(u.NewZero().Set(w), big.NewInt(2)))
	tcoords[1].Mul(tcoords[1], nag.Pow(u.NewZero().Set(w), big.NewInt(3)))

	return pt.SetCoords(tcoords)
}

func fp12PointFromfp1[K1 field.Finite[K1], K12 field.Finite[K12], G1 ecc.CurvePoint[G1, K1], GT ecc.CurvePoint[GT, K12]](pt GT, p1 G1) GT {
	k12 := pt.Coords()[0]
	coords := p1.Coords()
	tcoords := make([]K12, len(coords))
	for i := range tcoords {
		tcoords[i] = k12.NewZero().SetCoeffs(coords[i].Coeffs()...)
	}
	return pt.SetCoords(tcoords)
}

func bigs(is ...int64) []*big.Int {
	bs := make([]*big.Int, len(is))
	for i := range bs {
		bs[i] = big.NewInt(is[i])
	}
	return bs
}
