package ecc

import (
	"maps"
	"slices"

	"github.com/fumin/nag"
)

func reduceFraction[K nag.Field[K]](f [2]*nag.Polynomial[K]) [2]*nag.Polynomial[K] {
	gcd := Gcd(f[0], f[1])
	f[0], _ = Divide(f[0], gcd)
	f[1], _ = Divide(f[1], gcd)
	return f
}

func Lcm[K nag.Field[K]](f, g *nag.Polynomial[K]) *nag.Polynomial[K] {
	// Homogenize f and so that they can be analyzed by the homogeneous
	// Buchberger algorithm which guarantees completion within degree f*g.
	symbolH, symbolT, symbols := getSymbols(f, g)
	hf := homogenize(symbolH, f)
	hg := homogenize(symbolH, g)

	// Find unused symbol for the special variable t used in
	// Proposition 14 of Chapter 4.3,
	// Ideals, Varieties, and Algorithms, D. Cox, J. Little, D. O'Shea.
	k := f.Field()
	zero, one := k.NewZero(), k.NewOne()
	neg1 := k.NewZero().Sub(zero, one)
	t := nag.NewPolynomial(k, nag.ElimOrder(),
		nag.PolynomialTerm[K]{Coefficient: one, Monomial: []nag.Symbol{symbolT}})
	t1 := nag.NewPolynomial(k, nag.ElimOrder(),
		nag.PolynomialTerm[K]{Coefficient: one, Monomial: []nag.Symbol{symbolH}},
		nag.PolynomialTerm[K]{Coefficient: neg1, Monomial: []nag.Symbol{symbolT}})
	// Compute t*f and (1-t)*g.
	tf := nag.NewPolynomial(t.Field(), t.Order())
	copyPoly(tf, hf)
	tf.Mul(tf, t)
	tg := nag.NewPolynomial(t.Field(), t.Order())
	copyPoly(tg, hg)
	tg.Mul(tg, t1)

	fg := nag.NewPolynomial(t.Field(), t.Order()).Mul(f, g)
	fgDeg := 0
	for _, m := range fg.Terms() {
		fgDeg = max(fgDeg, len(m))
	}

	ideal := []*nag.Polynomial[K]{tf, tg}
	ideal = appendCommute(ideal, symbols)
	basis, _ := nag.BuchbergerHomogeneous(ideal, fgDeg+1)
	// noT returns whether p contains the variable "t".
	noT := func(p *nag.Polynomial[K]) bool {
		deg := 0
		for _, m := range p.Terms() {
			deg = max(deg, len(m))
			if slices.Contains(m, symbolT) {
				return false
			}
		}
		return !isCommuteRelation(p)
	}
	idx := slices.IndexFunc(basis, noT)
	hlcm := basis[idx]
	lcmElimOrder := dehomogenize(symbolH, hlcm)

	lcm := nag.NewPolynomial(f.Field(), f.Order())
	copyPoly(lcm, lcmElimOrder)
	monicize(lcm)
	return lcm
}

func Gcd[K nag.Field[K]](f, g *nag.Polynomial[K]) *nag.Polynomial[K] {
	lcm := Lcm(f, g)
	fg := nag.NewPolynomial(lcm.Field(), lcm.Order()).Mul(f, g)
	gcd, _ := Divide(fg, lcm)
	return gcd
}

func copyPoly[K nag.Field[K]](dst, src *nag.Polynomial[K]) {
	for srcC, srcM := range src.Terms() {
		m := make([]nag.Symbol, len(srcM))
		copy(m, srcM)
		term := nag.NewPolynomial(dst.Field(), dst.Order(), nag.PolynomialTerm[K]{Coefficient: srcC, Monomial: m})
		dst.Add(dst, term)
	}
}

func Divide[K nag.Field[K]](f, g *nag.Polynomial[K]) (*nag.Polynomial[K], *nag.Polynomial[K]) {
	symbolM := make(map[nag.Symbol]struct{})
	for _, m := range f.Terms() {
		for _, s := range m {
			symbolM[s] = struct{}{}
		}
	}
	for _, m := range g.Terms() {
		for _, s := range m {
			symbolM[s] = struct{}{}
		}
	}
	symbols := slices.Collect(maps.Keys(symbolM))

	k := f.Field()
	f2 := nag.NewPolynomial(k, f.Order()).Set(f)
	ideal := []*nag.Polynomial[K]{g}
	ideal = appendCommute(ideal, symbols)
	quotient := make([][]nag.Quotient[K], 0)
	quotient, r := nag.Divide(quotient, f2, ideal)

	q := nag.NewPolynomial(f.Field(), f.Order())
	q.SymbolStringer = f.SymbolStringer
	for j := range quotient[0] {
		c := nag.NewPolynomial(k, q.Order(), nag.PolynomialTerm[K]{Coefficient: quotient[0][j].Coefficient})
		left := nag.NewPolynomial(k, q.Order(), nag.PolynomialTerm[K]{Coefficient: k.NewOne(), Monomial: quotient[0][j].Left})
		right := nag.NewPolynomial(k, q.Order(), nag.PolynomialTerm[K]{Coefficient: k.NewOne(), Monomial: quotient[0][j].Right})

		cwgw := nag.NewPolynomial(k, f.Order())
		cwgw.SymbolStringer = f.SymbolStringer
		cwgw.Mul(c, left)
		cwgw.Mul(cwgw, right)

		q.Add(q, cwgw)
	}
	return q, r
}

func appendCommute[K nag.Field[K]](ideal []*nag.Polynomial[K], symbols []nag.Symbol) []*nag.Polynomial[K] {
	k := ideal[0].Field()
	one, neg1 := k.NewOne(), k.Sub(k.NewZero(), k.NewOne())
	for i := range symbols {
		for j := i + 1; j < len(symbols); j++ {
			commute := nag.NewPolynomial(k, ideal[0].Order(),
				nag.PolynomialTerm[K]{
					Coefficient: one,
					Monomial:    []nag.Symbol{symbols[i], symbols[j]}},
				nag.PolynomialTerm[K]{
					Coefficient: neg1,
					Monomial:    []nag.Symbol{symbols[j], symbols[i]}})
			commute.SymbolStringer = ideal[0].SymbolStringer
			ideal = append(ideal, commute)
		}
	}
	return ideal
}

func isCommuteRelation[K nag.Field[K]](p *nag.Polynomial[K]) bool {
	if p.Len() != 2 {
		return false
	}
	var c0, c1 K
	var m0, m1 []nag.Symbol
	i := -1
	for c, m := range p.Terms() {
		i++
		switch i {
		case 0:
			c0, m0 = c, m
		case 1:
			c1, m1 = c, m
		}
	}

	// Check that c0 == -c1.
	zero := c1.NewZero()
	negC1 := zero.Sub(zero, c1)
	if !c0.Equal(negC1) {
		return false
	}

	// Check that m0 = reverse(m1).
	if !(len(m0) == 2 && len(m1) == 2) {
		return false
	}
	if !(m0[0] == m1[1] && m0[1] == m1[0]) {
		return false
	}

	return true
}

func getSymbols[K nag.Field[K]](f, g *nag.Polynomial[K]) (nag.Symbol, nag.Symbol, []nag.Symbol) {
	symbolM := make(map[nag.Symbol]struct{})
	for _, m := range f.Terms() {
		for _, s := range m {
			symbolM[s] = struct{}{}
		}
	}
	for _, m := range g.Terms() {
		for _, s := range m {
			symbolM[s] = struct{}{}
		}
	}

	// symbol "h"
	symbolHI := -1
	for s := range 256 {
		if _, ok := symbolM[nag.Symbol(s)]; !ok {
			symbolHI = s
			break
		}
	}
	if symbolHI == -1 {
		panic("no unused symbol")
	}
	symbolH := nag.Symbol(symbolHI)
	symbolM[symbolH] = struct{}{}

	// symbol "t"
	symbolTI := -1
	for s := range 256 {
		if _, ok := symbolM[nag.Symbol(s)]; !ok {
			symbolTI = s
			break
		}
	}
	if symbolTI == -1 {
		panic("no unused symbol")
	}
	symbolT := nag.Symbol(symbolTI)
	symbolM[symbolT] = struct{}{}

	symbols := slices.Collect(maps.Keys(symbolM))
	return symbolH, symbolT, symbols
}

func homogenize[K nag.Field[K]](h nag.Symbol, p *nag.Polynomial[K]) *nag.Polynomial[K] {
	deg := 0
	for _, m := range p.Terms() {
		deg = max(deg, len(m))
	}

	hp := nag.NewPolynomial(p.Field(), p.Order())
	for c, m := range p.Terms() {
		hm := make([]nag.Symbol, deg)
		copy(hm, m)
		for i := len(m); i < deg; i++ {
			hm[i] = h
		}

		term := nag.NewPolynomial(p.Field(), p.Order(), nag.PolynomialTerm[K]{Coefficient: c, Monomial: hm})
		hp.Add(hp, term)
	}
	return hp
}

func dehomogenize[K nag.Field[K]](h nag.Symbol, hp *nag.Polynomial[K]) *nag.Polynomial[K] {
	p := nag.NewPolynomial(hp.Field(), hp.Order())
	p.SymbolStringer = hp.SymbolStringer
	for c, hm := range hp.Terms() {
		m := make([]nag.Symbol, 0)
		for _, s := range hm {
			if s != h {
				m = append(m, s)
			}
		}

		term := nag.NewPolynomial(p.Field(), p.Order(), nag.PolynomialTerm[K]{Coefficient: c, Monomial: m})
		p.Add(p, term)
	}
	return p
}

func monicize[K nag.Field[K]](p *nag.Polynomial[K]) {
	lc := p.LeadingTerm().Coefficient
	invlc := lc.NewZero().Inv(lc)
	lcp := nag.NewPolynomial(p.Field(), p.Order(), nag.PolynomialTerm[K]{Coefficient: invlc})
	p.Mul(p, lcp)
}
