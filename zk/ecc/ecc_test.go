package ecc

import (
	"bytes"
	"cmp"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"maps"
	"math/big"
	"os"
	"slices"
	"testing"

	"github.com/fumin/nag"
	"github.com/fumin/nag/field"
)

func TestDivisorReduction(t *testing.T) {
	p := big.NewInt(41)
	Fq := field.NewPrimeExtDeg(p, 1)
	vs := map[string]nag.Symbol{"x": 0}
	elliptic := "x^3 + 17x + 16"
	d := [][]*field.PrimeExt{
		{Fq.NewZero().SetCoeffs(big.NewInt(19)), Fq.NewZero().SetCoeffs(big.NewInt(8))},
		{Fq.NewZero().SetCoeffs(big.NewInt(4)), Fq.NewZero().SetCoeffs(big.NewInt(36))},
		{Fq.NewZero().SetCoeffs(big.NewInt(18)), Fq.NewZero().SetCoeffs(big.NewInt(39))},
		{Fq.NewZero().SetCoeffs(big.NewInt(22)), Fq.NewZero().SetCoeffs(big.NewInt(38))},
		{Fq.NewZero().SetCoeffs(big.NewInt(32)), Fq.NewZero().SetCoeffs(big.NewInt(6))},
	}
	const shouldDump = false
	if shouldDump {
		fpath := "ellipticFq.csv"
		dumpPoints(fpath, solveElliptic(parse(vs, Fq, elliptic), -1))
	}

	conway := []*big.Int{big.NewInt(35), big.NewInt(1), big.NewInt(0), big.NewInt(1)}
	Fq3 := field.NewPrimeExt(p, conway)
	if shouldDump {
		fpath := "ellipticFq3.csv"
		dumpPoints(fpath, solveElliptic(parse(vs, Fq3, elliptic), -1))
	}

	// divisor is the Mumford representation the divisor containing the points in d.
	// https://en.wikipedia.org/wiki/Imaginary_hyperelliptic_curve
	divisor := mumfordRep(d)
	wantS := "21x^4+18x^3+10x^2+31x+30"
	if divisor[1].String() != wantS {
		t.Errorf("%v", divisor[1])
	}

	// Reduce the divisor once.
	divisor = cantorReduceElliptic(parse(vs, divisor[0].Field(), elliptic), divisor)
	// The reduced divisor contains len(d)-2 points.
	if len(divisor[0].LeadingTerm().Monomial) != len(d)-2 {
		t.Errorf("%v", divisor[0])
	}
	wantS = "34x^2+19x+14"
	if divisor[1].String() != wantS {
		t.Errorf("%v", divisor[1])
	}
	// Check that the support of the divisor are not defined in Fq.
	// They actually lie in Fq^3.
	supportX := findRoots(parse(vs, Fq, divisor[0].String()))
	if len(supportX) != 0 {
		t.Errorf("%v", supportX)
	}
	if shouldDump {
		supportX3 := findRoots(parse(vs, Fq3, divisor[0].String()))
		d2 := make([][]*field.PrimeExt, 0)
		for _, x := range supportX3 {
			y := EvalPoly(parse(vs, x.NewZero(), divisor[1].String()), []*field.PrimeExt{x})
			d2 = append(d2, []*field.PrimeExt{x, y})
		}
		fpath := "d2.csv"
		dumpPoints(fpath, d2)
	}

	// Do another round of reduction to get to a single point.
	divisor = cantorReduceElliptic(parse(vs, divisor[0].Field(), elliptic), divisor)
	supportX = findRoots(divisor[0])
	if len(supportX) != 1 {
		t.Errorf("%v", supportX)
	}
	supportY := EvalPoly(divisor[1], []*field.PrimeExt{supportX[0]})
	reduced := []*field.PrimeExt{supportX[0], supportY}
	want := []*field.PrimeExt{
		Fq.NewZero().SetCoeffs(big.NewInt(21)),
		Fq.NewZero().SetCoeffs(big.NewInt(32))}
	if !slices.EqualFunc(reduced, want, func(a, b *field.PrimeExt) bool { return a.Equal(b) }) {
		t.Errorf("%v", reduced)
	}
}

func TestNumPoints(t *testing.T) {
	tests := []struct {
		p            int
		numPoints    int
		n            int
		numPointsExt int64
	}{
		{p: 41, numPoints: 35, n: 3, numPointsExt: 69440},
		{p: 47, numPoints: 51, n: 4, numPointsExt: 3 * 3 * 3 * 5 * 5 * 5 * 5 * 17 * 17},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			got := NumPoints(test.p, test.numPoints, test.n)
			if got.Int64() != test.numPointsExt {
				t.Errorf("NumPoints(%d, %d, %d) = %d want %d", test.p, test.numPoints, test.n, got, test.numPointsExt)
			}
		})
	}
}

func TestEllipticCurve(t *testing.T) {
	tests := []struct {
		fp   int64
		n    int
		a    int64
		p, q [][]int64
		pq   [][]int64
	}{
		// Example 2.1.2, Pairings for beginners, Craig Costello.
		{
			fp: 11, n: 1,
			a: -2,
			p: [][]int64{{5}, {7}}, q: [][]int64{{8}, {10}},
			pq: [][]int64{{10}, {10}},
		},
		// Example 2.1.2, add, Pairings for beginners, Craig Costello.
		{
			fp: 23, n: 1,
			a: 5,
			p: [][]int64{{2}, {5}}, q: [][]int64{{12}, {1}},
			pq: [][]int64{{11}, {17}},
		},
		// Example 2.1.2, double, Pairings for beginners, Craig Costello.
		{
			fp: 23, n: 1,
			a: 5,
			p: [][]int64{{2}, {5}}, q: [][]int64{{2}, {5}},
			pq: [][]int64{{12}, {1}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExtDeg(big.NewInt(test.fp), test.n)
			e := newEllipticCurve(k.NewZero().SetCoeffs(big.NewInt(test.a)))
			p := e.NewOne().SetCoords(newPoint(k, test.p))
			q := e.NewOne().SetCoords(newPoint(k, test.q))
			want := e.NewOne().SetCoords(newPoint(k, test.pq))
			pq := e.NewOne().Mul(p, q)
			qp := e.NewOne().Mul(q, p)
			if !pq.Equal(pq) {
				t.Errorf("%v != %v", pq, qp)
			}
			if !pq.Equal(want) {
				t.Errorf("Mul(%v, %v) = %v want %v", p, q, pq, want)
			}
		})
	}
}

func TestEllipticCurveMul(t *testing.T) {
	tests := []struct {
		fp  int64
		irr []int64
		a   int64
		p   [][]int64
		m   int64
		mp  [][]int64
	}{
		// Example 2.1.8, Pairings for beginners, Craig Costello.
		{
			fp: 1021, irr: []int64{1, 1},
			a:  -3,
			p:  [][]int64{{379}, {1011}},
			m:  655,
			mp: [][]int64{{388}, {60}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExt(big.NewInt(test.fp), bigs(test.irr...))
			e := newEllipticCurve(k.NewZero().SetCoeffs(big.NewInt(test.a)))
			p := e.NewOne().SetCoords(newPoint(k, test.p))
			want := e.NewOne().SetCoords(newPoint(k, test.mp))
			mp := multiply(test.m, p)
			if !mp.Equal(want) {
				t.Errorf("multiply(%d, %v) = %v want %v", test.m, p, mp, want)
			}
		})
	}
}

func TestFrobeniusEndo(t *testing.T) {
	tests := []struct {
		p   int64
		irr []int64
		in  [][]int64
		n   int
		out [][]int64
	}{
		// Example 2.2.5, Pairings for beginners, Craig Costello.
		{
			p: 67, irr: []int64{1, 0, 1},
			in:  [][]int64{{16, 2}, {39, 30}},
			n:   1,
			out: [][]int64{{16, 65}, {39, 37}},
		},
		{
			p: 67, irr: []int64{2, 0, 0, 1},
			in:  [][]int64{{8, 4, 15}, {21, 30, 44}},
			n:   1,
			out: [][]int64{{8, 14, 33}, {21, 38, 3}},
		},
		{
			p: 67, irr: []int64{2, 0, 0, 1},
			in:  [][]int64{{8, 4, 15}, {21, 30, 44}},
			n:   2,
			out: [][]int64{{8, 49, 19}, {21, 66, 20}},
		},
		{
			p: 67, irr: []int64{2, 0, 0, 1},
			in:  [][]int64{{8, 4, 15}, {21, 30, 44}},
			n:   3,
			out: [][]int64{{8, 4, 15}, {21, 30, 44}},
		},
		{
			p: 67, irr: []int64{2, 0, 0, 1},
			in:  [][]int64{{8, 4, 15}, {21, 30, 44}},
			n:   0,
			out: [][]int64{{8, 4, 15}, {21, 30, 44}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExt(big.NewInt(test.p), bigs(test.irr...))
			in, out := newPoint(k, test.in), newPoint(k, test.out)
			frob := FrobeniusEndo(in, test.n)
			if !slices.EqualFunc(frob, out, func(a, b *field.PrimeExt) bool { return a.Equal(b) }) {
				t.Errorf("FrobeniusEndo(%v, %d) = %v want %v", in, test.n, frob, out)
			}
		})
	}
}

func TestTorsion(t *testing.T) {
	tests := []struct {
		q               int64
		irr             []int64
		a               int64
		primitiveFormat bool
		torsion         [][][][]int64
	}{
		// Example 4.1.1, Pairings for beginners, Craig Costello.
		{
			q: 11, irr: []int64{1, 0, 1},
			a: 0,
			torsion: [][][][]int64{
				{{{0}, {2}}, {{0}, {9}}},
				{{{7, 2}, {0, 10}}, {{7, 2}, {0, 1}}},
				{{{7, 9}, {0, 1}}, {{7, 9}, {0, 10}}},
				{{{8}, {0, 1}}, {{8}, {0, 10}}},
			},
		},
		// Example 4.1.3, Pairings for beginners, Craig Costello.
		{
			q: 11, irr: []int64{4, 1, 0, 1},
			a:               7,
			primitiveFormat: true,
			torsion: [][][][]int64{
				{{{10, 0}, {7, 0}}, {{8, 0}, {3, 0}}, {{8, 0}, {8, 0}}, {{10, 0}, {4, 0}}, {{7, 0}, {3, 0}}, {{7, 0}, {8, 0}}},
				{{{942}, {749}}, {{1011}, {579}}, {{1324}, {1095}}, {{1011}, {1244}}, {{942}, {84}}, {{1324}, {430}}},
				{{{1161}, {464}}, {{419}, {172}}, {{643}, {1225}}, {{419}, {837}}, {{1161}, {1129}}, {{643}, {560}}},
				{{{159}, {862}}, {{663}, {595}}, {{663}, {1260}}, {{831}, {284}}, {{159}, {197}}, {{831}, {949}}},
				{{{423}, {840}}, {{619}, {1227}}, {{801}, {1114}}, {{619}, {562}}, {{423}, {175}}, {{801}, {449}}},
				{{{932}, {854}}, {{932}, {189}}, {{1301}, {234}}, {{1301}, {899}}, {{604}, {825}}, {{604}, {160}}},
				{{{481}, {1049}}, {{1052}, {924}}, {{1264}, {740}}, {{481}, {384}}, {{1052}, {259}}, {{1264}, {75}}},
				{{{1315}, {1150}}, {{1315}, {485}}, {{1165}, {680}}, {{845}, {165}}, {{1165}, {15}}, {{845}, {830}}},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			fqk := field.NewPrimeExt(big.NewInt(test.q), bigs(test.irr...))
			torsion := make([][][]*field.PrimeExt, len(test.torsion))
			for i := range test.torsion {
				torsion[i] = make([][]*field.PrimeExt, len(test.torsion[i]))
				for j := range test.torsion[i] {
					torsion[i][j] = newPointP(test.primitiveFormat, fqk, test.torsion[i][j])
				}
			}

			// For G1, the trace map equals the multiplication map [k].
			g1 := torsion[0]
			for _, pc := range g1 {
				p := newEllipticCurve(fqk.NewZero().SetCoeffs(big.NewInt(test.a))).SetCoords(pc)
				mp := multiply(int64(fqk.Degree()), p)

				tr := Trace(p)
				if !tr.Equal(mp) {
					t.Errorf("Trace(%v) = %v want %v", p, tr, mp)
				}
			}
			// For G2, trace maps to O.
			g2 := torsion[len(torsion)-1]
			for _, pc := range g2 {
				p := newEllipticCurve(fqk.NewZero().SetCoeffs(big.NewInt(test.a))).SetCoords(pc)
				tr := Trace(p)
				if !tr.Equal(p.NewOne()) {
					t.Errorf("Trace(%v) = %v not infinity", p, tr)
				}
			}
			// For the other subgroups, trace maps them to G1.
			for i, g := range torsion {
				if i == 0 || i == len(torsion)-1 {
					continue
				}
				for j, pc := range g {
					p := newEllipticCurve(fqk.NewZero().SetCoeffs(big.NewInt(test.a))).SetCoords(pc)
					tr := Trace(p)
					if !slices.ContainsFunc(g1, func(x []*field.PrimeExt) bool { return tr.NewOne().SetCoords(x).Equal(tr) }) {
						t.Errorf("%d %d Trace(%v) = %v not in g1", i, j, p, tr)
					}
				}
			}
		})
	}
}

func TestDivisorFuncRP(t *testing.T) {
	tests := []struct {
		q    int64
		a, b []int64
		r    int64
		p    [][]int64
		vs   []string
		want [2]string
	}{
		// Example 5.0.1, Pairings for beginners, Craig Costello.
		{
			q: 23,
			a: []int64{17}, b: []int64{6},
			r:    5,
			p:    [][]int64{{10}, {7}},
			vs:   []string{"x", "y"},
			want: [2]string{"(x+22)y+5x^2+3x+5", "1"},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExtDeg(big.NewInt(test.q), 1)
			a, b := k.NewZero().SetCoeffs(bigs(test.a...)...), k.NewZero().SetCoeffs(bigs(test.b...)...)
			r := big.NewInt(test.r)
			p := newEllipticCurve(a).SetCoords(newPoint(k, test.p))
			vs := make(map[string]nag.Symbol)
			for _, v := range test.vs {
				vs[v] = nag.Symbol(len(vs))
			}
			want := newDivisorFunc(a, b, parse(vs, k, test.want[0]), parse(vs, k, test.want[1]))

			frp := divisorFuncRP(r, p, a, b)
			if !frp.Equal(want) {
				t.Errorf("divisorFuncRP(%d, %v) = %v want %v", r, p, frp, want)
			}
		})
	}
}

func TestWeilPairingNaive(t *testing.T) {
	tests := []struct {
		fieldChar int64
		irr       []int64
		a, b      []int64
		order     int64
		p         [][]int64
		pmul      int64
		q         [][]int64
		qmul      int64
		r         [][]int64
		want      []int64
	}{
		// Example 5.1.1, Pairings for beginners, Craig Costello.
		{
			fieldChar: 23,
			irr:       []int64{1, 0, 1},
			a:         []int64{-1}, b: []int64{0},
			order: 3,
			p:     [][]int64{{2}, {11}},
			pmul:  1,
			q:     [][]int64{{21}, {0, 12}},
			qmul:  1,
			r:     [][]int64{{0, 17}, {21, 2}},
			want:  []int64{11, 15},
		},
		{
			fieldChar: 23,
			irr:       []int64{1, 0, 1},
			a:         []int64{-1}, b: []int64{0},
			order: 3,
			p:     [][]int64{{2}, {11}},
			pmul:  2,
			q:     [][]int64{{21}, {0, 12}},
			qmul:  1,
			r:     [][]int64{{0, 17}, {21, 2}},
			want:  []int64{11, 8},
		},
		{
			fieldChar: 23,
			irr:       []int64{1, 0, 1},
			a:         []int64{-1}, b: []int64{0},
			order: 3,
			p:     [][]int64{{2}, {11}},
			pmul:  1,
			q:     [][]int64{{21}, {0, 12}},
			qmul:  2,
			r:     [][]int64{{0, 17}, {21, 2}},
			want:  []int64{11, 8},
		},
		{
			fieldChar: 23,
			irr:       []int64{1, 0, 1},
			a:         []int64{-1}, b: []int64{0},
			order: 3,
			p:     [][]int64{{2}, {11}},
			pmul:  2,
			q:     [][]int64{{21}, {0, 12}},
			qmul:  2,
			r:     [][]int64{{0, 17}, {21, 2}},
			want:  []int64{11, 15},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			fieldChar := big.NewInt(test.fieldChar)
			fq2 := field.NewPrimeExt(fieldChar, bigs(test.irr...))
			a, b := fq2.NewZero().SetCoeffs(bigs(test.a...)...), fq2.NewZero().SetCoeffs(bigs(test.b...)...)
			order := big.NewInt(test.order)
			p := newEllipticCurve(a).SetCoords(newPoint(fq2, test.p))
			p = multiply(test.pmul, p)
			q := newEllipticCurve(a).SetCoords(newPoint(fq2, test.q))
			q = multiply(test.qmul, q)
			r := newEllipticCurve(a).SetCoords(newPoint(fq2, test.r))
			want := fq2.NewZero().SetCoeffs(bigs(test.want...)...)

			w := weilPairingNaive(order, p, q, a, b, r)
			if !w.Equal(want) {
				t.Errorf("weilPairing(%d, %v, %v, %v, %v) = %v != %v", order, p, q, a, b, w, want)
			}
		})
	}
}

// Example 5.2.1, Pairings for beginners, Craig Costello.
func TestExample_5_2_1(t *testing.T) {
	// Get the elliptic group.
	vs := map[string]nag.Symbol{"x": 0}
	fq2 := field.NewPrimeExt(big.NewInt(5), bigs(2, 0, 1))
	pcs := solveElliptic(parse(vs, fq2, "x^3-3"), -1)
	points := make([]*WeierstrassA0[*field.PrimeExt], len(pcs))
	for i := range points {
		points[i] = NewWeierstrassA0(fq2.NewZero().Set(pcs[i][0]), fq2.NewZero().Set(pcs[i][1]))
	}
	points = append(points, points[0].NewOne())
	if len(points) != 36 {
		t.Errorf("%d", len(points))
	}

	// Get the torsion.
	var r int64 = 3
	er := getTorsion(r, points)
	erWant := [][]*WeierstrassA0[*field.PrimeExt]{
		{NewWeierstrassA0(newPoint(fq2, [][]int64{{3}, {2}})...), NewWeierstrassA0(newPoint(fq2, [][]int64{{3}, {3}})...)},
		{NewWeierstrassA0(newPoint(fq2, [][]int64{{1, 3}, {2}})...), NewWeierstrassA0(newPoint(fq2, [][]int64{{1, 3}, {3}})...)},
		{NewWeierstrassA0(newPoint(fq2, [][]int64{{1, 2}, {2}})...), NewWeierstrassA0(newPoint(fq2, [][]int64{{1, 2}, {3}})...)},
		{NewWeierstrassA0(newPoint(fq2, [][]int64{{0}, {0, 2}})...), NewWeierstrassA0(newPoint(fq2, [][]int64{{0}, {0, 3}})...)},
	}
	if !slices.EqualFunc(er, erWant, func(a, b []*WeierstrassA0[*field.PrimeExt]) bool {
		return slices.EqualFunc(a, b, func(c, d *WeierstrassA0[*field.PrimeExt]) bool { return c.Equal(d) })
	}) {
		t.Errorf("%v != %v", er, erWant)
	}

	// Get the coset.
	rEm := make(map[string]*WeierstrassA0[*field.PrimeExt])
	for _, p := range points {
		rp := multiply(r, p)
		rEm[rp.String()] = rp
	}
	rE := slices.Collect(maps.Values(rEm))
	slices.SortFunc(rE, func(a, b *WeierstrassA0[*field.PrimeExt]) int { return cmp.Compare(a.String(), b.String()) })
	rEWant := []*WeierstrassA0[*field.PrimeExt]{
		rE[0].NewOne(),
		NewWeierstrassA0(newPoint(fq2, [][]int64{{2}, {0}})...),
		NewWeierstrassA0(newPoint(fq2, [][]int64{{4, 2}, {0}})...),
		NewWeierstrassA0(newPoint(fq2, [][]int64{{4, 3}, {0}})...),
	}
	if !slices.EqualFunc(rE, rEWant, func(a, b *WeierstrassA0[*field.PrimeExt]) bool { return a.Equal(b) }) {
		t.Errorf("%v != %v", rE, rEWant)
	}
}

func TestTatePairingNaive(t *testing.T) {
	tests := []struct {
		fieldChar int64
		irr       []int64
		a, b      []int64
		order     int64
		p         [][]int64
		pmul      int64
		q         [][]int64
		qmul      int64
		r         [][]int64
		want      []int64
	}{
		// Example 5.2.2, Pairings for beginners, Craig Costello.
		{
			fieldChar: 5,
			irr:       []int64{2, 0, 1},
			a:         []int64{0}, b: []int64{-3},
			order: 3,
			p:     [][]int64{{3}, {2}},
			pmul:  1,
			q:     [][]int64{{1, 1}, {2, 4}},
			qmul:  1,
			r:     [][]int64{{0, 2}, {2, 1}},
			want:  []int64{2, 1},
		},
		// Example 5.2.3, Pairings for beginners, Craig Costello.
		{
			fieldChar: 19,
			irr:       []int64{1, 0, 1},
			a:         []int64{14}, b: []int64{3},
			order: 5,
			p:     [][]int64{{17}, {9}},
			pmul:  1,
			q:     [][]int64{{16}, {0, 16}},
			qmul:  1,
			want:  []int64{2, 15},
		},
		{
			fieldChar: 19,
			irr:       []int64{1, 0, 1},
			a:         []int64{14}, b: []int64{3},
			order: 5,
			p:     [][]int64{{17}, {9}},
			pmul:  4,
			q:     [][]int64{{16}, {0, 16}},
			qmul:  1,
			want:  []int64{2, 4},
		},
		{
			fieldChar: 19,
			irr:       []int64{1, 0, 1},
			a:         []int64{14}, b: []int64{3},
			order: 5,
			p:     [][]int64{{17}, {9}},
			pmul:  1,
			q:     [][]int64{{16}, {0, 16}},
			qmul:  4,
			want:  []int64{2, 4},
		},
		{
			fieldChar: 19,
			irr:       []int64{1, 0, 1},
			a:         []int64{14}, b: []int64{3},
			order: 5,
			p:     [][]int64{{17}, {9}},
			pmul:  2,
			q:     [][]int64{{16}, {0, 16}},
			qmul:  2,
			want:  []int64{2, 4},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			fieldChar := big.NewInt(test.fieldChar)
			k := field.NewPrimeExt(fieldChar, bigs(test.irr...))
			a, b := k.NewZero().SetCoeffs(bigs(test.a...)...), k.NewZero().SetCoeffs(bigs(test.b...)...)
			order := big.NewInt(test.order)
			p := newEllipticCurve(a).SetCoords(newPoint(k, test.p))
			p = multiply(test.pmul, p)
			q := newEllipticCurve(a).SetCoords(newPoint(k, test.q))
			q = multiply(test.qmul, q)
			want := k.NewZero().SetCoeffs(bigs(test.want...)...)

			var rCoords []*field.PrimeExt
			if test.r == nil {
				rCoords = randEllipticPoints(a, b, 1)[0]
			} else {
				rCoords = newPoint(k, test.r)
			}
			r := newEllipticCurve(a).SetCoords(rCoords)

			w := tatePairingNaive(order, p, q, a, b, r)
			if !w.Equal(want) {
				t.Errorf("tatePairing(%d, %v, %v, %v, %v) = %v != %v", order, p, q, a, b, w, want)
			}
		})
	}
}

func TestMiller(t *testing.T) {
	tests := []struct {
		fieldChar *big.Int
		irr       []*big.Int
		a, b      []*big.Int
		order     *big.Int
		p         [][]*big.Int
		q         [][]*big.Int
		weil      []*big.Int
		tate      []*big.Int
	}{
		// Example 5.3.1, Pairings for beginners, Craig Costello.
		{
			fieldChar: big.NewInt(47),
			irr:       bigs(5, 0, -4, 0, 1),
			a:         bigs(21), b: bigs(15),
			order: big.NewInt(17),
			p:     [][]*big.Int{bigs(45), bigs(23)},
			q:     [][]*big.Int{bigs(29, 0, 31), bigs(0, 11, 0, 35)},
			weil:  bigs(13, 32, 12, 22),
			tate:  bigs(39, 45, 43, 33),
		},
		// Example 6.43, An Introduction to Mathematical Cyrptography, 2nd Ed., J. Hoffstein, J. Phiper, J. Silverman.
		{
			fieldChar: big.NewInt(631),
			irr:       bigs(1, 1),
			a:         bigs(30), b: bigs(34),
			order: big.NewInt(5),
			p:     [][]*big.Int{bigs(36), bigs(60)},
			q:     [][]*big.Int{bigs(121), bigs(387)},
			weil:  bigs(242),
			tate:  bigs(279),
		},
		{
			fieldChar: big.NewInt(631),
			irr:       bigs(1, 1),
			a:         bigs(30), b: bigs(34),
			order: big.NewInt(5),
			p:     [][]*big.Int{bigs(617), bigs(5)},
			q:     [][]*big.Int{bigs(121), bigs(244)},
			weil:  bigs(512),
			tate:  bigs(228),
		},
		// Section 3.10, On the Implementation of Pairing-based Cryptosystems, Ben Lynn, PhD thesis.
		{
			fieldChar: big.NewInt(59),
			irr:       bigs(1, 0, 1),
			a:         bigs(1), b: bigs(0),
			order: big.NewInt(5),
			p:     [][]*big.Int{bigs(25), bigs(30)},
			q:     [][]*big.Int{bigs(-25), bigs(0, 30)},
			weil:  bigs(46, 56),
			tate:  bigs(42, 40),
		},
		// Sage issue 4894, https://github.com/sagemath/sage/issues/4964#issuecomment-1417215833
		{
			fieldChar: big.NewInt(19),
			irr:       bigs(2, 11, 2, 0, 1),
			a:         bigs(-1), b: bigs(0),
			order: big.NewInt(360),
			p:     [][]*big.Int{bigs(14, 10, 12, 3), bigs(13, 10, 4, 5)},
			q:     [][]*big.Int{bigs(11, 4, 15, 12), bigs(6, 0, 15, 9)},
			weil:  bigs(14, 17, 1, 5),
			tate:  bigs(14, 17, 1, 5),
		},
		// Sage https://github.com/sagemath/sage/blob/8bed9c3744bfeaf3a443ad428dbcfe300b1a1b75/src/sage/schemes/elliptic_curves/ell_point.py#L2246
		{
			fieldChar: big.NewInt(65537),
			irr:       bigs(3, -1, 1),
			a:         bigs(0), b: bigs(1),
			order: big.NewInt(7282),
			p:     [][]*big.Int{bigs(22), bigs(28891)},
			q:     [][]*big.Int{bigs(45948), bigs(33436, 64202)},
			weil:  bigs(16502, 49838),
			tate:  bigs(53465, 55425),
		},
		// Sage https://github.com/sagemath/sage/blob/8bed9c3744bfeaf3a443ad428dbcfe300b1a1b75/src/sage/schemes/elliptic_curves/ell_point.py#L2437
		{
			fieldChar: big.NewInt(103),
			irr:       bigs(5, 30, 9, 96, 0, 0, 1),
			a:         bigs(1), b: bigs(18),
			order: big.NewInt(19),
			p:     [][]*big.Int{bigs(33), bigs(91)},
			q:     [][]*big.Int{bigs(6, 36, 60, 38, 58, 31), bigs(2, 10, 1, 88, 72, 86)},
			weil:  bigs(2, 20, 100, 42, 52, 100),
			tate:  bigs(45, 86, 69, 3, 34, 24),
		},
		// PARI/GP test suite, https://github.com/deepin-community/pari/blob/2b9ee5e8fe2997834d548ab7d0a471420f16ba02/src/test/in/ellweilpairing#L56
		{
			fieldChar: big.NewInt(3),
			irr:       bigs(1, 1, 1, 1, 1, 1, 1),
			a:         bigs(1), b: bigs(0),
			order: big.NewInt(28),
			p:     [][]*big.Int{bigs(2, 2, 1, 0, 2, 2), bigs(1, 0, 2, 1, 1, 2)},
			q:     [][]*big.Int{bigs(2, 1, 1, 1, 1, 0), bigs(0, 1, 2, 0, 2, 1)},
			weil:  bigs(0, 1, 0, 1, 1, 2),
			tate:  bigs(2, 2, 1, 0, 0, 2),
		},
		// PARI/GP test suite. The original curve is y^2+y = x^3,
		// which after completing the square (y -> y-1/2) becomes
		// y^2 = x^3 + 1/4, matching the y^2=x^3+ax+b form used here.
		// https://github.com/deepin-community/pari/blob/2b9ee5e8fe2997834d548ab7d0a471420f16ba02/src/test/in/ellweilpairing#L65
		{
			fieldChar: big.NewInt(5),
			irr:       bigs(1, 1, 1, 1, 1, 1, 1),
			a:         bigs(0), b: bigs(4),
			order: big.NewInt(126),
			p:     [][]*big.Int{bigs(0, 2, 2, 3, 4, 4), bigs(2, 0, 2, 2, 4, 4)},
			q:     [][]*big.Int{bigs(1, 4, 2, 4, 2, 1), bigs(4, 3, 0, 0, 1, 2)},
			weil:  bigs(3, 2, 1, 2, 3, 3),
			tate:  bigs(1, 3, 1, 1, 0, 4),
		},
		// PARI/GP test suite, https://github.com/deepin-community/pari/blob/2b9ee5e8fe2997834d548ab7d0a471420f16ba02/src/test/in/ellweilpairing#L95
		{
			fieldChar: Int10("36893488147419103363"),
			irr:       []*big.Int{Int10("36893488147419103362"), Int10("1"), Int10("1")},
			a:         []*big.Int{Int10("1"), Int10("36893488147419103362")}, b: []*big.Int{Int10("0")},
			order: Int10("36893488147419103362"),
			p:     [][]*big.Int{{Int10("24675141949190748313"), Int10("14078684373865444404")}, {Int10("9184592839883218620"), Int10("34082614562121616748")}},
			q:     [][]*big.Int{{Int10("22502667145150289531"), Int10("3606608601291892434")}, {Int10("22495649567796533868"), Int10("23709671617839429105")}},
			weil:  []*big.Int{Int10("9688432087730133707")},
			tate:  []*big.Int{Int10("9688432087730133707")},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExt(test.fieldChar, test.irr)
			a, b := k.NewZero().SetCoeffs(test.a...), k.NewZero().SetCoeffs(test.b...)
			points := make([]*ellipticCurve[*field.PrimeExt], 2)
			for i, cInts := range [][][]*big.Int{test.p, test.q} {
				cs := make([]*field.PrimeExt, len(cInts))
				for j := range cs {
					cs[j] = k.NewZero().SetCoeffs(cInts[j]...)
				}
				points[i] = newEllipticCurve(a).SetCoords(cs)
			}
			p, q := points[0], points[1]

			wantWeil := k.NewZero().SetCoeffs(test.weil...)
			if weil := weilPairing(test.order, p, q, a, b); !weil.Equal(wantWeil) {
				t.Errorf("weilPairing(%d, %v, %v, %v, %v) = %v want %v", test.order, p, q, a, b, weil, wantWeil)
			}

			wantTate := k.NewZero().SetCoeffs(test.tate...)
			if tate := tatePairing(test.order, p, q, a, b); !tate.Equal(wantTate) {
				t.Errorf("tatePairing(%d, %v, %v, %v, %v) = %v want %v", test.order, p, q, a, b, tate, wantTate)
			}
		})
	}
}

func TestTatePairing(t *testing.T) {
	tests := []struct {
		fieldChar int64
		irr       []int64
		a, b      []int64
		order     int64
		p         [][]int64
		q         [][]int64
		weil      []int64
		tate      []int64
	}{
		// Sage https://github.com/sagemath/sage/blob/8bed9c3744bfeaf3a443ad428dbcfe300b1a1b75/src/sage/schemes/elliptic_curves/ell_point.py#L2490
		{
			fieldChar: 65537,
			irr:       []int64{3, 46810, 1},
			a:         []int64{0}, b: []int64{1},
			order: 7282,
			p:     [][]int64{{22}, {28891}},
			q:     [][]int64{{-93}, {31573, 40438}},
			tate:  []int64{4063, 34585},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			fieldChar := big.NewInt(test.fieldChar)
			k := field.NewPrimeExt(fieldChar, bigs(test.irr...))
			a, b := k.NewZero().SetCoeffs(bigs(test.a...)...), k.NewZero().SetCoeffs(bigs(test.b...)...)
			order := big.NewInt(test.order)
			p := newEllipticCurve(a).SetCoords(newPoint(k, test.p))
			q := newEllipticCurve(a).SetCoords(newPoint(k, test.q))

			wantTate := k.NewZero().SetCoeffs(bigs(test.tate...)...)
			if tate := tatePairing(order, p, q, a, b); !tate.Equal(wantTate) {
				t.Errorf("tatePairing(%d, %v, %v, %v, %v) = %v want %v", order, p, q, a, b, tate, wantTate)
			}
		})
	}
}

func TestAtePairing(t *testing.T) {
	tests := []struct {
		fieldChar *big.Int
		irr       []*big.Int
		a, b      []*big.Int
		order     *big.Int
		trace     *big.Int
		q         [][]*big.Int
		p         [][]*big.Int
		ate       []*big.Int
	}{
		// Example 7.3.3, Pairings for beginners, Craig Costello.
		{
			fieldChar: big.NewInt(47),
			irr:       bigs(5, 0, -4, 0, 1),
			a:         bigs(21), b: bigs(15),
			order: big.NewInt(17),
			trace: big.NewInt(-3),
			q:     [][]*big.Int{bigs(29, 0, 31), bigs(0, 11, 0, 35)},
			p:     [][]*big.Int{bigs(45), bigs(23)},
			// When trace is negative, Sage and Magma perform an
			// additional inverse on the output.
			// Thus, we use the value Inv(25, 25, 37, 21) here.
			ate: bigs(25, 22, 37, 26),
		},
		// Sage https://github.com/sagemath/sage/blob/8bed9c3744bfeaf3a443ad428dbcfe300b1a1b75/src/sage/schemes/elliptic_curves/ell_point.py#L2632
		{
			fieldChar: big.NewInt(7549),
			irr:       bigs(2, 0, 0, 0, 0, 0, 1),
			a:         bigs(0), b: bigs(1),
			order: big.NewInt(157),
			trace: big.NewInt(14),
			q:     [][]*big.Int{bigs(0, 0, 0, 0, 6908), bigs(0, 0, 0, 3231)},
			p:     [][]*big.Int{bigs(3050), bigs(5371)},
			ate:   bigs(6733, 4022, 2064, 4350, 4230, 6708),
		},
		// Sage https://github.com/sagemath/sage/blob/8bed9c3744bfeaf3a443ad428dbcfe300b1a1b75/src/sage/schemes/elliptic_curves/ell_point.py#L2648
		{
			fieldChar: big.NewInt(2213),
			irr:       bigs(2, 0, 0, 0, 0, 0, 0, 1),
			a:         bigs(1), b: bigs(49),
			order: big.NewInt(1093),
			trace: big.NewInt(28),
			q: [][]*big.Int{
				bigs(722, 1883, 1592, 980, 245, 1767, 1729),
				bigs(1636, 309, 1457, 1513, 1030, 1877, 1299),
			},
			p:   [][]*big.Int{bigs(1583), bigs(1734)},
			ate: bigs(654, 2151, 2134, 239, 1979, 1538, 1665),
		},
		// Sage https://github.com/sagemath/sage/blob/8bed9c3744bfeaf3a443ad428dbcfe300b1a1b75/src/sage/schemes/elliptic_curves/ell_point.py#L2667
		{
			fieldChar: big.NewInt(2017),
			irr:       bigs(2, 0, 0, 0, 0, 0, 0, 1),
			a:         bigs(1), b: bigs(30),
			order: big.NewInt(29),
			trace: big.NewInt(-70),
			q: [][]*big.Int{
				bigs(770, 867, 1750, 1791, 660, 1778, 1226),
				bigs(1712, 273, 1200, 406, 1206, 198, 1764),
			},
			p:   [][]*big.Int{bigs(369), bigs(716)},
			ate: bigs(1315, 1905, 1950, 488, 576, 1161, 1794),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExt(test.fieldChar, test.irr)
			a, b := k.NewZero().SetCoeffs(test.a...), k.NewZero().SetCoeffs(test.b...)
			points := make([]*ellipticCurve[*field.PrimeExt], 2)
			for i, cInts := range [][][]*big.Int{test.q, test.p} {
				cs := make([]*field.PrimeExt, len(cInts))
				for j := range cs {
					cs[j] = k.NewZero().SetCoeffs(cInts[j]...)
				}
				points[i] = newEllipticCurve(a).SetCoords(cs)
			}
			q, p := points[0], points[1]

			wantAte := k.NewZero().SetCoeffs(test.ate...)
			if ate := AtePairing(test.order, test.trace, q, p, a, b); !ate.Equal(wantAte) {
				t.Errorf("AtePairing(%d, %d, %v, %v, %v, %v) = %v want %v", test.order, test.trace, q, p, a, b, ate, wantAte)
			}
		})
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}

func bigs(is ...int64) []*big.Int {
	bs := make([]*big.Int, len(is))
	for i := range bs {
		bs[i] = big.NewInt(is[i])
	}
	return bs
}

func multiply[G nag.Group[G]](n int64, x G) G {
	return nag.Pow(x.NewOne().Set(x), big.NewInt(n))
}

func parse[K field.Finite[K]](variables map[string]nag.Symbol, k K, s string) *nag.Polynomial[K] {
	p, err := field.Parse(variables, k, s)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return p
}

func newPoint(k *field.PrimeExt, is [][]int64) []*field.PrimeExt {
	p := make([]*field.PrimeExt, len(is))
	for i := range p {
		cs := bigs(is[i]...)
		p[i] = k.NewZero().SetCoeffs(cs...)
	}
	return p
}

func newPointP(primitiveFormat bool, k *field.PrimeExt, is [][]int64) []*field.PrimeExt {
	if primitiveFormat {
		if len(is[0]) == 2 {
			return newPoint(k, is)
		}
		primitive := k.NewZero().SetCoeffs(big.NewInt(0), big.NewInt(1))
		pnt := make([]*field.PrimeExt, len(is))
		for i := range pnt {
			exponent := big.NewInt(is[i][0])
			pnt[i] = nag.Pow(k.NewZero().Set(primitive), exponent)
		}
		return pnt
	}
	return newPoint(k, is)
}

func dumpPoints(fpath string, points [][]*field.PrimeExt) {
	var records [][]string
	for _, p := range points {
		var line []string
		for _, x := range p {
			for _, c := range x.Coeffs() {
				line = append(line, c.String())
			}
		}
		records = append(records, line)
	}

	buf := bytes.NewBuffer(nil)
	w := csv.NewWriter(buf)
	w.WriteAll(records)
	if err := w.Error(); err != nil {
		panic(err)
	}
	if err := os.WriteFile(fpath, buf.Bytes(), 0644); err != nil {
		panic(err)
	}
}

func findRoots[K field.Finite[K]](poly *nag.Polynomial[K]) []K {
	k := poly.LeadingTerm().Coefficient
	order := new(big.Int).Exp(k.Characteristic(), big.NewInt(int64(k.Degree())), nil)
	roots := make([]K, 0)
	for i := big.NewInt(0); i.Cmp(order) < 0; i.Add(i, big.NewInt(1)) {
		x := setIth(k.NewZero(), i)
		y := EvalPoly(poly, []K{x})
		if y.Equal(k.NewZero()) {
			roots = append(roots, x)
		}
	}
	return roots
}

func mumfordRep[K nag.Field[K]](d [][]K) [2]*nag.Polynomial[K] {
	// Create u = (x - d[0].x) * (x - d[1].x)....
	k := d[0][0].NewZero()
	u := nag.NewPolynomial(k, nag.Deglex, nag.PolynomialTerm[K]{Coefficient: k.NewOne()})
	u.SymbolStringer = func(s nag.Symbol) string { return "x" }
	for i := range d {
		neg1c := u.Field().NewZero()
		neg1c.Sub(neg1c, u.Field().NewOne())
		neg1c.Mul(neg1c, d[i][0])
		pntPoly := nag.NewPolynomial(u.Field(), u.Order(),
			nag.PolynomialTerm[K]{Coefficient: u.Field().NewOne(), Monomial: make([]nag.Symbol, 1)},
			nag.PolynomialTerm[K]{Coefficient: neg1c},
		)
		u.Mul(u, pntPoly)
	}

	v := FitPoints(d)
	return [2]*nag.Polynomial[K]{u, v}
}

func cantorReduceElliptic[K nag.Field[K]](elliptic *nag.Polynomial[K], d [2]*nag.Polynomial[K]) [2]*nag.Polynomial[K] {
	// Roots of u(x) are the points of the divisor d.
	// v(x) returns the y-coordinate of the points of the divisor d.
	u, v := d[0], d[1]

	// Compute v2MinusElliptic = v^2 - elliptic.
	// Let d-curve be the curve interpolating the points in the divisor d.
	// The roots of v2MinusElliptic(x) = 0 are the x-coordinates of the intersection of d-curve and elliptic.
	v2MinusElliptic := nag.NewPolynomial(v.Field(), v.Order()).Set(v)
	v2MinusElliptic.Mul(v, v)
	neg1c := v.Field().NewZero()
	neg1c.Sub(neg1c, v.Field().NewOne())
	neg1 := nag.NewPolynomial(v.Field(), v.Order(), nag.PolynomialTerm[K]{Coefficient: neg1c})
	neg1.Mul(neg1, elliptic)
	v2MinusElliptic.Add(v2MinusElliptic, neg1)

	// Find the extra leftover points of the intersection that do not belong to the divisor d.
	// leftover(x) = 0 contains the x-coordinate of the leftover points.
	// deg(leftover) is the number of leftover points.
	leftover, _ := Divide(v2MinusElliptic, u)

	// Simplify v(x) for the leftover points.
	_, leftoverV := Divide(v, leftover)
	return [2]*nag.Polynomial[K]{leftover, leftoverV}
}

func getTorsion[K field.Finite[K], E CurvePoint[E, K]](r int64, points []E) [][]E {
	er := make([][]E, 0)
	for _, p := range points {
		if p.Equal(p.NewOne()) {
			continue
		}
		if !multiply(r, p).Equal(p.NewOne()) {
			continue
		}

		// Attempt to insert into an existing group.
		insertExisting := false
		for i := range er {
			inGroup := false
			for j := range r {
				jp := multiply(j, p)
				if slices.ContainsFunc(er[i], func(a E) bool { return a.Equal(jp) }) {
					inGroup = true
					break
				}
			}
			if inGroup {
				er[i] = append(er[i], p)
				insertExisting = true
			}
		}
		// p belongs to a new subgroup.
		if !insertExisting {
			er = append(er, []E{p})
		}
	}

	// Find G1 where the trace acts like a multiplication.
	k := er[0][0].Coords()[0]
	g1Idx := -1
	for i := range er {
		p := er[i][0]
		mp := multiply(int64(k.Degree()), p)
		tp := Trace(p)
		if mp.Equal(tp) {
			g1Idx = i
			break
		}
	}
	er[0], er[g1Idx] = er[g1Idx], er[0]

	// Find G1 where the trace sends the subgroup to O.
	g2Idx := -1
	for i := range er {
		p := er[i][0]
		if Trace(p).Equal(p.NewOne()) {
			g2Idx = i
			break
		}
	}
	er[len(er)-1], er[g2Idx] = er[g2Idx], er[len(er)-1]

	return er
}
