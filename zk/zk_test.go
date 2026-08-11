package zk

import (
	"bytes"
	"cmp"
	"crypto/rand"
	"flag"
	"fmt"
	"log"
	"math/big"
	"slices"
	"testing"

	bn256 "github.com/ethereum/go-ethereum/crypto/bn256/cloudflare"
	"github.com/fumin/learn/zk/bn128"
	"github.com/fumin/learn/zk/ecc"
	"github.com/fumin/nag"
	"github.com/fumin/nag/field"
)

func TestRS2_9(t *testing.T) {
	// Define the circuit:
	//     x^3 + 5*x + -2 == y
	//     y^3 == 3241792
	r1cs := newRS2_3_R1CS()
	numConstraints := len(r1cs.L)
	k := field.NewPrimeExtDeg(bn256.Order, 1)
	qap := qapFromR1CS(k, r1cs)

	// tau must not be in the set {1, 2, ..., numConstraints}, since this
	// set is where the QAP polynomials are interpolated.
	// Let the QAP polynomial equation be u*v = w + h*t.
	// If tau is in this set, then qap.T(tau) = 0, and thus h(x) * t(x) = 0 regardless of h(x).
	// Moreover, when tau is in this set, u(tau) = l[tau]*witness,
	// v(tau) = r[tau]*witness,... where l, r are the R1CS matrices.
	// Therefore, if tau is say 3, then the proof collapses to checking
	// only constraint 3, and ignoring the other constraints.
	n := new(big.Int).SetInt64(int64(numConstraints + 1))
	tau, _ := rand.Int(rand.Reader, new(big.Int).Sub(bn256.Order, n))
	tau.Add(tau, n)
	srs := newStructuredRefStr(tau, numConstraints, qap.T)

	// Check that wrong proofs are not verified.
	const correctX, correctY = 5, 148
	wrongs := [][2]int{
		{1, 4},   // passes only constraint 0
		{1, 148}, // passes only constraint 1
	}
	for x := correctX - 3; x <= correctX+3; x++ {
		for y := correctY - 3; y <= correctY+3; y++ {
			if !(x == correctX && y == correctY) {
				wrongs = append(wrongs, [2]int{x, y})
			}
		}
	}
	for _, bad := range wrongs {
		proof := newProof(r1cs.ToWitness(bad[0], bad[1]), qap, srs)
		if verify(proof) {
			t.Errorf("wrong proofs should not be verified")
			return
		}
	}

	witness := r1cs.ToWitness(correctX, correctY)
	proof := newProof(witness, qap, srs)
	if !verify(proof) {
		t.Errorf("correct proofs should be verified")
	}
}

func TestRS2_7(t *testing.T) {
	r1cs := newRS2_7_R1CS()
	k := field.NewPrimeExtDeg(big.NewInt(79), 1)
	qap := qapFromR1CS(k, r1cs)

	witness := r1cs.ToWitness(4, -2, -64)
	u, v, w, h := mulWitnessQAP(witness, qap)

	if h.String() != "68x^2+17x+59" {
		t.Errorf("%v", h)
	}
	// Since witness is the correct solution, u*v = w +h*t should hold.
	uv := nag.NewPolynomial(k, u.Order()).Mul(u, v)
	wht := nag.NewPolynomial(k, u.Order()).Mul(h, qap.T)
	wht.Add(wht, w)
	if !uv.Equal(wht) {
		t.Errorf("%v != %v", uv, wht)
	}
}

func TestRS2_3(t *testing.T) {
	// Define the circuit:
	//     x^3 + 5*x + -2 == y
	//     y^3 == 3241792
	circuit := newRS2_3_R1CS()

	// Check that wrong proofs are not verified.
	const correctX, correctY = 5, 148
	wrongs := [][2]int{
		{1, 4},   // passes only constraint 0
		{1, 148}, // passes only constraint 1
	}
	for x := correctX - 3; x <= correctX+3; x++ {
		for y := correctY - 3; y <= correctY+3; y++ {
			if !(x == correctX && y == correctY) {
				wrongs = append(wrongs, [2]int{x, y})
			}
		}
	}
	for _, bad := range wrongs {
		proof := newR1CSProof(circuit.ToWitness(bad[0], bad[1]))
		if verifyR1CS(circuit, proof) {
			t.Errorf("wrong proofs should not be verified")
		}
	}

	solution := newR1CSProof(circuit.ToWitness(correctX, correctY))
	if !verifyR1CS(circuit, solution) {
		t.Errorf("correct proofs should be verified")
	}
}

func TestRS2_1(t *testing.T) {
	a := []*bn256.G1{
		bn128.NewG1([]*big.Int{
			ecc.Int10("3010198690406615200373504922352659861758983907867017329644089018310584441462"),
			ecc.Int10("17861058253836152797273815394432013122766662423622084931972383889279925210507"),
		}),
		bn128.NewG1([]*big.Int{
			ecc.Int10("4503322228978077916651710446042370109107355802721800704639343137502100212473"),
			ecc.Int10("6132642251294427119375180147349983541569387941788025780665104001559216576968"),
		}),
	}
	b := []*bn256.G2{
		bn128.NewG2([]*big.Int{
			ecc.Int10("2725019753478801796453339367788033689375851816420509565303521482350756874229"),
			ecc.Int10("7273165102799931111715871471550377909735733521218303035754523677688038059653"),
			ecc.Int10("2512659008974376214222774206987427162027254181373325676825515531566330959255"),
			ecc.Int10("957874124722006818841961785324909313781880061366718538693995380805373202866"),
		}),
		bn128.NewG2([]*big.Int{
			ecc.Int10("18029695676650738226693292988307914797657423701064905010927197838374790804409"),
			ecc.Int10("14583779054894525174450323658765874724019480979794335525732096752006891875705"),
			ecc.Int10("2140229616977736810657479771656733941598412651537078903776637920509952744750"),
			ecc.Int10("11474861747383700316476719153975578001603231366361248090558603872215261634898"),
		}),
	}
	if !bn256.PairingCheck(a, b) {
		t.Errorf("pairing check failed")
	}
}

func TestRS1_3_PolyRoot(t *testing.T) {
	order := big.NewInt(int64(103))
	k := field.NewPrimeExtDeg(order, 1)
	p3 := parse(k, "-x + 1")
	p4 := parse(k, "-x + 2")
	p34 := nag.NewPolynomial(k.NewZero(), nag.Deglex).Set(p3).Mul(p3, p4)
	p34Want := parse(k, "x^2+100x+2")
	if !p34.Equal(p34Want) {
		t.Errorf("%s != %s", p34, p34Want)
	}

	tests := []struct {
		order   *big.Int
		p       string
		factors []factorStr
	}{
		{
			order: order,
			p:     p34.String(),
			factors: []factorStr{
				{p: "x-2", n: 1},
				{p: "x-1", n: 1},
			},
		},
		{
			order: ecc.Int10("21888242871839275222246405745257275088548364400416034343698204186575808495617"),
			p:     "x^2+2x-8",
			factors: []factorStr{
				{p: "x-2", n: 1},
				{p: "x+4", n: 1},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExtDeg(test.order, 1)
			factors := field.Factor(parse(k, test.p))
			slices.SortFunc(factors, func(a, b field.IrrFactor[*field.PrimeExt]) int { return cmp.Compare(a.P.String(), b.P.String()) })
			want := make([]field.IrrFactor[*field.PrimeExt], 0, len(test.factors))
			for _, f := range test.factors {
				want = append(want, field.IrrFactor[*field.PrimeExt]{P: parse(k, f.p), N: big.NewInt(int64(f.n))})
			}
			if !slices.EqualFunc(factors, want, func(a, b field.IrrFactor[*field.PrimeExt]) bool { return a.P.Equal(b.P) && a.N.Cmp(b.N) == 0 }) {
				t.Errorf("%v != %v", factors, want)
			}
		})
	}
}

func TestRS1_3_Ex3(t *testing.T) {
	// y = -1/2x + 3/2
	// y = -4/8x + 1/8
	e := newField(11, 1)
	x0 := div(neg(e[1]), e[2])
	c0 := div(e[3], e[2])
	x1 := div(neg(e[4]), e[8])
	c1 := div(e[1], e[8])
	if !x0.Equal(x1) {
		t.Errorf("%v != %v", x0, x1)
	}
	if !c0.Equal(c1) {
		t.Errorf("%v != %v", c0, c1)
	}
}

// TestRS1_3_FFSquareRoot is from Module 1, Section 3, in the RareSkills book of Zero Knowledge:
// https://rareskills.io/zk-book
func TestRS1_3_FFSquareRoot(t *testing.T) {
	e := newField(11, 1)
	tests := []struct {
		e   *field.PrimeExt
		sr0 *field.PrimeExt
		sr1 *field.PrimeExt
	}{
		{e: e[0], sr0: e[0], sr1: nil},
		{e: e[1], sr0: e[1], sr1: e[10]},
		{e: e[3], sr0: e[5], sr1: e[6]},
		{e: e[4], sr0: e[2], sr1: e[9]},
		{e: e[5], sr0: e[4], sr1: e[7]},
		{e: e[9], sr0: e[3], sr1: e[8]},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			sq0 := mul(test.sr0, test.sr0)
			if !sq0.Equal(test.e) {
				t.Errorf("%v^2 = %v, want %v", test.sr0, sq0, test.e)
			}
			if test.sr1 != nil {
				sq1 := mul(test.sr1, test.sr1)
				if !sq1.Equal(test.e) {
					t.Errorf("%v^2 = %v, want %v", test.sr1, sq1, test.e)
				}
			}
		})
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}

type StructuredReferenceStrings struct {
	Omega []*bn256.G1
	Theta []*bn256.G2
	Y     []*bn256.G1
}

func newStructuredRefStr[K field.Finite[K]](tau *big.Int, numConstraints int, tPoly *nag.Polynomial[K]) StructuredReferenceStrings {
	iBig, tauI := new(big.Int), new(big.Int)

	tauK := tPoly.Field().NewZero().SetCoeffs(tau)
	tTau := ecc.EvalPoly(tPoly, []K{tauK}).Coeffs()[0]

	srs := StructuredReferenceStrings{
		Omega: make([]*bn256.G1, numConstraints),
		Theta: make([]*bn256.G2, numConstraints),
		Y:     make([]*bn256.G1, numConstraints),
	}
	for i := int64(numConstraints - 1); i >= 0; i-- {
		tauI.Exp(tau, iBig.SetInt64(i), nil)
		srs.Omega[i] = new(bn256.G1).ScalarBaseMult(tauI)
		srs.Theta[i] = new(bn256.G2).ScalarBaseMult(tauI)

		tauI.Mul(tauI, tTau)
		srs.Y[i] = new(bn256.G1).ScalarBaseMult(tauI)
	}
	return srs
}

type QAP[K nag.Field[K]] struct {
	U []*nag.Polynomial[K]
	V []*nag.Polynomial[K]
	W []*nag.Polynomial[K]
	T *nag.Polynomial[K]
}

func qapFromR1CS[K field.Finite[K]](k K, r1cs r1CS) QAP[K] {
	numVars := len(r1cs.L[0])
	qap := QAP[K]{
		U: make([]*nag.Polynomial[K], numVars),
		V: make([]*nag.Polynomial[K], numVars),
		W: make([]*nag.Polynomial[K], numVars),
	}

	// Compute U, V, and W.
	numConstraints := len(r1cs.L)
	points := make([][]K, numConstraints)
	for i := range points {
		points[i] = []K{k.NewZero(), k.NewZero()}
	}
	polys := [3][]*nag.Polynomial[K]{qap.U, qap.V, qap.W}
	mats := [3][][]int{r1cs.L, r1cs.R, r1cs.O}
	for pi, poly := range polys {
		mat := mats[pi]

		for j := range numVars {
			for i := range points {
				points[i][0].SetCoeffs(big.NewInt(int64(i + 1)))
				points[i][1].SetCoeffs(big.NewInt(int64(mat[i][j])))
			}
			poly[j] = ecc.FitPoints(points)
		}
	}

	// Compute T.
	zero, one := k.NewZero(), k.NewOne()
	negXi := k.NewZero()
	qap.T = nag.NewPolynomial(k, qap.U[0].Order(), nag.PolynomialTerm[K]{Coefficient: one})
	for i := range points {
		negXi.Sub(zero, points[i][0])
		xiP := nag.NewPolynomial(k, qap.T.Order(),
			nag.PolynomialTerm[K]{Coefficient: one, Monomial: make([]nag.Symbol, 1)},
			nag.PolynomialTerm[K]{Coefficient: negXi})
		qap.T.Mul(qap.T, xiP)
	}

	return qap
}

func mulWitnessQAP[K field.Finite[K]](witnessI []int, qap QAP[K]) (*nag.Polynomial[K], *nag.Polynomial[K], *nag.Polynomial[K], *nag.Polynomial[K]) {
	k := qap.U[0].Field()
	order := qap.U[0].Order()

	witness := make([]K, len(witnessI))
	for i, w := range witnessI {
		witness[i] = k.NewZero().SetCoeffs(big.NewInt(int64(w)))
	}

	uw := nag.NewPolynomial(k, order)
	vw := nag.NewPolynomial(k, order)
	ww := nag.NewPolynomial(k, order)
	uvw := [][]*nag.Polynomial[K]{qap.U, qap.V, qap.W}
	for i, sum := range []*nag.Polynomial[K]{uw, vw, ww} {
		u := uvw[i]
		for j := range witness {
			a := nag.NewPolynomial(k, order, nag.PolynomialTerm[K]{Coefficient: witness[j]})
			a.Mul(a, u[j])
			sum.Add(sum, a)
		}
	}

	neg1 := k.NewZero()
	neg1.Sub(neg1, k.NewOne())
	negW := nag.NewPolynomial(k, order, nag.PolynomialTerm[K]{Coefficient: neg1})
	negW.Mul(negW, ww)
	uvnw := nag.NewPolynomial(k, order).Mul(uw, vw)
	uvnw.Add(uvnw, negW)
	h, _ := ecc.Divide(uvnw, qap.T)

	return uw, vw, ww, h
}

func g1Product[K field.Finite[K]](sum *bn256.G1, poly *nag.Polynomial[K], powers []*bn256.G1) {
	e := new(bn256.G1)
	for cF, m := range poly.Terms() {
		c := cF.Coeffs()[0]
		deg := len(m)

		e.Set(powers[deg])
		e.ScalarMult(e, c)
		sum.Add(sum, e)
	}
}

func g2Product[K field.Finite[K]](sum *bn256.G2, poly *nag.Polynomial[K], powers []*bn256.G2) {
	e := new(bn256.G2)
	for cF, m := range poly.Terms() {
		c := cF.Coeffs()[0]
		deg := len(m)

		e.Set(powers[deg])
		e.ScalarMult(e, c)
		sum.Add(sum, e)
	}
}

type Proof struct {
	A *bn256.G1
	B *bn256.G2
	C *bn256.G1
}

func newProof[K field.Finite[K]](witness []int, qap QAP[K], srs StructuredReferenceStrings) Proof {
	u, v, w, h := mulWitnessQAP(witness, qap)

	proof := Proof{
		A: new(bn256.G1).ScalarBaseMult(bn256.Order),
		B: new(bn256.G2).ScalarBaseMult(bn256.Order),
		C: new(bn256.G1).ScalarBaseMult(bn256.Order),
	}

	g1Product(proof.A, u, srs.Omega)
	g2Product(proof.B, v, srs.Theta)
	g1Product(proof.C, w, srs.Omega)
	g1Product(proof.C, h, srs.Y)

	return proof
}

func verify(proof Proof) bool {
	negG2Scalar := new(big.Int).Sub(bn256.Order, big.NewInt(1))
	negG2 := new(bn256.G2).ScalarBaseMult(negG2Scalar)

	a := []*bn256.G1{proof.A, proof.C}
	b := []*bn256.G2{proof.B, negG2}
	return bn256.PairingCheck(a, b)
}

type r1csProof struct {
	G1 []*bn256.G1
	G2 []*bn256.G2
}

func newR1CSProof(witness []int) r1csProof {
	proof := r1csProof{
		G1: make([]*bn256.G1, len(witness)),
		G2: make([]*bn256.G2, len(witness)),
	}
	for i, w := range witness {
		proof.G1[i] = new(bn256.G1).ScalarBaseMult(big.NewInt(int64(w)))
		proof.G2[i] = new(bn256.G2).ScalarBaseMult(big.NewInt(int64(w)))
	}
	return proof
}

type r1CS struct {
	ToWitness func(xs ...int) []int
	L         [][]int
	R         [][]int
	O         [][]int
}

func verifyR1CS(circuit r1CS, proof r1csProof) bool {
	// Allocate memory.
	bi := new(big.Int)
	g1, g1Sum := new(bn256.G1), new(bn256.G1)
	g2, g2Sum := new(bn256.G2), new(bn256.G2)
	// Prepare the identity element of GT.
	g1.ScalarBaseMult(bn256.Order)
	g2.ScalarBaseMult(bn256.Order)
	gTOne := bn256.Pair(g1, g2).Marshal()

	// Check that G1 and G2 in the proof represent the same integer.
	numVars := len(circuit.L[0])
	for j := range numVars {
		gT1 := bn256.Pair(proof.G1[j], g2)
		gT2 := bn256.Pair(g1, proof.G2[j])
		gT1.Add(gT1, gT2.Neg(gT2))

		if !bytes.Equal(gT1.Marshal(), gTOne) {
			return false
		}
	}

	// Check that all constraints are satisfied.
	numConstraints := len(circuit.L)
	for i := range numConstraints {
		// Add circuit L and R.
		g1Sum.ScalarBaseMult(bn256.Order)
		g2Sum.ScalarBaseMult(bn256.Order)
		for j := range numVars {
			bi.SetInt64(int64(circuit.L[i][j]))
			g1.Set(proof.G1[j]).ScalarMult(g1, bi)
			g1Sum.Add(g1Sum, g1)

			bi.SetInt64(int64(circuit.R[i][j]))
			g2.Set(proof.G2[j]).ScalarMult(g2, bi)
			g2Sum.Add(g2Sum, g2)
		}
		gT := bn256.Pair(g1Sum, g2Sum)

		// Add circuit O.
		g1Sum.ScalarBaseMult(bn256.Order)
		for j := range numVars {
			bi.SetInt64(int64(circuit.O[i][j]))
			g1.Set(proof.G1[j]).ScalarMult(g1, bi)
			g1Sum.Add(g1Sum, g1)
		}
		g2.ScalarBaseMult(big.NewInt(1))
		gTO := bn256.Pair(g1Sum, g2)
		gT.Add(gT, gTO.Neg(gTO))

		if !bytes.Equal(gT.Marshal(), gTOne) {
			return false
		}
	}
	return true
}

func newRS2_3_R1CS() r1CS {
	// Define the circuit of the satisfiability of the below equations:
	//     x^3 + 5*x + -2 == y
	//     y^3 == 3241792
	//
	// It is based on Chapter 3, Module 2 of the RareSkills Zero-Knowledge book.
	// https://rareskills.io/post/r1cs-zkp
	circuit := r1CS{}
	// Use the circom script testdata/rs23.circom to generate the r1CS
	// constraints, which are encoded below:
	//
	// Represent circom variables.
	const one, x, x2, x3, y, y2 = 0, 1, 2, 3, 4, 5
	const numVars = 6
	circuit.ToWitness = func(vs ...int) []int {
		wx, wy := vs[0], vs[1]
		witness := make([]int, numVars)
		witness[one] = 1
		witness[x] = wx
		witness[x2] = wx * wx
		witness[x3] = wx * wx * wx
		witness[y] = wy
		witness[y2] = wy * wy
		return witness
	}

	const numConstraints = 5
	circuit.L = make([][]int, numConstraints)
	circuit.R = make([][]int, numConstraints)
	circuit.O = make([][]int, numConstraints)
	for i := range numConstraints {
		circuit.L[i] = make([]int, numVars)
		circuit.R[i] = make([]int, numVars)
		circuit.O[i] = make([]int, numVars)
	}

	// Encode the circom r1CS constraints.
	// Constraint 0.
	circuit.L[0][x] = -1
	circuit.R[0][x] = 1
	circuit.O[0][x2] = -1
	// Constraint 1.
	circuit.L[1][x2] = -1
	circuit.R[1][x] = 1
	circuit.O[1][x3] = -1
	// Constraint 2.
	circuit.L[2][y] = -1
	circuit.R[2][y] = 1
	circuit.O[2][y2] = -1
	// Constraint 3.
	circuit.L[3][y2] = -1
	circuit.R[3][y] = 1
	circuit.O[3][one] = -3241792
	// Constraint 4.
	circuit.O[4][one] = 2
	circuit.O[4][x] = -5
	circuit.O[4][y] = 1
	circuit.O[4][x3] = -1

	return circuit
}

func newRS2_7_R1CS() r1CS {
	// Circuit equation:
	//     z = x^4 - 5y^2x^2
	// It is based on Chapter 7, Module 2 of the RareSkills Zero-Knowledge book.
	// https://rareskills.io/post/r1cs-to-qap
	circuit := r1CS{}
	const one, z, x, y, v1, v2, v3 = 0, 1, 2, 3, 4, 5, 6
	const numVars = 7
	circuit.ToWitness = func(vs ...int) []int {
		wx, wy, wz := vs[0], vs[1], vs[2]
		witness := make([]int, numVars)
		witness[one] = 1
		witness[z] = wz
		witness[x] = wx
		witness[y] = wy
		witness[v1] = wx * wx
		witness[v2] = witness[v1] * witness[v1]
		witness[v3] = -5 * wy * wy
		return witness
	}

	const numConstraints = 4
	circuit.L = make([][]int, numConstraints)
	circuit.R = make([][]int, numConstraints)
	circuit.O = make([][]int, numConstraints)
	for i := range numConstraints {
		circuit.L[i] = make([]int, numVars)
		circuit.R[i] = make([]int, numVars)
		circuit.O[i] = make([]int, numVars)
	}

	// Constraint 0.
	circuit.L[0][x] = 1
	circuit.R[0][x] = 1
	circuit.O[0][v1] = 1
	// Constraint 1.
	circuit.L[1][v1] = 1
	circuit.R[1][v1] = 1
	circuit.O[1][v2] = 1
	// Constraint 2.
	circuit.L[2][y] = -5
	circuit.R[2][y] = 1
	circuit.O[2][v3] = 1
	// Constraint 4.
	circuit.L[3][v3] = 1
	circuit.R[3][v1] = 1
	circuit.O[3][z] = 1
	circuit.O[3][v2] = -1

	return circuit
}

func newField(p, n int) []*field.PrimeExt {
	k := field.NewPrimeExtDeg(big.NewInt(int64(p)), n)
	order := new(big.Int).Exp(k.Characteristic(), big.NewInt(int64(k.Degree())), nil).Int64()
	e := make([]*field.PrimeExt, 0, order)
	for i := range order {
		e = append(e, setIth(k.NewZero(), big.NewInt(i)))
	}
	return e
}

func neg(a *field.PrimeExt) *field.PrimeExt {
	return sub(a.NewZero(), a)
}

func sub(a, b *field.PrimeExt) *field.PrimeExt {
	return a.NewZero().Sub(a, b)
}

func mul(a, b *field.PrimeExt) *field.PrimeExt {
	return a.NewZero().Mul(a, b)
}

func div(a, b *field.PrimeExt) *field.PrimeExt {
	return a.NewZero().Div(a, b)
}

func parse[K field.Finite[K]](k K, s string) *nag.Polynomial[K] {
	vs := map[string]nag.Symbol{"x": 0}
	p, err := field.Parse(vs, k, s)
	if err != nil {
		panic(err)
	}
	return p
}

type factorStr struct {
	p string
	n int
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
