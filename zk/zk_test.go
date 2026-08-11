package zk

import (
	"bytes"
	"cmp"
	"crypto/md5"
	"crypto/rand"
	_ "embed"
	"flag"
	"fmt"
	"log"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	bn256 "github.com/ethereum/go-ethereum/crypto/bn256/cloudflare"
	"github.com/fumin/learn/zk/bn128"
	"github.com/fumin/learn/zk/circom"
	"github.com/fumin/learn/zk/circom/witnesscalc"
	"github.com/fumin/learn/zk/ecc"
	groth16 "github.com/fumin/learn/zk/groth16/bn254"
	"github.com/fumin/nag"
	"github.com/fumin/nag/field"
	"github.com/pkg/errors"
)

//go:embed testdata/rs3_21/rs3_21.circom
var rs3_21Circom []byte

func TestRS3_21(t *testing.T) {
	// Compile script if circom is installed.
	workDir, _ := os.Getwd()
	linkLibs := []string{filepath.Join(workDir, "testdata")}
	cmpl, err := circom.Compile(rs3_21Circom, linkLibs)
	if err != nil {
		return
	}

	// Create structured reference strings.
	r1cs, err := circom.ParseR1CS(bytes.NewReader(cmpl.R1CS))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	tw := newToxicWasteRS2_10()
	srs := newStructuredRefStrRS2_10_V2(tw, r1cs)

	// Define witness function.
	newWitness := func(preimage []byte, image *big.Int) ([]*big.Int, error) {
		witCalc, err := witnesscalc.NewCircom2WitnessCalculator(cmpl.WitnessWasm, true)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		inputs := map[string]any{
			"preimage": make([]*big.Int, len(preimage)),
			"image":    new(big.Int).Set(image),
		}
		for i, b := range preimage {
			inputs["preimage"].([]*big.Int)[i] = big.NewInt(int64(b))
		}
		return witCalc.CalculateWitness(inputs, true)
	}
	targetMD5 := ecc.Int10("246193259845151292174181299259247598493")
	witness, err := newWitness([]byte("RareSkills"), targetMD5)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	imageIdx := slices.IndexFunc(witness, func(e *big.Int) bool { return e.Cmp(targetMD5) == 0 })

	// Run all test cases.
	tests := []struct {
		preimage []byte
		ok       bool
	}{
		{preimage: []byte("RareSkills"), ok: true},
		{preimage: []byte("rareskills"), ok: false},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s", test.preimage), func(t *testing.T) {
			t.Parallel()

			imageB := md5.Sum(test.preimage)
			image := new(big.Int).SetBytes(imageB[:])
			witness, err := newWitness(test.preimage, image)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			witness[imageIdx].Set(targetMD5)

			sc := newRS2_10ProofSecret()
			proof, err := newRS2_10_V2Proof(witness, sc, r1cs, srs)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			if v := verifyRS2_10_V2(proof, srs); v != test.ok {
				t.Errorf("verify(%s) = %v want %v", test.preimage, v, test.ok)
			}
		})
	}
}

//go:embed testdata/rs3_11/rs3_11.r1cs.json
var rs3_11R1CSJson []byte

func TestRS3_11(t *testing.T) {
	// Circom program:
	// 	a * b === 1;
	// 	i <-- a * b;    // this is the bug, i can actually be anything.
	// 	out <== i * c;  // take advantage of this bug via out != c
	r1cs, err := circom.ParseR1CSJson(bytes.NewReader(rs3_11R1CSJson))
	if err != nil {
		t.Errorf("%+v", err)
	}
	k := field.NewPrimeExtDeg(r1cs.Prime, 1)
	qap := qapFromR1CS(k, r1cs)

	// Symbols are {1, out, a, b, c, i}, where out is output which in circom
	// is treated as public inputs, and a, b, c are public inputs.
	const numPublicInputs = 5
	tw := newToxicWasteRS2_10()
	srs := newStructuredRefStrRS2_10(tw, r1cs.NumConstraints, numPublicInputs, qap)

	// Create a witness where out != c:
	//                      {1, out, a, b, c, i}
	bogusWitnessI := []int64{1, 10, 1, 1, 5, 2}
	bogusWitness := make([]*big.Int, len(bogusWitnessI))
	for i, w := range bogusWitnessI {
		bogusWitness[i] = big.NewInt(w)
	}
	secret := newRS2_10ProofSecret()
	bogusProof := newRS2_10Proof(bogusWitness, secret, qap, srs)
	if !verifyRS2_10(bogusProof, srs) {
		t.Errorf("bogus proof should be verified due to bug in circuit")
	}
}

func TestRS2_10_V2(t *testing.T) {
	r1cs := newRS2_3_R1CS()

	tw := newToxicWasteRS2_10()
	srs := newStructuredRefStrRS2_10_V2(tw, r1cs.R1CS)

	wrongs := [][2]int{
		{1, 4},   // passes only constraint 0
		{1, 148}, // passes only constraint 1
	}
	const correctX, correctY = 5, 148
	for x := correctX - 3; x <= correctX+3; x++ {
		for y := correctY - 3; y <= correctY+3; y++ {
			if !(x == correctX && y == correctY) {
				wrongs = append(wrongs, [2]int{x, y})
			}
		}
	}
	for _, bad := range wrongs {
		witness := r1cs.ToWitness(bad[0], bad[1])
		secret := newRS2_10ProofSecret()
		proof, err := newRS2_10_V2Proof(witness, secret, r1cs.R1CS, srs)
		if err != nil {
			t.Errorf("%+v", err)
		}
		if verifyRS2_10_V2(proof, srs) {
			t.Errorf("wrong proofs should not be verified")
		}
	}

	witness := r1cs.ToWitness(correctX, correctY)
	secret := newRS2_10ProofSecret()
	proof, err := newRS2_10_V2Proof(witness, secret, r1cs.R1CS, srs)
	if err != nil {
		t.Errorf("%+v", err)
	}
	if !verifyRS2_10_V2(proof, srs) {
		t.Errorf("correct proofs should be verified")
	}
}

func TestRS2_10(t *testing.T) {
	// Define the circuit:
	//     x^3 + 5*x + -2 == y
	//     y^3 == 3241792
	r1cs := newRS2_3_R1CS()
	k := field.NewPrimeExtDeg(bn256.Order, 1)
	qap := qapFromR1CS(k, r1cs.R1CS)

	const numPublicInputs = 0
	tw := newToxicWasteRS2_10()
	srs := newStructuredRefStrRS2_10(tw, r1cs.NumConstraints, numPublicInputs, qap)

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
		witness := r1cs.ToWitness(bad[0], bad[1])
		secret := newRS2_10ProofSecret()
		proof := newRS2_10Proof(witness, secret, qap, srs)
		if verifyRS2_10(proof, srs) {
			t.Errorf("wrong proofs should not be verified")
		}
	}

	witness := r1cs.ToWitness(correctX, correctY)
	secret := newRS2_10ProofSecret()
	proof := newRS2_10Proof(witness, secret, qap, srs)
	if !verifyRS2_10(proof, srs) {
		t.Errorf("correct proofs should be verified")
	}
}

func TestRS2_9(t *testing.T) {
	// Define the circuit:
	//     x^3 + 5*x + -2 == y
	//     y^3 == 3241792
	r1cs := newRS2_3_R1CS()
	numConstraints := r1cs.NumConstraints
	k := field.NewPrimeExtDeg(bn256.Order, 1)
	qap := qapFromR1CS(k, r1cs.R1CS)

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
	srs := newStructuredRefStrRS2_9(tau, numConstraints, qap.T)

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
		proof := newRS2_9Proof(r1cs.ToWitness(bad[0], bad[1]), qap, srs)
		if verifyRS2_9(proof) {
			t.Errorf("wrong proofs should not be verified")
		}
	}

	witness := r1cs.ToWitness(correctX, correctY)
	proof := newRS2_9Proof(witness, qap, srs)
	if !verifyRS2_9(proof) {
		t.Errorf("correct proofs should be verified")
	}
}

func TestRS2_7(t *testing.T) {
	r1cs := newRS2_7_R1CS()
	k := field.NewPrimeExtDeg(big.NewInt(79), 1)
	qap := qapFromR1CS(k, r1cs.R1CS)

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
		witness := circuit.ToWitness(bad[0], bad[1])
		proof := newR1CSProof(witness)
		if verifyR1CS(circuit, proof) {
			t.Errorf("wrong proofs should not be verified")
		}
	}

	witness := circuit.ToWitness(correctX, correctY)
	solution := newR1CSProof(witness)
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

// tAt evaluates the t polynomial at tau.
// The t polynomial is t(x) = x^N - 1, where N is the smallest power of two >= numConstraints.
// Alternatively, t(x) = (x-ω^0)(x-ω^1)(x-ω^2)...(x-ω^(N-1)).
func tAt(numConstraints int, tau *big.Int) *big.Int {
	n := new(big.Int).SetUint64(fft.NewDomain(uint64(numConstraints)).Cardinality)
	tauN := new(big.Int).Exp(tau, n, bn254.ID.ScalarField())
	return tauN.Sub(tauN, big.NewInt(1))
}

// fitAt returns the interpolated polynomial at each column, evaluated at tau.
// The x coordinates of the interpolation points are the roots of unity.
// The algorithm used here is Fast-Fourier-Transform which is faster than Gaussian Elimination.
func fitAt(m circom.Matrix, tau *big.Int) []*big.Int {
	domain := fft.NewDomain(uint64(m.Rows))
	lagrange := lagrangeCoeffsAtTau(tau, domain)

	u := make([]*big.Int, m.Cols)
	for i := range u {
		u[i] = big.NewInt(0)
	}

	term := new(big.Int)
	for _, e := range m.COO {
		term.Mul(e.Val, lagrange[e.Row])
		u[e.Col].Add(u[e.Col], term)
	}
	return u
}

type toxicWasteRS2_10 struct {
	// tau is for verifying QAP polynomials match.
	tau *big.Int
	// alpha and beta are for the verifier to make sure the prover is not
	// providing trivial A, B, and Cs.
	alpha *big.Int
	beta  *big.Int
	// gamma and delta are for the verifier to make sure the public/private
	// separation provided by the prover is correct.
	gamma *big.Int
	delta *big.Int
}

func newToxicWasteRS2_10() toxicWasteRS2_10 {
	waste := toxicWasteRS2_10{}
	waste.tau, _ = rand.Int(rand.Reader, bn256.Order)
	waste.alpha, _ = rand.Int(rand.Reader, bn256.Order)
	waste.beta, _ = rand.Int(rand.Reader, bn256.Order)
	waste.gamma, _ = rand.Int(rand.Reader, bn256.Order)
	waste.delta, _ = rand.Int(rand.Reader, bn256.Order)
	return waste
}

type StructuredReferenceStringsRS2_10_V2 struct {
	Alpha1 *bn254.G1Affine
	Beta1  *bn254.G1Affine
	Beta2  *bn254.G2Affine
	Gamma2 *bn254.G2Affine
	Delta1 *bn254.G1Affine
	Delta2 *bn254.G2Affine

	HT              []*bn254.G1Affine
	NumPublicInputs int
	Psi             []*bn254.G1Affine

	U1 []*bn254.G1Affine
	V1 []*bn254.G1Affine
	V2 []*bn254.G2Affine
}

func newStructuredRefStrRS2_10_V2(tw toxicWasteRS2_10, r1cs circom.R1CS) StructuredReferenceStringsRS2_10_V2 {
	u := fitAt(r1cs.L, tw.tau)
	v := fitAt(r1cs.R, tw.tau)
	w := fitAt(r1cs.O, tw.tau)
	t := tAt(r1cs.NumConstraints, tw.tau)

	// tDeg is the degree of the vanishing polynomial t = x^n-1.
	// It is also the roots-of-unity domain's cardinality, since
	// t(x) = (x-ω^0)(x-ω^1)(x-ω^2)...(x-ω^(N-1)).
	tDeg := fft.NewDomain(uint64(r1cs.NumConstraints)).Cardinality
	numPublicInputs := 1 + r1cs.NumOutputs + r1cs.NumPublicInputs
	srs := StructuredReferenceStringsRS2_10_V2{
		Alpha1: new(bn254.G1Affine).ScalarMultiplicationBase(tw.alpha),
		Beta1:  new(bn254.G1Affine).ScalarMultiplicationBase(tw.beta),
		Beta2:  new(bn254.G2Affine).ScalarMultiplicationBase(tw.beta),
		Gamma2: new(bn254.G2Affine).ScalarMultiplicationBase(tw.gamma),
		Delta1: new(bn254.G1Affine).ScalarMultiplicationBase(tw.delta),
		Delta2: new(bn254.G2Affine).ScalarMultiplicationBase(tw.delta),

		// h = (u*v-w)/t, so h can have up to tDeg-1 nonzero coefficients (degree tDeg-2).
		HT:              make([]*bn254.G1Affine, tDeg),
		NumPublicInputs: numPublicInputs,
		Psi:             make([]*bn254.G1Affine, r1cs.NumVars),

		U1: make([]*bn254.G1Affine, r1cs.NumVars),
		V1: make([]*bn254.G1Affine, r1cs.NumVars),
		V2: make([]*bn254.G2Affine, r1cs.NumVars),
	}

	// Compute HT.
	deltaInv := new(big.Int).ModInverse(tw.delta, bn256.Order)
	htWork := func(chunk [2]int, tmp *big.Int) {
		// Initialize tmp to (tau^n) * t / delta.
		tau := field.NewPrimeExtDeg(r1cs.Prime, 1).SetCoeffs(tw.tau)
		tauPowI := nag.Pow(tau, big.NewInt(int64(chunk[0]))).Coeffs()[0]
		tmp.Mul(t, deltaInv)
		tmp.Mul(tmp, tauPowI)
		for i := chunk[0]; i < chunk[1]; i++ {
			srs.HT[i] = new(bn254.G1Affine).ScalarMultiplicationBase(tmp)
			tmp.Mul(tmp, tw.tau)
			tmp.Mod(tmp, bn256.Order)
		}
	}
	htConcurrency := runtime.NumCPU()
	htChunkLen := max(len(srs.HT)/htConcurrency, 1)
	htChunks := make(chan [2]int)
	go func() {
		defer close(htChunks)
		for i := 0; i < len(srs.HT); i += htChunkLen {
			c := [2]int{i, min(i+htChunkLen, len(srs.HT))}
			htChunks <- c
		}
	}()
	var wgHT sync.WaitGroup
	for range htConcurrency {
		wgHT.Go(func() {
			for c := range htChunks {
				htWork(c, new(big.Int))
			}
		})
	}

	// Compute Psi, U1, V1, V2.
	concurrency := runtime.NumCPU()
	chunkLen := max(r1cs.NumVars/concurrency, 1)
	chunks := make(chan [2]int)
	go func() {
		defer close(chunks)
		for i := 0; i < r1cs.NumVars; i += chunkLen {
			c := [2]int{i, min(i+chunkLen, r1cs.NumVars)}
			chunks <- c
		}
	}()
	gammaInv := new(big.Int).ModInverse(tw.gamma, bn256.Order)
	work := func(chunk [2]int, tmp0, tmp1 *big.Int) {
		for i := chunk[0]; i < chunk[1]; i++ {
			tmp0.Set(w[i])
			tmp0.Add(tmp0, tmp1.Mul(tw.alpha, v[i]))
			tmp0.Add(tmp0, tmp1.Mul(tw.beta, u[i]))
			if i < numPublicInputs {
				tmp0.Mul(tmp0, gammaInv)
			} else {
				tmp0.Mul(tmp0, deltaInv)
			}
			srs.Psi[i] = new(bn254.G1Affine).ScalarMultiplicationBase(tmp0)

			srs.U1[i] = new(bn254.G1Affine).ScalarMultiplicationBase(u[i])
			srs.V1[i] = new(bn254.G1Affine).ScalarMultiplicationBase(v[i])
			srs.V2[i] = new(bn254.G2Affine).ScalarMultiplicationBase(v[i])
		}
	}
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			tmp0, tmp1 := new(big.Int), new(big.Int)
			for c := range chunks {
				work(c, tmp0, tmp1)
			}
		})
	}

	wgHT.Wait()
	wg.Wait()

	return srs
}

func g2Dot(is []*big.Int, g2 []*bn254.G2Affine) *bn254.G2Affine {
	concurrency := runtime.NumCPU()
	chunkLen := max(len(is)/concurrency, 1)
	chunks := make(chan [2]int)
	go func() {
		defer close(chunks)
		for i := 0; i < len(is); i += chunkLen {
			c := [2]int{i, min(i+chunkLen, len(is))}
			chunks <- c
		}
	}()

	// Fan out.
	work := func(chunk [2]int) *bn254.G2Affine {
		e := new(bn254.G2Affine).SetInfinity()
		for i := chunk[0]; i < chunk[1]; i++ {
			e.Add(e, new(bn254.G2Affine).ScalarMultiplication(g2[i], is[i]))
		}
		return e
	}
	outChan := make(chan *bn254.G2Affine)
	go func() {
		defer close(outChan)
		var wg sync.WaitGroup
		for range concurrency {
			wg.Go(func() {
				for c := range chunks {
					outChan <- work(c)
				}
			})
		}
		wg.Wait()
	}()

	// Fan in.
	sum := new(bn254.G2Affine).SetInfinity()
	for o := range outChan {
		sum.Add(sum, o)
	}

	return sum
}

func calcHT(witness []*big.Int, r1cs circom.R1CS, srs StructuredReferenceStringsRS2_10_V2) *bn254.G1Affine {
	a := mulWitness(witness, r1cs.L)
	b := mulWitness(witness, r1cs.R)
	c := mulWitness(witness, r1cs.O)
	domain := fft.NewDomain(uint64(r1cs.NumConstraints))
	hEvals := groth16.ComputeH(a, b, c, domain)
	h := frToBigInts(hEvals)

	concurrency := runtime.NumCPU()
	chunkLen := max(len(h)/concurrency, 1)
	chunks := make(chan [2]int)
	go func() {
		defer close(chunks)
		for i := 0; i < len(h); i += chunkLen {
			c := [2]int{i, min(i+chunkLen, len(h))}
			chunks <- c
		}
	}()

	// Fan out.
	work := func(chunk [2]int) *bn254.G1Affine {
		ht := new(bn254.G1Affine).SetInfinity()
		for i := chunk[0]; i < chunk[1]; i++ {
			ht.Add(ht, new(bn254.G1Affine).ScalarMultiplication(srs.HT[i], h[i]))
		}
		return ht
	}
	htChan := make(chan *bn254.G1Affine)
	go func() {
		defer close(htChan)
		var wg sync.WaitGroup
		for range concurrency {
			wg.Go(func() {
				for c := range chunks {
					htChan <- work(c)
				}
			})
		}
		wg.Wait()
	}()

	// Fan in.
	ht := new(bn254.G1Affine).SetInfinity()
	for hti := range htChan {
		ht.Add(ht, hti)
	}

	return ht
}

type rs2_10_V2Proof struct {
	A      *bn254.G1Affine
	B      *bn254.G2Affine
	C      *bn254.G1Affine
	Public []*big.Int
}

func newRS2_10_V2Proof(witness []*big.Int, secret rs2_10ProofSecret, r1cs circom.R1CS, srs StructuredReferenceStringsRS2_10_V2) (rs2_10_V2Proof, error) {
	if len(witness) != r1cs.NumVars {
		return rs2_10_V2Proof{}, errors.Errorf("wrong witness length %d want %d", len(witness), r1cs.NumVars)
	}
	proof := rs2_10_V2Proof{
		A:      new(bn254.G1Affine).SetInfinity(),
		B:      new(bn254.G2Affine).SetInfinity(),
		C:      new(bn254.G1Affine).SetInfinity(),
		Public: make([]*big.Int, srs.NumPublicInputs),
	}

	// Compute proof.A.
	aDone := make(chan struct{})
	go func() {
		defer close(aDone)
		proof.A.Set(srs.Alpha1)
		for i, w := range witness {
			proof.A.Add(proof.A, new(bn254.G1Affine).ScalarMultiplication(srs.U1[i], w))
		}
		rDelta := new(bn254.G1Affine).ScalarMultiplication(srs.Delta1, secret.r)
		proof.A.Add(proof.A, rDelta)
	}()

	// Compute proof.B.
	bDone := make(chan struct{})
	go func() {
		defer close(bDone)
		proof.B.Set(srs.Beta2)
		proof.B.Add(proof.B, g2Dot(witness, srs.V2))
		sDelta2 := new(bn254.G2Affine).ScalarMultiplication(srs.Delta2, secret.s)
		proof.B.Add(proof.B, sDelta2)
	}()

	// Compute b1.
	b1 := new(bn254.G1Affine).Set(srs.Beta1)
	b1Done := make(chan struct{})
	go func() {
		defer close(b1Done)
		for i, w := range witness {
			b1.Add(b1, new(bn254.G1Affine).ScalarMultiplication(srs.V1[i], w))
		}
		sDelta1 := new(bn254.G1Affine).ScalarMultiplication(srs.Delta1, secret.s)
		b1.Add(b1, sDelta1)
	}()

	// Compute witness*Psi.
	wpsi := new(bn254.G1Affine).SetInfinity()
	psiDone := make(chan struct{})
	go func() {
		defer close(psiDone)
		for i := srs.NumPublicInputs; i < len(srs.Psi); i++ {
			aPsi := new(bn254.G1Affine).ScalarMultiplication(srs.Psi[i], witness[i])
			wpsi.Add(wpsi, aPsi)
		}
	}()

	// Compute h*t.
	ht := calcHT(witness, r1cs, srs)

	<-aDone
	<-b1Done
	<-psiDone

	proof.C.Add(proof.C, wpsi)
	proof.C.Add(proof.C, ht)
	// Compute A*s.
	as := new(bn254.G1Affine).ScalarMultiplication(proof.A, secret.s)
	proof.C.Add(proof.C, as)
	// Compute B*r.
	br := new(bn254.G1Affine).ScalarMultiplication(b1, secret.r)
	proof.C.Add(proof.C, br)
	// Compute -r*s*delta.
	negRS := new(big.Int).Mul(secret.r, secret.s)
	negRS.Neg(negRS)
	negRSDelta := new(bn254.G1Affine).ScalarMultiplication(srs.Delta1, negRS)
	proof.C.Add(proof.C, negRSDelta)

	for i := range proof.Public {
		proof.Public[i] = new(big.Int).Set(witness[i])
	}

	<-bDone
	return proof, nil
}

func verifyRS2_10_V2(proof rs2_10_V2Proof, srs StructuredReferenceStringsRS2_10_V2) bool {
	// Constant term should always be 1, otherwise prover could present
	// bogus proofs that do not respect constraints with constants.
	if proof.Public[0].Cmp(big.NewInt(1)) != 0 {
		return false
	}
	// Compute x = witness * Psi.
	x := new(bn254.G1Affine).SetInfinity()
	for i := range proof.Public {
		aPsi := new(bn254.G1Affine).ScalarMultiplication(srs.Psi[i], proof.Public[i])
		x.Add(x, aPsi)
	}

	// Left hand side.
	lhs, err := bn254.Pair([]bn254.G1Affine{*proof.A}, []bn254.G2Affine{*proof.B})
	if err != nil {
		return false
	}
	// Right hand side.
	rhs, err := bn254.Pair(
		[]bn254.G1Affine{*srs.Alpha1, *x, *proof.C},
		[]bn254.G2Affine{*srs.Beta2, *srs.Gamma2, *srs.Delta2},
	)
	if err != nil {
		return false
	}
	// Left hand side should equal right hand side.
	return lhs.Equal(&rhs)
}

type StructuredReferenceStringsRS2_10 struct {
	Alpha1 *bn256.G1
	Beta1  *bn256.G1
	Beta2  *bn256.G2
	Gamma2 *bn256.G2
	Delta1 *bn256.G1
	Delta2 *bn256.G2

	G1              []*bn256.G1
	G2              []*bn256.G2
	HT              []*bn256.G1
	NumPublicInputs int
	Psi             []*bn256.G1
}

func newStructuredRefStrRS2_10[K field.Finite[K]](tw toxicWasteRS2_10, numConstraints, numPublicInputs int, qap QAP[K]) StructuredReferenceStringsRS2_10 {
	numVars := len(qap.U)
	srs := StructuredReferenceStringsRS2_10{
		Alpha1: new(bn256.G1).ScalarBaseMult(tw.alpha),
		Beta1:  new(bn256.G1).ScalarBaseMult(tw.beta),
		Beta2:  new(bn256.G2).ScalarBaseMult(tw.beta),
		Gamma2: new(bn256.G2).ScalarBaseMult(tw.gamma),
		Delta1: new(bn256.G1).ScalarBaseMult(tw.delta),
		Delta2: new(bn256.G2).ScalarBaseMult(tw.delta),

		G1:              make([]*bn256.G1, numConstraints),
		G2:              make([]*bn256.G2, numConstraints),
		HT:              make([]*bn256.G1, numConstraints),
		NumPublicInputs: numPublicInputs,
		Psi:             make([]*bn256.G1, numVars),
	}

	// Allocate memory.
	iBig, tauI := new(big.Int), new(big.Int)

	// Compute t(tau).
	tauK := qap.T.Field().NewZero().SetCoeffs(tw.tau)
	tTau := ecc.EvalPoly(qap.T, []K{tauK}).Coeffs()[0]

	// Compute G1, G2, and HT.
	deltaInv := new(big.Int).ModInverse(tw.delta, bn256.Order)
	for i := range numConstraints {
		tauI.Exp(tw.tau, iBig.SetInt64(int64(i)), nil)
		srs.G1[i] = new(bn256.G1).ScalarBaseMult(tauI)
		srs.G2[i] = new(bn256.G2).ScalarBaseMult(tauI)

		tauI.Mul(tauI, tTau)
		tauI.Mul(tauI, deltaInv)
		tauI.Mod(tauI, bn256.Order)
		srs.HT[i] = new(bn256.G1).ScalarBaseMult(tauI)
	}

	// Compute Psi.
	gammaInv := new(big.Int).ModInverse(tw.gamma, bn256.Order)
	for i := range srs.Psi {
		vTau := ecc.EvalPoly(qap.V[i], []K{tauK}).Coeffs()[0]
		uTau := ecc.EvalPoly(qap.U[i], []K{tauK}).Coeffs()[0]
		wTau := ecc.EvalPoly(qap.W[i], []K{tauK}).Coeffs()[0]
		s := tauI.SetInt64(0)
		s.Add(s, iBig.Mul(tw.alpha, vTau))
		s.Add(s, iBig.Mul(tw.beta, uTau))
		s.Add(s, wTau)

		if i < numPublicInputs {
			s.Mul(s, gammaInv)
		} else {
			s.Mul(s, deltaInv)
		}
		s.Mod(s, bn256.Order)

		srs.Psi[i] = new(bn256.G1).ScalarBaseMult(s)
	}

	return srs
}

type rs2_10Proof struct {
	A      *bn256.G1
	B      *bn256.G2
	C      *bn256.G1
	Public []*big.Int
}

type rs2_10ProofSecret struct {
	// r and s are to make sure no one can guess our witness from the
	// published encrypted proof.
	r *big.Int
	s *big.Int
}

func newRS2_10ProofSecret() rs2_10ProofSecret {
	key := rs2_10ProofSecret{}
	key.r, _ = rand.Int(rand.Reader, bn256.Order)
	key.s, _ = rand.Int(rand.Reader, bn256.Order)
	return key
}

func newRS2_10Proof[K field.Finite[K]](witness []*big.Int, secret rs2_10ProofSecret, qap QAP[K], srs StructuredReferenceStringsRS2_10) rs2_10Proof {
	u, v, _, h := mulWitnessQAP(witness, qap)

	proof := rs2_10Proof{
		A:      new(bn256.G1).ScalarBaseMult(bn256.Order),
		B:      new(bn256.G2).ScalarBaseMult(bn256.Order),
		C:      new(bn256.G1).ScalarBaseMult(bn256.Order),
		Public: make([]*big.Int, srs.NumPublicInputs),
	}

	// Compute proof.A.
	proof.A.Set(srs.Alpha1)
	g1PolyProduct(proof.A, u, srs.G1)
	rDelta := new(bn256.G1).ScalarMult(srs.Delta1, secret.r)
	proof.A.Add(proof.A, rDelta)

	// Compute proof.B.
	proof.B.Set(srs.Beta2)
	g2PolyProduct(proof.B, v, srs.G2)
	sDelta2 := new(bn256.G2).ScalarMult(srs.Delta2, secret.s)
	proof.B.Add(proof.B, sDelta2)

	// Compute proof.C.
	// Begin by computing B in G1.
	b1 := new(bn256.G1).Set(srs.Beta1)
	g1PolyProduct(b1, v, srs.G1)
	sDelta1 := new(bn256.G1).ScalarMult(srs.Delta1, secret.s)
	b1.Add(b1, sDelta1)
	// Compute witness*Psi.
	for i := srs.NumPublicInputs; i < len(srs.Psi); i++ {
		aPsi := new(bn256.G1).ScalarMult(srs.Psi[i], witness[i])
		proof.C.Add(proof.C, aPsi)
	}
	// Compute h*t.
	g1PolyProduct(proof.C, h, srs.HT)
	// Compute A*s.
	as := new(bn256.G1).ScalarMult(proof.A, secret.s)
	proof.C.Add(proof.C, as)
	// Compute B*r.
	br := new(bn256.G1).ScalarMult(b1, secret.r)
	proof.C.Add(proof.C, br)
	// Compute -r*s*delta.
	negRS := new(big.Int).Mul(secret.r, secret.s)
	negRS.Neg(negRS)
	negRSDelta := new(bn256.G1).ScalarMult(srs.Delta1, negRS)
	proof.C.Add(proof.C, negRSDelta)

	for i := range proof.Public {
		proof.Public[i] = new(big.Int).Set(witness[i])
	}

	return proof
}

func verifyRS2_10(proof rs2_10Proof, srs StructuredReferenceStringsRS2_10) bool {
	x := new(bn256.G1).ScalarBaseMult(bn256.Order)
	for i := range proof.Public {
		aPsi := new(bn256.G1).ScalarMult(srs.Psi[i], proof.Public[i])
		x.Add(x, aPsi)
	}

	lhs := bn256.Pair(proof.A, proof.B)

	// Compute right hand side.
	//
	// Alpha and Beta prevents the prover from faking trivial solutions to
	// the equation A_1 * B_2 = C_1 * G_2.
	// This is because now the equation is A_1 * B_2 = (αβ)_T + C_1 * G_2.
	// To provide fake solutions, the malicious prover need to solve for C_1
	// in the equation χ_12 = C_1 * G_2, where χ_12 = A_1 * B_2 - (αβ)_T.
	// Since χ_12 is an arbitrary number determined by the trusted setup,
	// solving for C_1 is infeasible assuming the Diffie–Hellman assumption
	// holds.
	//
	// The only way out for the prover is to utilize the fact that α and β
	// are baked into the structured reference string. Use this fact, along
	// with the correct solution to build C that satisfies A*B = αβ + C.
	rhs := bn256.Pair(srs.Alpha1, srs.Beta2)
	rhs.Add(rhs, bn256.Pair(x, srs.Gamma2))
	rhs.Add(rhs, bn256.Pair(proof.C, srs.Delta2))

	return bytes.Equal(lhs.Marshal(), rhs.Marshal())
}

type StructuredReferenceStringsRS2_9 struct {
	Omega []*bn256.G1
	Theta []*bn256.G2
	Y     []*bn256.G1
}

func newStructuredRefStrRS2_9[K field.Finite[K]](tau *big.Int, numConstraints int, tPoly *nag.Polynomial[K]) StructuredReferenceStringsRS2_9 {
	iBig, tauI := new(big.Int), new(big.Int)

	tauK := tPoly.Field().NewZero().SetCoeffs(tau)
	tTau := ecc.EvalPoly(tPoly, []K{tauK}).Coeffs()[0]

	srs := StructuredReferenceStringsRS2_9{
		Omega: make([]*bn256.G1, numConstraints),
		Theta: make([]*bn256.G2, numConstraints),
		Y:     make([]*bn256.G1, numConstraints),
	}
	for i := range numConstraints {
		tauI.Exp(tau, iBig.SetInt64(int64(i)), nil)
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

// groupByRow buckets a sparse COO matrix's entries by row.
func groupByRow(m circom.Matrix) map[int][]circom.CooEntry {
	rows := make(map[int][]circom.CooEntry, m.Rows)
	for _, e := range m.COO {
		rows[e.Row] = append(rows[e.Row], e)
	}
	return rows
}

// groupByCol buckets a sparse COO matrix's entries by column.
func groupByCol(m circom.Matrix) map[int][]circom.CooEntry {
	cols := make(map[int][]circom.CooEntry, m.Cols)
	for _, e := range m.COO {
		cols[e.Col] = append(cols[e.Col], e)
	}
	return cols
}

func qapFromR1CS[K field.Finite[K]](k K, r1cs circom.R1CS) QAP[K] {
	numVars := r1cs.NumVars
	qap := QAP[K]{
		U: make([]*nag.Polynomial[K], numVars),
		V: make([]*nag.Polynomial[K], numVars),
		W: make([]*nag.Polynomial[K], numVars),
	}

	// Compute U, V, and W.
	numConstraints := r1cs.NumConstraints
	points := make([][]K, numConstraints)
	for i := range points {
		points[i] = []K{k.NewZero(), k.NewZero()}
		points[i][0].SetCoeffs(big.NewInt(int64(i + 1)))
	}
	polys := [3][]*nag.Polynomial[K]{qap.U, qap.V, qap.W}
	mats := [3]circom.Matrix{r1cs.L, r1cs.R, r1cs.O}
	for pi, poly := range polys {
		cols := groupByCol(mats[pi])

		for j := range numVars {
			for i := range points {
				points[i][1].SetCoeffs(big.NewInt(0))
			}
			for _, e := range cols[j] {
				points[e.Row][1].SetCoeffs(e.Val)
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

func mulWitnessQAP[K field.Finite[K]](witnessI []*big.Int, qap QAP[K]) (*nag.Polynomial[K], *nag.Polynomial[K], *nag.Polynomial[K], *nag.Polynomial[K]) {
	k := qap.U[0].Field()
	order := qap.U[0].Order()

	witness := make([]K, len(witnessI))
	for i, w := range witnessI {
		witness[i] = k.NewZero().SetCoeffs(w)
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

func g1PolyProduct[K field.Finite[K]](sum *bn256.G1, poly *nag.Polynomial[K], powers []*bn256.G1) {
	e := new(bn256.G1)
	for cF, m := range poly.Terms() {
		c := cF.Coeffs()[0]
		deg := len(m)

		e.Set(powers[deg])
		e.ScalarMult(e, c)
		sum.Add(sum, e)
	}
}

func g2PolyProduct[K field.Finite[K]](sum *bn256.G2, poly *nag.Polynomial[K], powers []*bn256.G2) {
	e := new(bn256.G2)
	for cF, m := range poly.Terms() {
		c := cF.Coeffs()[0]
		deg := len(m)

		e.Set(powers[deg])
		e.ScalarMult(e, c)
		sum.Add(sum, e)
	}
}

type rs2_9Proof struct {
	A *bn256.G1
	B *bn256.G2
	C *bn256.G1
}

func newRS2_9Proof[K field.Finite[K]](witness []*big.Int, qap QAP[K], srs StructuredReferenceStringsRS2_9) rs2_9Proof {
	u, v, w, h := mulWitnessQAP(witness, qap)

	proof := rs2_9Proof{
		A: new(bn256.G1).ScalarBaseMult(bn256.Order),
		B: new(bn256.G2).ScalarBaseMult(bn256.Order),
		C: new(bn256.G1).ScalarBaseMult(bn256.Order),
	}

	g1PolyProduct(proof.A, u, srs.Omega)
	g2PolyProduct(proof.B, v, srs.Theta)
	g1PolyProduct(proof.C, w, srs.Omega)
	g1PolyProduct(proof.C, h, srs.Y)

	return proof
}

func verifyRS2_9(proof rs2_9Proof) bool {
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

func newR1CSProof(witness []*big.Int) r1csProof {
	proof := r1csProof{
		G1: make([]*bn256.G1, len(witness)),
		G2: make([]*bn256.G2, len(witness)),
	}
	for i, w := range witness {
		proof.G1[i] = new(bn256.G1).ScalarBaseMult(w)
		proof.G2[i] = new(bn256.G2).ScalarBaseMult(w)
	}
	return proof
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
	numVars := circuit.NumVars
	for j := range numVars {
		gT1 := bn256.Pair(proof.G1[j], g2)
		gT2 := bn256.Pair(g1, proof.G2[j])
		gT1.Add(gT1, gT2.Neg(gT2))

		if !bytes.Equal(gT1.Marshal(), gTOne) {
			return false
		}
	}

	// Check that all constraints are satisfied.
	numConstraints := circuit.NumConstraints
	lRows := groupByRow(circuit.L)
	rRows := groupByRow(circuit.R)
	oRows := groupByRow(circuit.O)
	for i := range numConstraints {
		// Add circuit L and R.
		g1Sum.ScalarBaseMult(bn256.Order)
		g2Sum.ScalarBaseMult(bn256.Order)
		for _, e := range lRows[i] {
			bi.Set(e.Val)
			g1.Set(proof.G1[e.Col]).ScalarMult(g1, bi)
			g1Sum.Add(g1Sum, g1)
		}
		for _, e := range rRows[i] {
			bi.Set(e.Val)
			g2.Set(proof.G2[e.Col]).ScalarMult(g2, bi)
			g2Sum.Add(g2Sum, g2)
		}
		gT := bn256.Pair(g1Sum, g2Sum)

		// Add circuit O.
		g1Sum.ScalarBaseMult(bn256.Order)
		for _, e := range oRows[i] {
			bi.Set(e.Val)
			g1.Set(proof.G1[e.Col]).ScalarMult(g1, bi)
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

type r1CS struct {
	ToWitness func(xs ...int) []*big.Int
	circom.R1CS
}

//go:embed circom/testdata/rs2_3/rs2_3.r1cs.json
var rs23R1CSJson []byte

func newRS2_3_R1CS() r1CS {
	r1cs := r1CS{}
	var err error
	r1cs.R1CS, err = circom.ParseR1CSJson(bytes.NewReader(rs23R1CSJson))
	if err != nil {
		panic(err)
	}

	const one, x, y, x3, x2, y2 = 0, 1, 2, 3, 4, 5
	const numVars = 6
	r1cs.ToWitness = func(vs ...int) []*big.Int {
		wx, wy := int64(vs[0]), int64(vs[1])
		witness := make([]*big.Int, numVars)
		witness[one] = big.NewInt(1)
		witness[x] = big.NewInt(wx)
		witness[x2] = big.NewInt(wx * wx)
		witness[x3] = big.NewInt(wx * wx * wx)
		witness[y] = big.NewInt(wy)
		witness[y2] = big.NewInt(wy * wy)
		return witness
	}
	return r1cs
}

func newRS2_7_R1CS() r1CS {
	// Circuit equation:
	//     z = x^4 - 5y^2x^2
	// It is based on Chapter 7, Module 2 of the RareSkills Zero-Knowledge book.
	// https://rareskills.io/post/r1cs-to-qap
	circuit := r1CS{}
	const one, z, x, y, v1, v2, v3 = 0, 1, 2, 3, 4, 5, 6
	const numVars = 7
	circuit.ToWitness = func(vs ...int) []*big.Int {
		wx, wy, wz := int64(vs[0]), int64(vs[1]), int64(vs[2])
		witness := make([]*big.Int, numVars)
		witness[one] = big.NewInt(1)
		witness[z] = big.NewInt(wz)
		witness[x] = big.NewInt(wx)
		witness[y] = big.NewInt(wy)
		witness[v1] = big.NewInt(wx * wx)
		witness[v2] = new(big.Int).Mul(witness[v1], witness[v1])
		witness[v3] = big.NewInt(-5 * wy * wy)
		return witness
	}

	const numConstraints = 4
	circuit.NumVars = numVars
	circuit.NumConstraints = numConstraints
	circuit.L = circom.Matrix{Rows: numConstraints, Cols: numVars, COO: []circom.CooEntry{
		{Row: 0, Col: x, Val: big.NewInt(1)},  // Constraint 0.
		{Row: 1, Col: v1, Val: big.NewInt(1)}, // Constraint 1.
		{Row: 2, Col: y, Val: big.NewInt(-5)}, // Constraint 2.
		{Row: 3, Col: v3, Val: big.NewInt(1)}, // Constraint 4.
	}}
	circuit.R = circom.Matrix{Rows: numConstraints, Cols: numVars, COO: []circom.CooEntry{
		{Row: 0, Col: x, Val: big.NewInt(1)},
		{Row: 1, Col: v1, Val: big.NewInt(1)},
		{Row: 2, Col: y, Val: big.NewInt(1)},
		{Row: 3, Col: v1, Val: big.NewInt(1)},
	}}
	circuit.O = circom.Matrix{Rows: numConstraints, Cols: numVars, COO: []circom.CooEntry{
		{Row: 0, Col: v1, Val: big.NewInt(1)},
		{Row: 1, Col: v2, Val: big.NewInt(1)},
		{Row: 2, Col: v3, Val: big.NewInt(1)},
		{Row: 3, Col: z, Val: big.NewInt(1)},
		{Row: 3, Col: v2, Val: big.NewInt(-1)},
	}}

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
