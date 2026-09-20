package zk

import (
	"bytes"
	"cmp"
	"crypto/md5"
	"crypto/rand"
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
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
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
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

func TestRS4_8(t *testing.T) {
	tests := []struct {
		a []int
		b []int
	}{
		{
			a: []int{808, 140, 166, 209},
			b: []int{88, 242, 404, 602},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			a, b := bigs(test.a...), bigs(test.b...)
			basis := newBulletProofBasis(len(a))
			proverSecret := newBulletProverSecret(basis)
			commitment := bulletProofCommit(a, b, basis, proverSecret)

			proof := bulletProve(a, b, commitment, basis, proverSecret)
			if !bulletProofVerify(proof, commitment, basis) {
				t.Errorf("should verify")
			}

			badA := make([]*big.Int, len(a))
			for i := range a {
				badA[i] = new(big.Int).Set(a[i])
			}
			badA[0].SetInt64(98765)
			badProof := bulletProve(badA, b, commitment, basis, proverSecret)
			if bulletProofVerify(badProof, commitment, basis) {
				t.Errorf("should not verify")
			}
		})
	}
}

func TestRS4_8_FrozenHeart(t *testing.T) {
	a := bigs([]int{808, 140, 166, 209}...)
	basis := newBulletProofBasis(len(a))
	mod := bn254.ID.ScalarField()

	inf := new(bn254.G1Affine).SetInfinity()

	proofTu := big.NewInt(555)
	negTu := new(big.Int).Neg(proofTu)
	negTu.Mod(negTu, mod)
	proofC := new(bn254.G1Affine).ScalarMultiplication(basis.q, negTu)
	fakeProof := bulletProof{
		c:    proofC,
		tu:   proofTu,
		pilr: big.NewInt(1337),
		pit:  big.NewInt(777),
		ipp: InnerProductProof{
			L:     []*bn254.G1Affine{inf, inf},
			R:     []*bn254.G1Affine{inf, inf},
			lastA: big.NewInt(0),
			lastB: big.NewInt(0),
		},
	}

	// cm.a, cm.v solved to match proof.c, proof.tu (no u term to solve
	// backward against, since cm.s = cm.t1 = cm.t2 = infinity).
	tmp := new(bn254.G1Affine)
	cmA := new(bn254.G1Affine).Set(fakeProof.c)
	cmA.Add(cmA, tmp.ScalarMultiplication(basis.b, fakeProof.pilr))
	tuq := new(bn254.G1Affine).ScalarMultiplication(basis.q, fakeProof.tu)
	cmV := new(bn254.G1Affine).Set(tuq)
	cmV.Add(cmV, tmp.ScalarMultiplication(basis.b, fakeProof.pit))
	fakeCommitment := bulletProofCommitment{
		n:  len(a),
		a:  cmA,
		s:  inf,
		v:  cmV,
		t1: inf,
		t2: inf,
	}

	if !bulletProofVerifyFrozenHeart(fakeProof, fakeCommitment, basis) {
		t.Errorf("should verify")
	}
	if bulletProofVerify(fakeProof, fakeCommitment, basis) {
		t.Errorf("should not verify")
	}
}

// TestRS4_7_Protocol1 demonstrates the necessity of using u^x instead of u in
// Protocol 1 of [Bulletproofs], Bunz et. al.
//
// In our implementation, this means verifyCommitmentsLog checks only for
// relation (3) in the Bulletproofs paper, but not relation (2).
//
// Relation (2) is the more stringent claim that
//
//	(P = <a, g> + <b, h>) && (c = <a, b>)
//
// Relation (3) is the more lax claim that
//
//	P = <a, g> + <b, h> + <a, b>*u
//
// Notice that while relation (2) concerns two variables P and c, relation (3)
// is concerned only with one variable P. In other words, whether or not some
// variable c equals <a, b> is totally out of scope of relation (3)'s abilities.
// This matches the arguments received by verifyCommitmentsLog, which includes
// only the commitment P, but nothing related to the value of <a, b>.
//
// This test demonstrates an example (C, v) that satisfies
//
//	C + vQ = <a, G> + <b, H> + <a, b>*Q
//
// yet
//
//	(C != <a, G> + <b, H>) nor (v != <a, b>)
//
// [Bulletproofs]: https://eprint.iacr.org/2017/1066
func TestRS4_7_Protocol1(t *testing.T) {
	mod := bn254.ID.ScalarField()
	commit := func(agbh *bn254.G1Affine, ab *big.Int, basis bulletProofBasis) *bn254.G1Affine {
		p := new(bn254.G1Affine).ScalarMultiplication(basis.q, ab)
		return p.Add(p, agbh)
	}

	basis := rs4_7_Case.basis
	a := rs4_7_Case.a
	b := rs4_7_Case.b

	// Compute <a, b>.
	ab, tmpi := big.NewInt(0), new(big.Int)
	for i := range a {
		ab.Add(ab, tmpi.Mul(a[i], b[i]))
	}
	ab.Mod(ab, mod)

	// Create bogus v where v != <a, b>.
	v := big.NewInt(999)

	// Create c where c != <a, G> + <b, h>.
	tmp := new(bn254.G1Affine)
	agbh := new(bn254.G1Affine).SetInfinity()
	for i := range a {
		agbh.Add(agbh, tmp.ScalarMultiplication(basis.g[i], a[i]))
	}
	for i := range b {
		agbh.Add(agbh, tmp.ScalarMultiplication(basis.h[i], b[i]))
	}
	c := new(bn254.G1Affine).Set(agbh)
	c.Add(c, tmp.ScalarMultiplication(basis.q, new(big.Int).Sub(ab, v)))

	// Show that verifyCommitmentsLog accepts fakeCommitment = c + v*q.
	// This happens because fakeCommitment itself is numerically correct,
	// even though both c and v are wrong.
	fakeCommitment := commit(c, v, basis)
	proof, _ := proveCommitmentsLog(basis.g, basis.h, basis.q, a, b, nil)
	if !verifyCommitmentsLog(proof, len(a), fakeCommitment, basis.g, basis.h, basis.q, nil) {
		t.Errorf("should verify")
	}
}

func TestRS4_7(t *testing.T) {
	commit := func(a, b []*big.Int, basis bulletProofBasis) *bn254.G1Affine {
		v, tmpi := big.NewInt(0), new(big.Int)
		for i := range a {
			v.Add(v, tmpi.Mul(a[i], b[i]))
			v.Mod(v, bn254.ID.ScalarField())
		}

		tmp := new(bn254.G1Affine)
		cm := new(bn254.G1Affine).SetInfinity()
		cm.Add(cm, tmp.ScalarMultiplication(basis.q, v))
		for i := range a {
			cm.Add(cm, tmp.ScalarMultiplication(basis.g[i], a[i]))
		}
		for i := range b {
			cm.Add(cm, tmp.ScalarMultiplication(basis.h[i], b[i]))
		}
		return cm
	}

	basis := rs4_7_Case.basis
	a := rs4_7_Case.a
	b := rs4_7_Case.b

	commitment := commit(a, b, basis)
	if !proveCommitmentsLogInteractive(commitment, basis.g, basis.h, basis.q, a, b) {
		t.Errorf("should verify")
	}
	badA := []*big.Int{big.NewInt(3), a[1], a[2], a[3]}
	if proveCommitmentsLogInteractive(commitment, basis.g, basis.h, basis.q, badA, b) {
		t.Errorf("should not verify")
	}

	proof, _ := proveCommitmentsLog(basis.g, basis.h, basis.q, a, b, nil)
	if !verifyCommitmentsLog(proof, len(a), commitment, basis.g, basis.h, basis.q, nil) {
		t.Errorf("should verify")
	}
	badProof, _ := proveCommitmentsLog(basis.g, basis.h, basis.q, badA, b, nil)
	if verifyCommitmentsLog(badProof, len(a), commitment, basis.g, basis.h, basis.q, nil) {
		t.Errorf("should not verify")
	}
}

func TestRS4_7_FrozenHeart(t *testing.T) {
	basis := rs4_7_Case.basis
	a := rs4_7_Case.a

	// Frozen Heart attack: verifyCommitmentsLogFrozenHeart never hashes n
	// (nor the commitment p itself) into the Fiat-Shamir transcript,
	// so each challenge u_i depends only on L_i, R_i. That means L, R,
	// and the final revealed scalars can all be picked with no witness
	// (a, b) in mind at all, and the "commitment" can be solved for last,
	// exactly like the v' trick in
	// https://blog.trailofbits.com/2022/04/15/the-frozen-heart-vulnerability-in-bulletproofs
	inf := new(bn254.G1Affine).SetInfinity()
	frozenHeartProof := InnerProductProof{
		L:     []*bn254.G1Affine{inf, inf},
		R:     []*bn254.G1Affine{inf, inf},
		lastA: big.NewInt(42),
		lastB: big.NewInt(1337),
	}

	// Replay exactly what verifyCommitmentsLogFrozenHeart does to fold
	// g, h for this L, R sequence, so gP[0], hP[0] are known in advance.
	gP, hP := basis.g, basis.h
	var transcript Transcript
	u, uInv := new(big.Int), new(big.Int)
	for i := range frozenHeartProof.L {
		l, r := frozenHeartProof.L[i], frozenHeartProof.R[i]
		transcript.Write(fmt.Sprintf("l%d", i), l.Marshal())
		transcript.Write(fmt.Sprintf("r%d", i), r.Marshal())
		u.SetBytes(transcript.Read("u"))
		uInv.ModInverse(u, bn254.ID.ScalarField())
		gP = foldG(gP, uInv)
		hP = foldG(hP, u)
	}

	// L = R = infinity contribute nothing to the folded commitment, so
	// the base-case check forces exactly:
	//   frozenHeartCommitment = gP[0]*lastA + hP[0]*lastB + q*(lastA*lastB)
	// This is solved backwards, with no (a, b) opening it whatsoever.
	tmp, tmpi := new(bn254.G1Affine), new(big.Int)
	frozenHeartCommitment := new(bn254.G1Affine).ScalarMultiplication(gP[0], frozenHeartProof.lastA)
	frozenHeartCommitment.Add(frozenHeartCommitment, tmp.ScalarMultiplication(hP[0], frozenHeartProof.lastB))
	tmpi.Mul(frozenHeartProof.lastA, frozenHeartProof.lastB)
	frozenHeartCommitment.Add(frozenHeartCommitment, tmp.ScalarMultiplication(basis.q, tmpi))

	if !verifyCommitmentsLogFrozenHeart(frozenHeartProof, len(a), frozenHeartCommitment, basis.g, basis.h, basis.q, nil) {
		t.Errorf("should verify")
	}
	if verifyCommitmentsLog(frozenHeartProof, len(a), frozenHeartCommitment, basis.g, basis.h, basis.q, nil) {
		t.Errorf("should not verify")
	}
}

var rs4_7_Case = struct {
	basis bulletProofBasis
	a     []*big.Int
	b     []*big.Int
}{
	basis: bulletProofBasis{
		g: []*bn254.G1Affine{
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("6286155310766333871795042970372566906087502116590250812133967451320632869759")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("2167390362195738854837661032213065766665495464946848931705307210578191331138")),
			},
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("6981010364086016896956769942642952706715308592529989685498391604818592148727")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("8391728260743032188974275148610213338920590040698592463908691408719331517047")),
			},
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("15884001095869889564203381122824453959747209506336645297496580404216889561240")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("14397810633193722880623034635043699457129665948506123809325193598213289127838")),
			},
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("6756792584920245352684519836070422133746350830019496743562729072905353421352")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("3439606165356845334365677247963536173939840949797525638557303009070611741415")),
			},
		},
		h: []*bn254.G1Affine{
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("13728162449721098615672844430261112538072166300311022796820929618959450231493")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("12153831869428634344429877091952509453770659237731690203490954547715195222919")),
			},
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("17471368056527239558513938898018115153923978020864896155502359766132274520000")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("4119036649831316606545646423655922855925839689145200049841234351186746829602")),
			},
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("8730867317615040501447514540731627986093652356953339319572790273814347116534")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("14893717982647482203420298569283769907955720318948910457352917488298566832491")),
			},
			&bn254.G1Affine{
				X: *new(fp.Element).SetBigInt(ecc.Int10("419294495583131907906527833396935901898733653748716080944177732964425683442")),
				Y: *new(fp.Element).SetBigInt(ecc.Int10("14467906227467164575975695599962977164932514254303603096093942297417329342836")),
			},
		},
		q: &bn254.G1Affine{
			X: *new(fp.Element).SetBigInt(ecc.Int10("11573005146564785208103371178835230411907837176583832948426162169859927052980")),
			Y: *new(fp.Element).SetBigInt(ecc.Int10("895714868375763218941449355207566659176623507506487912740163487331762446439")),
		},
	},
	a: bigs(4, 2, 42, 420),
	b: bigs(1, 3, 7, 13),
}

func TestRS4_3(t *testing.T) {
	commit := func(f, gamma []fr.Element, g, b *bn254.G1Affine) []*bn254.G1Affine {
		tmp, tmpi := new(bn254.G1Affine), new(big.Int)
		commitment := make([]*bn254.G1Affine, len(f))
		for i := range commitment {
			commitment[i] = new(bn254.G1Affine).ScalarMultiplication(g, f[i].BigInt(tmpi))
			commitment[i].Add(commitment[i], tmp.ScalarMultiplication(b, gamma[i].BigInt(tmpi)))
		}
		return commitment
	}
	type Proof struct {
		y  fr.Element
		pi fr.Element
	}
	prove := func(f, gamma []fr.Element, u fr.Element) Proof {
		polys := [][]fr.Element{f, gamma}
		es := []*fr.Element{new(fr.Element).SetZero(), new(fr.Element).SetZero()}
		cup, up := new(fr.Element), new(fr.Element)
		for i, cs := range polys {
			e := es[i]
			up.SetOne()
			for _, c := range cs {
				cup.Mul(&c, up)
				e.Add(e, cup)
				up.Mul(up, &u)
			}
		}
		return Proof{y: *es[0], pi: *es[1]}
	}
	verify := func(commitment []*bn254.G1Affine, g, b *bn254.G1Affine, u fr.Element, proof Proof) bool {
		lhs := new(bn254.G1Affine).SetInfinity()
		cup, up, bi := new(bn254.G1Affine), new(fr.Element), new(big.Int)
		up.SetOne()
		for _, c := range commitment {
			cup.ScalarMultiplication(c, up.BigInt(bi))
			lhs.Add(lhs, cup)
			up.Mul(up, &u)
		}

		rhs := new(bn254.G1Affine).SetInfinity()
		rhs.Add(rhs, cup.ScalarMultiplication(g, proof.y.BigInt(bi)))
		rhs.Add(rhs, cup.ScalarMultiplication(b, proof.pi.BigInt(bi)))

		return lhs.Equal(rhs)
	}

	// g and b are elliptic points whose logarithm the prover does not know.
	g := &bn254.G1Affine{
		X: *new(fp.Element).SetBigInt(ecc.Int10("6286155310766333871795042970372566906087502116590250812133967451320632869759")),
		Y: *new(fp.Element).SetBigInt(ecc.Int10("2167390362195738854837661032213065766665495464946848931705307210578191331138")),
	}
	b := &bn254.G1Affine{
		X: *new(fp.Element).SetBigInt(ecc.Int10("12848606535045587128788889317230751518392478691112375569775390095112330602489")),
		Y: *new(fp.Element).SetBigInt(ecc.Int10("18818936887558347291494629972517132071247847502517774285883500818572856935411")),
	}

	// Prover commits to the polynomial f(x).
	f := []fr.Element{fr.NewElement(666), fr.NewElement(1337), fr.NewElement(42)}
	// gamma is just a random polynomial used by the prover to prevent
	// others from guess f(x), in case the problem at hand requires f(x)
	// to be chosen among a small number of candidates.
	gamma := []fr.Element{fr.NewElement(111), fr.NewElement(222), fr.NewElement(333)}
	commitment := commit(f, gamma, g, b)

	// Verifier chooses u, and asks prover to evaluate f(u).
	ui, _ := rand.Int(rand.Reader, fr.Modulus())
	u := *new(fr.Element).SetBigInt(ui)

	// Check that only proofs evaluated at u are verified.
	tests := []struct {
		f  []fr.Element
		x  fr.Element
		ok bool
	}{
		{f: f, x: u, ok: true},
		{f: f, x: *new(fr.Element).Add(&u, new(fr.Element).SetInt64(1)), ok: false},
		{f: f, x: *new(fr.Element).Add(&u, new(fr.Element).SetInt64(-1)), ok: false},
		{f: f, x: *new(fr.Element).SetZero(), ok: false},
		{f: []fr.Element{fr.NewElement(9), f[1], f[2]}, x: u, ok: false},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			proof := prove(test.f, gamma, test.x)
			if v := verify(commitment, g, b, u, proof); v != test.ok {
				t.Errorf("verify got %v want %v", v, test.ok)
			}
		})
	}
}

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

type bulletProof struct {
	c    *bn254.G1Affine
	tu   *big.Int
	pilr *big.Int
	pit  *big.Int
	ipp  InnerProductProof
}

func bulletProve(a, b []*big.Int, cm bulletProofCommitment, basis bulletProofBasis, secret bulletProverSecret) bulletProof {
	var transcript Transcript
	transcript.Write("n", binary.BigEndian.AppendUint64(nil, uint64(cm.n)))
	transcript.Write("a", cm.a.Marshal())
	transcript.Write("s", cm.s.Marshal())
	transcript.Write("v", cm.v.Marshal())
	transcript.Write("t1", cm.t1.Marshal())
	transcript.Write("t2", cm.t2.Marshal())

	u := new(big.Int).SetBytes(transcript.Read("u"))

	proof := bulletProof{
		c:    new(bn254.G1Affine).SetInfinity(),
		tu:   big.NewInt(0),
		pilr: big.NewInt(0),
		pit:  big.NewInt(0),
	}
	// Compute t(u) = a*b + (a*sr + b*sl)*u + sl*sr*u^2.
	// t(u) a*b term.
	v, tmpi := big.NewInt(0), new(big.Int)
	for i := range a {
		v.Add(v, tmpi.Mul(a[i], b[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	proof.tu.Add(proof.tu, v)
	// t(u) (a*sr + b*sl)*u term.
	v.SetInt64(0)
	for i := range a {
		v.Add(v, tmpi.Mul(a[i], secret.sr[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	for i := range b {
		v.Add(v, tmpi.Mul(b[i], secret.sl[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	v.Mul(v, u)
	proof.tu.Add(proof.tu, v)
	// t(u) sl*sr*u^2 term.
	v.SetInt64(0)
	for i := range secret.sl {
		v.Add(v, tmpi.Mul(secret.sl[i], secret.sr[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	v.Mul(v, u)
	v.Mul(v, u)
	proof.tu.Add(proof.tu, v)

	// Compute π_lr
	proof.pilr.Add(proof.pilr, secret.alpha)
	proof.pilr.Add(proof.pilr, tmpi.Mul(secret.beta, u))

	// Compute π_t
	proof.pit.Add(proof.pit, secret.gamma)
	proof.pit.Add(proof.pit, tmpi.Mul(secret.tau1, u))
	tmpi.Mul(secret.tau2, u)
	proof.pit.Add(proof.pit, tmpi.Mul(tmpi, u))

	// Compute c = lu*g + lr*h, which is a commitment to the vectors
	// l(x) and r(x) evaluated at u:
	// 	l(x) = a + sl*x
	// 	r(x) = u + sr*x
	lu := make([]*big.Int, len(a))
	ru := make([]*big.Int, len(b))
	for i := range lu {
		lu[i] = new(big.Int).Mul(secret.sl[i], u)
		lu[i].Add(lu[i], a[i])
	}
	for i := range ru {
		ru[i] = new(big.Int).Mul(secret.sr[i], u)
		ru[i].Add(ru[i], b[i])
	}
	tmp := new(bn254.G1Affine)
	for i := range basis.g {
		proof.c.Add(proof.c, tmp.ScalarMultiplication(basis.g[i], lu[i]))
	}
	for i := range basis.h {
		proof.c.Add(proof.c, tmp.ScalarMultiplication(basis.h[i], ru[i]))
	}

	// Create the inner product proof.
	transcript.Write("c", proof.c.Marshal())
	w := new(big.Int).SetBytes(transcript.Read("w"))
	q := new(bn254.G1Affine).ScalarMultiplication(basis.q, w)
	proof.ipp, transcript = proveCommitmentsLog(basis.g, basis.h, q, lu, ru, transcript)

	return proof
}

func bulletProofVerify(proof bulletProof, cm bulletProofCommitment, basis bulletProofBasis) bool {
	var transcript Transcript
	transcript.Write("n", binary.BigEndian.AppendUint64(nil, uint64(cm.n)))
	transcript.Write("a", cm.a.Marshal())
	transcript.Write("s", cm.s.Marshal())
	transcript.Write("v", cm.v.Marshal())
	transcript.Write("t1", cm.t1.Marshal())
	transcript.Write("t2", cm.t2.Marshal())

	u := new(big.Int).SetBytes(transcript.Read("u"))

	// Make q unpredictable to prevent fake proofs where t(u) != l(u)*r(u).
	//
	// verifyCommitmentsLog shows only that ctuq = <l,G> + <r,H> + <l,r>*Q
	// while proof.tu itself may not be l*r, as an attacker can send
	// 	proof.c = C + (<l,r> - 999)*Q
	// 	proof.tu = <l,r> + 999
	// so proof.c + proof.tu*Q = C + <l,r>*Q = <l,G> + <r,H> + <l,r>*Q,
	// yet t(u) != <l(u), r(u)>.
	//
	// This attack is prevented by making verifyCommitmentsLog check instead
	// 	proof.c + proof.tu*(w*Q) ?= <l,G> + <r,H> + <l,r>*(w*Q)
	// where w = SHA256(proof.c)
	// Since SHA256 is a one-way function, the attacker is now no longer
	// able to solve for bogus proof.c and proof.tu.
	//
	// This is the u^x operation (u is our basis.q, x is our w) in Protocol 1 in https://eprint.iacr.org/2017/1066
	// Also, see https://github.com/zkcrypto/bulletproofs/blob/2bc6cb73f718dd2406d70d3c55f0fb85a87a4a8f/src/range_proof.rs#L243
	transcript.Write("c", proof.c.Marshal())
	w := new(big.Int).SetBytes(transcript.Read("w"))
	q := new(bn254.G1Affine).ScalarMultiplication(basis.q, w)

	ctuq := new(bn254.G1Affine).ScalarMultiplication(q, proof.tu)
	ctuq.Add(ctuq, proof.c)
	if !verifyCommitmentsLog(proof.ipp, cm.n, ctuq, basis.g, basis.h, q, transcript) {
		return false
	}

	// Verifier alone checks l(u), r(u), t(u) matches commitment.
	// Check that l(u), r(u) match commitment in one go.
	tmp, tmpi := new(bn254.G1Affine), new(big.Int)
	rhs := ctuq.Set(cm.a)
	rhs.Add(rhs, tmp.ScalarMultiplication(cm.s, u))
	tmpi.SetInt64(0)
	tmpi.Sub(tmpi, proof.pilr)
	rhs.Add(rhs, tmp.ScalarMultiplication(basis.b, tmpi))
	if !proof.c.Equal(rhs) {
		return false
	}
	// Check that t(u) is evaluated correctly.
	tuq := ctuq.ScalarMultiplication(basis.q, proof.tu)
	rhs = new(bn254.G1Affine).Set(cm.v)
	rhs.Add(rhs, tmp.ScalarMultiplication(cm.t1, u))
	rhs.Add(rhs, tmp.ScalarMultiplication(cm.t2, tmpi.Mul(u, u)))
	tmpi.SetInt64(0)
	tmpi.Sub(tmpi, proof.pit)
	rhs.Add(rhs, tmp.ScalarMultiplication(basis.b, tmpi))
	if !tuq.Equal(rhs) {
		return false
	}
	return true
}

func bulletProofVerifyFrozenHeart(proof bulletProof, cm bulletProofCommitment, basis bulletProofBasis) bool {
	// This variant of bulletProofVerify is susceptible to the frozen heart
	// vulnerability, since it does not add cm to transcript.
	var transcript Transcript
	u := new(big.Int).SetBytes(transcript.Read("u"))

	ctuq := new(bn254.G1Affine).ScalarMultiplication(basis.q, proof.tu)
	ctuq.Add(ctuq, proof.c)
	if !verifyCommitmentsLog(proof.ipp, cm.n, ctuq, basis.g, basis.h, basis.q, transcript) {
		return false
	}

	// Verifier alone checks l(u), r(u), t(u) matches commitment.
	// Check that l(u), r(u) match commitment in one go.
	tmp, tmpi := new(bn254.G1Affine), new(big.Int)
	rhs := ctuq.Set(cm.a)
	rhs.Add(rhs, tmp.ScalarMultiplication(cm.s, u))
	tmpi.SetInt64(0)
	tmpi.Sub(tmpi, proof.pilr)
	rhs.Add(rhs, tmp.ScalarMultiplication(basis.b, tmpi))
	if !proof.c.Equal(rhs) {
		return false
	}
	// Check that t(u) is evaluated correctly.
	tuq := ctuq.ScalarMultiplication(basis.q, proof.tu)
	rhs = new(bn254.G1Affine).Set(cm.v)
	rhs.Add(rhs, tmp.ScalarMultiplication(cm.t1, u))
	rhs.Add(rhs, tmp.ScalarMultiplication(cm.t2, tmpi.Mul(u, u)))
	tmpi.SetInt64(0)
	tmpi.Sub(tmpi, proof.pit)
	rhs.Add(rhs, tmp.ScalarMultiplication(basis.b, tmpi))
	if !tuq.Equal(rhs) {
		return false
	}
	return true
}

type bulletProofCommitment struct {
	n  int
	a  *bn254.G1Affine
	s  *bn254.G1Affine
	v  *bn254.G1Affine
	t1 *bn254.G1Affine
	t2 *bn254.G1Affine
}

func bulletProofCommit(a, b []*big.Int, basis bulletProofBasis, secret bulletProverSecret) bulletProofCommitment {
	cm := bulletProofCommitment{
		n:  len(a),
		a:  new(bn254.G1Affine).SetInfinity(),
		s:  new(bn254.G1Affine).SetInfinity(),
		v:  new(bn254.G1Affine).SetInfinity(),
		t1: new(bn254.G1Affine).SetInfinity(),
		t2: new(bn254.G1Affine).SetInfinity(),
	}
	tmp := new(bn254.G1Affine)

	// Commit to the polynomials l(x), r(x), and their product t(x).
	// 	l(x) = a + sl*x
	// 	r(x) = b + sr*x
	// 	t(x) = l(x)*r(x) = a*b + (a*sr + b*sl)*x + sl*sr*x^2
	// See https://rareskills.io/post/zk-multiplication .
	//
	// Compute a, which is a commitment to a, b in l(x), r(x).
	for i := range a {
		cm.a.Add(cm.a, tmp.ScalarMultiplication(basis.g[i], a[i]))
	}
	for i := range b {
		cm.a.Add(cm.a, tmp.ScalarMultiplication(basis.h[i], b[i]))
	}
	cm.a.Add(cm.a, tmp.ScalarMultiplication(basis.b, secret.alpha))

	// Compute s, which is a commitment to sl, sr in l(x), r(x).
	for i := range secret.sl {
		cm.s.Add(cm.s, tmp.ScalarMultiplication(basis.g[i], secret.sl[i]))
	}
	for i := range secret.sr {
		cm.s.Add(cm.s, tmp.ScalarMultiplication(basis.h[i], secret.sr[i]))
	}
	cm.s.Add(cm.s, tmp.ScalarMultiplication(basis.b, secret.beta))

	// Compute v, which is a commitment to a*b in t(x).
	v, tmpi := big.NewInt(0), new(big.Int)
	for i := range a {
		v.Add(v, tmpi.Mul(a[i], b[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	cm.v.Add(cm.v, tmp.ScalarMultiplication(basis.q, v))
	cm.v.Add(cm.v, tmp.ScalarMultiplication(basis.b, secret.gamma))

	// Compute t1, which is a commitment to (a*sr + b*sl) in t(x).
	v.SetInt64(0)
	for i := range a {
		v.Add(v, tmpi.Mul(a[i], secret.sr[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	for i := range b {
		v.Add(v, tmpi.Mul(b[i], secret.sl[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	cm.t1.Add(cm.t1, tmp.ScalarMultiplication(basis.q, v))
	cm.t1.Add(cm.t1, tmp.ScalarMultiplication(basis.b, secret.tau1))

	// Compute t2, which is a commitment to sl*sr in t(x).
	v.SetInt64(0)
	for i := range secret.sl {
		v.Add(v, tmpi.Mul(secret.sl[i], secret.sr[i]))
		v.Mod(v, bn254.ID.ScalarField())
	}
	cm.t2.Add(cm.t2, tmp.ScalarMultiplication(basis.q, v))
	cm.t2.Add(cm.t2, tmp.ScalarMultiplication(basis.b, secret.tau2))

	return cm
}

type bulletProverSecret struct {
	alpha *big.Int
	beta  *big.Int
	gamma *big.Int
	tau1  *big.Int
	tau2  *big.Int
	sl    []*big.Int
	sr    []*big.Int
}

func newBulletProverSecret(basis bulletProofBasis) bulletProverSecret {
	var s bulletProverSecret
	s.alpha, _ = rand.Int(rand.Reader, bn254.ID.ScalarField())
	s.beta, _ = rand.Int(rand.Reader, bn254.ID.ScalarField())
	s.gamma, _ = rand.Int(rand.Reader, bn254.ID.ScalarField())
	s.tau1, _ = rand.Int(rand.Reader, bn254.ID.ScalarField())
	s.tau2, _ = rand.Int(rand.Reader, bn254.ID.ScalarField())
	s.sl = randInts(bn254.ID.ScalarField(), len(basis.g))
	s.sr = randInts(bn254.ID.ScalarField(), len(basis.h))
	return s
}

type bulletProofBasis struct {
	g []*bn254.G1Affine
	h []*bn254.G1Affine
	q *bn254.G1Affine
	b *bn254.G1Affine
}

func newBulletProofBasis(n int) bulletProofBasis {
	gs := randG1s(2*n + 2)
	return bulletProofBasis{
		g: gs[:n],
		h: gs[n : 2*n],
		q: gs[2*n],
		b: gs[2*n+1],
	}
}

type InnerProductProof struct {
	L     []*bn254.G1Affine
	R     []*bn254.G1Affine
	lastA *big.Int
	lastB *big.Int
}

func proveCommitmentsLog(g, h []*bn254.G1Affine, q *bn254.G1Affine, a, b []*big.Int, transcript Transcript) (InnerProductProof, Transcript) {
	proof := InnerProductProof{
		L:     make([]*bn254.G1Affine, 0),
		R:     make([]*bn254.G1Affine, 0),
		lastA: new(big.Int),
		lastB: new(big.Int),
	}

	// Prepare Fiat Shamir transform.
	v, tmpi := big.NewInt(0), new(big.Int)
	for i := range a {
		v.Add(v, tmpi.Mul(a[i], b[i]))
	}
	p := new(bn254.G1Affine).ScalarMultiplication(q, v)
	tmp := new(bn254.G1Affine)
	for i := range a {
		p.Add(p, tmp.ScalarMultiplication(g[i], a[i]))
	}
	for i := range b {
		p.Add(p, tmp.ScalarMultiplication(h[i], b[i]))
	}
	transcript.Write("p", p.Marshal())

	u, uInv := v, tmpi
	gP := g
	hP := h
	aP := a
	bP := b
	for len(gP) != 1 {
		// Compute off diagonal terms.
		l, r := calcOffDiagonal(gP, hP, q, aP, bP)
		proof.L = append(proof.L, l)
		proof.R = append(proof.R, r)
		round := len(proof.L) - 1
		transcript.Write(fmt.Sprintf("l%d", round), l.Marshal())
		transcript.Write(fmt.Sprintf("r%d", round), r.Marshal())

		// Fiat Shamir Transform.
		u.SetBytes(transcript.Read("u"))

		uInv.ModInverse(u, bn254.ID.ScalarField())
		gP = foldG(gP, uInv)
		hP = foldG(hP, u)
		aP = fold(aP, u)
		bP = fold(bP, uInv)
	}

	proof.lastA.Set(aP[0])
	proof.lastB.Set(bP[0])
	return proof, transcript
}

func verifyCommitmentsLog(proof InnerProductProof, n int, p *bn254.G1Affine, g, h []*bn254.G1Affine, q *bn254.G1Affine, transcript Transcript) bool {
	if (1 << len(proof.L)) != n {
		return false
	}

	u, uInv, tmp, tmpi := new(big.Int), new(big.Int), new(bn254.G1Affine), new(big.Int)
	pPNew := new(bn254.G1Affine)

	// Hash p for the Fiat Shamir transform.
	transcript.Write("p", p.Marshal())

	// Create final folded commitment pP, using L, R vectors in the proof.
	pP := new(bn254.G1Affine).Set(p)
	gP := g
	hP := h
	for i := range proof.L {
		l, r := proof.L[i], proof.R[i]
		transcript.Write(fmt.Sprintf("l%d", i), l.Marshal())
		transcript.Write(fmt.Sprintf("r%d", i), r.Marshal())

		u.SetBytes(transcript.Read("u"))

		tmpi.Mul(u, u)
		pPNew.ScalarMultiplication(l, tmpi)
		pPNew.Add(pPNew, pP)
		tmpi.ModInverse(tmpi, bn254.ID.ScalarField())
		pP.Add(pPNew, tmp.ScalarMultiplication(r, tmpi))

		uInv.ModInverse(u, bn254.ID.ScalarField())
		gP = foldG(gP, uInv)
		hP = foldG(hP, u)
	}

	rhs := new(bn254.G1Affine).ScalarMultiplication(gP[0], proof.lastA)
	rhs.Add(rhs, tmp.ScalarMultiplication(hP[0], proof.lastB))
	u.Mul(proof.lastA, proof.lastB)
	rhs.Add(rhs, tmp.ScalarMultiplication(q, u))
	return pP.Equal(rhs)
}

func verifyCommitmentsLogFrozenHeart(proof InnerProductProof, n int, p *bn254.G1Affine, g, h []*bn254.G1Affine, q *bn254.G1Affine, transcript Transcript) bool {
	if (1 << len(proof.L)) != n {
		return false
	}

	u, uInv, tmp, tmpi := new(big.Int), new(big.Int), new(bn254.G1Affine), new(big.Int)
	pPNew := new(bn254.G1Affine)

	// In this version of verifyCommitmentsLog, we have a frozen heart
	// vulnerability since we do not hash p.

	// Create final folded commitment pP, using L, R vectors in the proof.
	pP := new(bn254.G1Affine).Set(p)
	gP := g
	hP := h
	for i := range proof.L {
		l, r := proof.L[i], proof.R[i]
		transcript.Write(fmt.Sprintf("l%d", i), l.Marshal())
		transcript.Write(fmt.Sprintf("r%d", i), r.Marshal())

		u.SetBytes(transcript.Read("u"))

		tmpi.Mul(u, u)
		pPNew.ScalarMultiplication(l, tmpi)
		pPNew.Add(pPNew, pP)
		tmpi.ModInverse(tmpi, bn254.ID.ScalarField())
		pP.Add(pPNew, tmp.ScalarMultiplication(r, tmpi))

		uInv.ModInverse(u, bn254.ID.ScalarField())
		gP = foldG(gP, uInv)
		hP = foldG(hP, u)
	}

	rhs := new(bn254.G1Affine).ScalarMultiplication(gP[0], proof.lastA)
	rhs.Add(rhs, tmp.ScalarMultiplication(hP[0], proof.lastB))
	u.Mul(proof.lastA, proof.lastB)
	rhs.Add(rhs, tmp.ScalarMultiplication(q, u))
	return pP.Equal(rhs)
}

// A Transcript is a Fiat Shamir transcript.
// All transcript operations require a unique label for domain separation, which
// ensures all transcript reads return unique unguessable values.
type Transcript []byte

func (t *Transcript) Read(label string) []byte {
	*t = append(*t, []byte(label)...)
	h := sha256.Sum256(*t)
	return h[:]
}

func (t *Transcript) Write(label string, data []byte) {
	*t = append(*t, []byte(label)...)
	*t = append(*t, data...)
}

func calcOffDiagonal(g, h []*bn254.G1Affine, q *bn254.G1Affine, a, b []*big.Int) (*bn254.G1Affine, *bn254.G1Affine) {
	tmp, tmpi := new(bn254.G1Affine), new(big.Int)
	// Compute l which is the sum of bottom left terms.
	l := new(bn254.G1Affine).SetInfinity()
	bi := big.NewInt(0)
	for i := 0; i < len(a); i += 2 {
		bi.Add(bi, tmpi.Mul(a[i], b[i+1]))
	}
	l.Add(l, tmp.ScalarMultiplication(q, bi))
	for i := 0; i < len(a); i += 2 {
		l.Add(l, tmp.ScalarMultiplication(g[i+1], a[i]))
	}
	for i := 0; i < len(h); i += 2 {
		l.Add(l, tmp.ScalarMultiplication(h[i], b[i+1]))
	}

	// Compute r which is the sum of top right terms.
	r := new(bn254.G1Affine).SetInfinity()
	bi.SetInt64(0)
	for i := 0; i < len(a); i += 2 {
		bi.Add(bi, tmpi.Mul(a[i+1], b[i]))
	}
	r.Add(r, tmp.ScalarMultiplication(q, bi))
	for i := 0; i < len(a); i += 2 {
		r.Add(r, tmp.ScalarMultiplication(g[i], a[i+1]))
	}
	for i := 0; i < len(h); i += 2 {
		r.Add(r, tmp.ScalarMultiplication(h[i+1], b[i]))
	}

	return l, r
}

func proveCommitmentsLogInteractive(p *bn254.G1Affine, g, h []*bn254.G1Affine, q *bn254.G1Affine, a, b []*big.Int) bool {
	tmp, tmpi := new(bn254.G1Affine), new(big.Int)
	if len(a) == 1 {
		rhs := new(bn254.G1Affine).SetInfinity()
		for i := range a {
			rhs.Add(rhs, tmp.ScalarMultiplication(g[i], a[i]))
		}
		for i := range b {
			rhs.Add(rhs, tmp.ScalarMultiplication(h[i], b[i]))
		}
		rhs.Add(rhs, tmp.ScalarMultiplication(q, tmpi.Mul(a[0], b[0])))
		return p.Equal(rhs)
	}

	// Prover sends verifier off-diagonal terms.
	l, r := calcOffDiagonal(g, h, q, a, b)

	// Verifier picks random u, and computes p', g', h', which are folded
	// versions of p, g, h.
	u, _ := rand.Int(rand.Reader, bn254.ID.ScalarField())
	tmpi.Mul(u, u)
	pPrime := new(bn254.G1Affine).ScalarMultiplication(l, tmpi)
	pPrime.Add(pPrime, p)
	tmpi.ModInverse(tmpi, bn254.ID.ScalarField())
	pPrime.Add(pPrime, tmp.ScalarMultiplication(r, tmpi))
	uInv := new(big.Int).ModInverse(u, bn254.ID.ScalarField())
	gPrime := foldG(g, uInv)
	hPrime := foldG(h, u)

	// Prover privately folds a, b for the next round.
	aPrime := fold(a, u)
	bPrime := fold(b, uInv)
	return proveCommitmentsLogInteractive(pPrime, gPrime, hPrime, q, aPrime, bPrime)
}

func foldG(a []*bn254.G1Affine, x *big.Int) []*bn254.G1Affine {
	tmp := new(bn254.G1Affine)
	xInv := new(big.Int).ModInverse(x, bn254.ID.ScalarField())
	f := make([]*bn254.G1Affine, len(a)/2)
	for i := 0; i < len(a); i += 2 {
		f[i/2] = new(bn254.G1Affine).ScalarMultiplication(a[i], x)
		f[i/2].Add(f[i/2], tmp.ScalarMultiplication(a[i+1], xInv))
	}
	return f
}

func fold(a []*big.Int, x *big.Int) []*big.Int {
	tmp := new(big.Int)
	xInv := new(big.Int).ModInverse(x, bn254.ID.ScalarField())
	f := make([]*big.Int, len(a)/2)
	for i := 0; i < len(a); i += 2 {
		f[i/2] = new(big.Int).Mul(a[i], x)
		f[i/2].Add(f[i/2], tmp.Mul(a[i+1], xInv))
		f[i/2].Mod(f[i/2], bn254.ID.ScalarField())
	}
	return f
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

func bigs(is ...int) []*big.Int {
	bs := make([]*big.Int, len(is))
	for i := range bs {
		bs[i] = big.NewInt(int64(is[i]))
	}
	return bs
}

func randInts(m *big.Int, n int) []*big.Int {
	is := make([]*big.Int, n)
	for i := range is {
		is[i], _ = rand.Int(rand.Reader, m)
	}
	return is
}

func randG1s(n int) []*bn254.G1Affine {
	b := sha256.Sum256([]byte("RareSkills"))
	x := new(fp.Element).SetBytes(b[:])

	gs := make([]*bn254.G1Affine, 0, n)
	for range n {
		gs = append(gs, randG1(x))
	}
	return gs
}

func randG1(x *fp.Element) *bn254.G1Affine {
	_, b := bn254.CurveCoefficients()
	y, one := new(fp.Element), new(fp.Element).SetOne()

	xb := x.Bytes()
	xh := sha256.Sum256(xb[:])
	x.SetBytes(xh[:])
	for {
		y.Mul(x, x)
		y.Mul(y, x)
		y.Add(y, &b)
		if y.Sqrt(y) != nil {
			break
		}
		x.Add(x, one)
	}

	// Pick upper or lower point depending on whether xh is odd or even.
	if (xh[len(xh)-1] & 1) != 0 {
		y.Neg(y)
	}

	point := &bn254.G1Affine{X: *x, Y: *y}
	return point
}
