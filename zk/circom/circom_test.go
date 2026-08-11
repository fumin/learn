package circom

import (
	"bytes"
	_ "embed"
	"flag"
	"fmt"
	"log"
	"math/big"
	"slices"
	"testing"

	"github.com/fumin/learn/zk/circom/witnesscalc"
	"github.com/fumin/nag/field"
)

//go:embed testdata/rs2_3/rs2_3.wasm
var rs23Wasm []byte

func TestNewWitnessCalculator(t *testing.T) {
	tests := []struct {
		w      []byte
		inputs map[string]any
		want   []*big.Int
	}{
		{
			w:      rs23Wasm,
			inputs: map[string]any{"x": big.NewInt(5), "y": big.NewInt(148)},
			want:   []*big.Int{big.NewInt(1), big.NewInt(5), big.NewInt(148), big.NewInt(125), big.NewInt(25), big.NewInt(21904)},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			w, err := witnesscalc.NewCircom2WitnessCalculator(test.w, true)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			got, err := w.CalculateWitness(test.inputs, true)
			if err != nil {
				t.Errorf("%+v", err)
			}
			if !slices.EqualFunc(got, test.want, func(a, b *big.Int) bool { return a.Cmp(b) == 0 }) {
				t.Errorf("Calc(%v) = %v want %v", test.inputs, got, test.want)
			}
		})
	}
}

//go:embed testdata/rs2_3/rs2_3.r1cs.json
var rs23R1CSJson []byte

func TestParseR1CSJson(t *testing.T) {
	r1cs, err := ParseR1CSJson(bytes.NewReader(rs23R1CSJson))
	if err != nil {
		t.Errorf("%+v", err)
	}
	wantR1CS := newRS2_3_R1CS()

	k := field.NewPrimeExtDeg(r1cs.Prime, 1)
	lroStr := [3]string{"l", "r", "o"}
	lro := [3]Matrix{r1cs.L, r1cs.R, r1cs.O}
	wantLRO := [3]Matrix{wantR1CS.L, wantR1CS.R, wantR1CS.O}
	for i := range lro {
		got := denseFromCOO(lro[i])
		want := denseFromCOO(wantLRO[i])
		for j := range got {
			for m := range got[j] {
				g := k.NewZero().SetCoeffs(got[j][m])
				w := k.NewZero().SetCoeffs(want[j][m])
				if !g.Equal(w) {
					t.Errorf("%s[%d][%d] = %v want %v", lroStr[i], j, m, g, w)
				}
			}
		}
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}

func newRS2_3_R1CS() R1CS {
	// Define the circuit of the satisfiability of the below equations:
	//     x^3 + 5*x + -2 == y
	//     y^3 == 3241792
	//
	// It is based on Chapter 3, Module 2 of the RareSkills Zero-Knowledge book.
	// https://rareskills.io/post/r1cs-zkp
	circuit := R1CS{}
	circuit.NumVars = 6
	circuit.NumConstraints = 5

	// Encode the circom r1CS constraints.
	const one, x, y, x3, x2, y2 = 0, 1, 2, 3, 4, 5
	circuit.L = Matrix{Rows: circuit.NumConstraints, Cols: circuit.NumVars, COO: []CooEntry{
		{Row: 0, Col: x, Val: big.NewInt(-1)},  // Constraint 0.
		{Row: 1, Col: x2, Val: big.NewInt(-1)}, // Constraint 1.
		{Row: 2, Col: y, Val: big.NewInt(-1)},  // Constraint 2.
		{Row: 3, Col: y2, Val: big.NewInt(-1)}, // Constraint 3.
	}}
	circuit.R = Matrix{Rows: circuit.NumConstraints, Cols: circuit.NumVars, COO: []CooEntry{
		{Row: 0, Col: x, Val: big.NewInt(1)},
		{Row: 1, Col: x, Val: big.NewInt(1)},
		{Row: 2, Col: y, Val: big.NewInt(1)},
		{Row: 3, Col: y, Val: big.NewInt(1)},
	}}
	circuit.O = Matrix{Rows: circuit.NumConstraints, Cols: circuit.NumVars, COO: []CooEntry{
		{Row: 0, Col: x2, Val: big.NewInt(-1)},
		{Row: 1, Col: x3, Val: big.NewInt(-1)},
		{Row: 2, Col: y2, Val: big.NewInt(-1)},
		{Row: 3, Col: one, Val: big.NewInt(-3241792)},
		// Constraint 4.
		{Row: 4, Col: one, Val: big.NewInt(2)},
		{Row: 4, Col: x, Val: big.NewInt(-5)},
		{Row: 4, Col: y, Val: big.NewInt(1)},
		{Row: 4, Col: x3, Val: big.NewInt(-1)},
	}}

	return circuit
}

// denseFromCOO expands a sparse COO matrix into a dense matrix.
func denseFromCOO(m Matrix) [][]*big.Int {
	dense := make([][]*big.Int, m.Rows)
	for i := range dense {
		dense[i] = make([]*big.Int, m.Cols)
		for j := range dense[i] {
			dense[i][j] = big.NewInt(0)
		}
	}
	for _, e := range m.COO {
		dense[e.Row][e.Col] = e.Val
	}
	return dense
}
