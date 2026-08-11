package circom

import (
	"bytes"
	_ "embed"
	"testing"
)

//go:embed testdata/rs2_3/rs2_3.r1cs
var rs23R1CS []byte

func TestParseR1CS(t *testing.T) {
	r1cs, err := ParseR1CS(bytes.NewReader(rs23R1CS))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	r1csJson, err := ParseR1CSJson(bytes.NewReader(rs23R1CSJson))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if r1cs.Prime.Cmp(r1csJson.Prime) != 0 {
		t.Errorf("%d != %d", r1cs.Prime, r1csJson.Prime)
	}
	if r1cs.NumVars != r1csJson.NumVars {
		t.Errorf("%d != %d", r1cs.NumVars, r1csJson.NumVars)
	}
	if r1cs.NumOutputs != r1csJson.NumOutputs {
		t.Errorf("%d != %d", r1cs.NumOutputs, r1csJson.NumOutputs)
	}
	if r1cs.NumPublicInputs != r1csJson.NumPublicInputs {
		t.Errorf("%d != %d", r1cs.NumPublicInputs, r1csJson.NumPublicInputs)
	}
	if r1cs.NumPrivateInputs != r1csJson.NumPrivateInputs {
		t.Errorf("%d != %d", r1cs.NumVars, r1csJson.NumVars)
	}
	if r1cs.NumConstraints != r1csJson.NumConstraints {
		t.Errorf("%d != %d", r1cs.NumConstraints, r1csJson.NumConstraints)
	}
	ms := []Matrix{r1cs.L, r1cs.R, r1cs.O}
	mjs := []Matrix{r1csJson.L, r1csJson.R, r1csJson.O}
	for i, m := range ms {
		mj := mjs[i]
		if len(m.COO) != len(mj.COO) {
			t.Errorf("%d != %d", len(m.COO), len(mj.COO))
		}
		for j, e := range m.COO {
			ej := mj.COO[j]

			if e.Row != ej.Row {
				t.Errorf("%v %v", e, ej)
			}
			if e.Col != ej.Col {
				t.Errorf("%v %v", e, ej)
			}
			if e.Val.Cmp(ej.Val) != 0 {
				t.Errorf("%v %v", e, ej)
			}
		}
	}
}
