package circom

import (
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"io"
	"math/big"
	"slices"
	"strconv"

	"github.com/pkg/errors"
)

// A CooEntry is an entry of a sparse matrix in Coordinate list (COO) format.
type CooEntry struct {
	Row int
	Col int
	Val *big.Int
}

// A Matrix is a sparse matrix.
type Matrix struct {
	Rows int
	Cols int
	COO  []CooEntry
}

type R1CS struct {
	Prime            *big.Int
	NumVars          int
	NumOutputs       int
	NumPublicInputs  int
	NumPrivateInputs int
	NumConstraints   int
	L                Matrix
	R                Matrix
	O                Matrix
}

type circomLRO [3][]CooEntry

func (lro *circomLRO) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if k := dec.PeekKind(); k != '[' {
		return &json.SemanticError{JSONKind: k}
	}
	if _, err := dec.ReadToken(); err != nil {
		return err
	}

	i := -1
	for dec.PeekKind() != ']' {
		i++
		cstrt := [3]map[string]string{}
		if err := json.UnmarshalDecode(dec, &cstrt); err != nil {
			return err
		}
		for j := range 3 {
			m := cstrt[j]
			keys := make([]int, 0, len(m))
			for k := range m {
				ki, err := strconv.Atoi(k)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("%d %d", i, j))
				}
				keys = append(keys, ki)
			}
			slices.Sort(keys)

			for _, k := range keys {
				v := m[strconv.Itoa(k)]
				val := new(big.Int)
				if _, ok := val.SetString(v, 10); !ok {
					return errors.Errorf("%d %d %d", i, j, k)
				}
				lro[j] = append(lro[j], CooEntry{Row: i, Col: k, Val: val})
			}
		}
	}

	if _, err := dec.ReadToken(); err != nil {
		return err
	}
	return nil
}

// ParseR1CS parses a circom generated R1CS json format output.
func ParseR1CSJson(r io.Reader) (R1CS, error) {
	out := struct {
		Prime            string    `json:"prime"`
		NVars            int       `json:"nVars"`
		NumOutputs       int       `json:"nOutputs"`
		NumPublicInputs  int       `json:"nPubInputs"`
		NumPrivateInputs int       `json:"nPrvInputs"`
		NConstraints     int       `json:"nConstraints"`
		Constraints      circomLRO `json:"constraints"`
	}{}
	if err := json.UnmarshalRead(r, &out); err != nil {
		return R1CS{}, errors.Wrap(err, "")
	}

	r1cs := R1CS{
		Prime:            new(big.Int),
		NumVars:          out.NVars,
		NumOutputs:       out.NumOutputs,
		NumPublicInputs:  out.NumPublicInputs,
		NumPrivateInputs: out.NumPrivateInputs,
		NumConstraints:   out.NConstraints,
		L:                Matrix{Rows: out.NConstraints, Cols: out.NVars, COO: out.Constraints[0]},
		R:                Matrix{Rows: out.NConstraints, Cols: out.NVars, COO: out.Constraints[1]},
		O:                Matrix{Rows: out.NConstraints, Cols: out.NVars, COO: out.Constraints[2]},
	}
	if _, ok := r1cs.Prime.SetString(out.Prime, 10); !ok {
		return R1CS{}, errors.Errorf("SetString error %s", out.Prime)
	}
	return r1cs, nil
}
