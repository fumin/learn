package circom

import (
	"bytes"
	"encoding/binary"
	"io"
	"math/big"

	"github.com/pkg/errors"
)

// Section types, as defined in the r1cs binary format.
// https://github.com/iden3/r1csfile/blob/master/doc/r1cs_bin_format.md
const (
	r1csSectionHeader      = 1
	r1csSectionConstraints = 2
)

// ParseR1CS parses a circom generated R1CS binary format output.
func ParseR1CS(r io.Reader) (R1CS, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return R1CS{}, errors.Wrap(err, "")
	}
	if string(magic[:]) != "r1cs" {
		return R1CS{}, errors.Errorf("bad magic %q", magic)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return R1CS{}, errors.Wrap(err, "")
	}

	var numSections uint32
	if err := binary.Read(r, binary.LittleEndian, &numSections); err != nil {
		return R1CS{}, errors.Wrap(err, "")
	}

	// Sections need not appear in any particular order, and the constraints
	// section cannot be decoded without the field size found in the header
	// section, so buffer every section's raw bytes by type before decoding.
	sections := make(map[uint32][]byte, numSections)
	for range numSections {
		var sType uint32
		if err := binary.Read(r, binary.LittleEndian, &sType); err != nil {
			return R1CS{}, errors.Wrap(err, "")
		}
		var sSize uint64
		if err := binary.Read(r, binary.LittleEndian, &sSize); err != nil {
			return R1CS{}, errors.Wrap(err, "")
		}
		buf := make([]byte, sSize)
		if _, err := io.ReadFull(r, buf); err != nil {
			return R1CS{}, errors.Wrap(err, "")
		}
		sections[sType] = buf
	}

	headerB, ok := sections[r1csSectionHeader]
	if !ok {
		return R1CS{}, errors.Errorf("missing header section")
	}
	hdr, err := parseR1CSHeader(headerB)
	if err != nil {
		return R1CS{}, errors.Wrap(err, "")
	}

	r1cs := R1CS{
		Prime:            hdr.prime,
		NumVars:          hdr.numVars,
		NumOutputs:       hdr.numPubOut,
		NumPublicInputs:  hdr.numPubIn,
		NumPrivateInputs: hdr.numPrvIn,
		NumConstraints:   hdr.numConstraints,
		L:                Matrix{Rows: hdr.numConstraints, Cols: hdr.numVars},
		R:                Matrix{Rows: hdr.numConstraints, Cols: hdr.numVars},
		O:                Matrix{Rows: hdr.numConstraints, Cols: hdr.numVars},
	}

	constraintsB, ok := sections[r1csSectionConstraints]
	if !ok {
		return R1CS{}, errors.Errorf("missing constraints section")
	}
	if err := parseR1CSConstraints(constraintsB, hdr.fieldSize, hdr.numConstraints, &r1cs.L, &r1cs.R, &r1cs.O); err != nil {
		return R1CS{}, errors.Wrap(err, "")
	}

	return r1cs, nil
}

// r1csHeader is the decoded content of the header section (type 1).
type r1csHeader struct {
	fieldSize      uint32
	prime          *big.Int
	numVars        int
	numPubOut      int
	numPubIn       int
	numPrvIn       int
	numConstraints int
}

func parseR1CSHeader(b []byte) (r1csHeader, error) {
	br := bytes.NewReader(b)

	var fieldSize uint32
	if err := binary.Read(br, binary.LittleEndian, &fieldSize); err != nil {
		return r1csHeader{}, errors.Wrap(err, "")
	}

	primeB := make([]byte, fieldSize)
	if _, err := io.ReadFull(br, primeB); err != nil {
		return r1csHeader{}, errors.Wrap(err, "")
	}

	var numVars, numPubOut, numPubIn, numPrvIn uint32
	for _, f := range []*uint32{&numVars, &numPubOut, &numPubIn, &numPrvIn} {
		if err := binary.Read(br, binary.LittleEndian, f); err != nil {
			return r1csHeader{}, errors.Wrap(err, "")
		}
	}

	var numLabels uint64
	if err := binary.Read(br, binary.LittleEndian, &numLabels); err != nil {
		return r1csHeader{}, errors.Wrap(err, "")
	}

	var numConstraints uint32
	if err := binary.Read(br, binary.LittleEndian, &numConstraints); err != nil {
		return r1csHeader{}, errors.Wrap(err, "")
	}

	return r1csHeader{
		fieldSize:      fieldSize,
		prime:          leBytesToBigInt(primeB),
		numVars:        int(numVars),
		numPubOut:      int(numPubOut),
		numPubIn:       int(numPubIn),
		numPrvIn:       int(numPrvIn),
		numConstraints: int(numConstraints),
	}, nil
}

// parseR1CSConstraints decodes the constraints section (type 2) into the L,
// R, and O matrices. Each constraint contributes one row to each matrix,
// encoded as three linear combinations (for L, R, and O respectively): a
// non-zero factor count, followed by that many (wire ID, coefficient) pairs
// sorted by ascending wire ID.
func parseR1CSConstraints(b []byte, fieldSize uint32, numConstraints int, l, r, o *Matrix) error {
	br := bytes.NewReader(b)
	mats := [3]*Matrix{l, r, o}
	for row := range numConstraints {
		for _, m := range mats {
			var numFactors uint32
			if err := binary.Read(br, binary.LittleEndian, &numFactors); err != nil {
				return errors.Wrap(err, "")
			}
			for range numFactors {
				var wireID uint32
				if err := binary.Read(br, binary.LittleEndian, &wireID); err != nil {
					return errors.Wrap(err, "")
				}
				coeffB := make([]byte, fieldSize)
				if _, err := io.ReadFull(br, coeffB); err != nil {
					return errors.Wrap(err, "")
				}
				m.COO = append(m.COO, CooEntry{Row: row, Col: int(wireID), Val: leBytesToBigInt(coeffB)})
			}
		}
	}
	return nil
}

// leBytesToBigInt interprets b as a little-endian unsigned integer.
func leBytesToBigInt(b []byte) *big.Int {
	be := make([]byte, len(b))
	for i, v := range b {
		be[len(b)-1-i] = v
	}
	return new(big.Int).SetBytes(be)
}
