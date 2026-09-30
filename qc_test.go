package learn

import (
	"flag"
	"log"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"strconv"
	"testing"

	"github.com/fumin/learn/q"
	"github.com/fumin/nag"
	"github.com/fumin/nag/field"
	"github.com/pkg/errors"
)

func TestHenryYuen_20260923Talk_31m48s(t *testing.T) {
	m := newBitMat([][]int{
		{1, 0, 0, 1},
		{0, 1, 1, 1},
		{0, 0, 1, 1},
		{1, 1, 0, 0},
	})
	n := len(m)

	// Compute inverse of m.
	k := m[0][0]
	one := k.NewOne()
	mInv, err := invMat(m)
	if err != nil {
		t.Errorf("%+v", err)
	}
	ancillas := q.Sys(q.Z0, n+1)
	for i := n + 2; i <= 2*n; i++ {
		ancillas = 𐌈(ancillas, q.Sys(q.Z0, i))
	}

	// Compute the fanout circuit.
	// Begin by applying m on data qubits as control and ancillas as target.
	// After operation, ancillas would hold the classical result bits.
	nots := make([]int, 0, n)
	gates := make([]*q.Dense, 0, 3*n)
	for i := range n {
		nots = nots[:0]
		for j, x := range getCol(m, i) {
			if x.Equal(one) {
				nots = append(nots, n+j+1)
			}
		}
		gates = append(gates, fanout(i+1, nots...))
	}
	// Erase data qubits, without modifying the ancillas holding the result.
	// This is achieved by applying mInv on the data qubits as target.
	// After operation, data qubits would be in the initial state of the ancillas, which is zero.
	for i := range n {
		nots = nots[:0]
		for j, x := range getCol(mInv, i) {
			if x.Equal(one) {
				nots = append(nots, j+1)
			}
		}
		gates = append(gates, fanout(n+i+1, nots...))
	}
	// Move answer from ancillas to data qubits.
	for i := range n {
		gates = append(gates, swap(i+1, n+i+1))
	}
	slices.Reverse(gates)
	mFanout := q.Dot(gates...)

	bits, col := make([]int, n), make([]int, n)
	for i := range n {
		littleEndian(bits, 1<<i)
		ψ := littleEndianState(bits)
		ψ = 𐌈(ψ, ancillas)

		// Run the circuit.
		mψ := q.Dot(ancillas.H(), mFanout, ψ)

		// Check that mψ == m[:, i].
		sa := stateLittleEndian(mψ, 1e-6)
		if len(sa) != 1 {
			t.Errorf("%v", sa)
		}
		for j, e := range getCol(m, i) {
			col[j] = int(e.Coeffs()[0].Int64())
		}
		if !slices.Equal(sa[0].state, col) {
			t.Errorf("%v want %v", sa[0].state, col)
		}
	}
}

func TestHenryYuen_20260923Talk_24m38s(t *testing.T) {
	// https://youtu.be/6HHlaw7adVg?si=D-fq8UGyToR087zU&t=1478
	// Figure 1, Quantum Fan-out is Powerful, Peter Hoyer, Robert Spalek
	parityFromFanout := func(n int) *q.Dense {
		// Hadamard.
		h := q.T2([][]complex64{{1 / math.Sqrt2, 1 / math.Sqrt2}, {1 / math.Sqrt2, -1 / math.Sqrt2}})
		// Control gates on particle n+1.
		z0 := q.Sys(𐌈(q.Z0, q.Z0.H()), n+1)
		z1 := q.Sys(𐌈(q.Z1, q.Z1.H()), n+1)

		gates := make([]*q.Dense, 0)
		for i := 1; i <= n+1; i++ {
			gates = append(gates, q.Sys(h, i))
		}

		// Fanout, condition on particle n+1, fans out to 1..n.
		for i := 1; i <= n; i++ {
			cnot := 𐌈(z0, q.Sys(q.One, i)).Add(1, 𐌈(z1, q.Sys(σx, i)))
			gates = append(gates, cnot)
		}

		for i := 1; i <= n+1; i++ {
			gates = append(gates, q.Sys(h, i))
		}
		return q.Dot(gates...)
	}

	for n := 1; n <= 3; n++ {
		parityGate := parity(n)
		parityFanout := parityFromFanout(n).Transpose(parityGate.Axis)
		if err := parityGate.Equal(parityFanout, 1e-6); err != nil {
			t.Errorf("%+v", err)
		}

		bits := make([]int, n)
		for i := range 1 << n {
			littleEndian(bits, i)
			ψ := littleEndianState(bits)
			ψ = 𐌈(ψ, q.Sys(q.Z0, n+1))
			ψParity := 0
			for _, b := range bits {
				ψParity += int(b)
			}
			ψParity = ψParity % 2

			// Check bit flip probability equals parity.
			ψP := q.Dot(parityGate, ψ)
			prob := real(q.Dot(ψP.H(), q.Sys(𐌈(q.Z1, q.Z1.H()), n+1), ψP).D.At(0))
			if int(prob) != ψParity {
				t.Errorf("%v want %v", prob, ψParity)
			}
		}
	}
}

// QCLec book is https://www.scottaaronson.com/qclec.pdf
func TestQCLec28_9_258(t *testing.T) {
	s000 := 𐌈(q.Z0, q.Sys(q.Z0, 2), q.Sys(q.Z0, 3))
	s111 := 𐌈(q.Z1, q.Sys(q.Z1, 2), q.Sys(q.Z1, 3))
	十 := s000.Add(1, s111).Mul(1 / sqrt(2))
	一 := s000.Add(-1, s111).Mul(1 / sqrt(2))

	sysPlus := func(a *q.Dense, s int) *q.Dense {
		b := &q.Dense{Axis: make([]q.Axis, len(a.Axis)), D: a.D}
		copy(b.Axis, a.Axis)
		for i, ax := range b.Axis {
			sys := ax.System.(int)
			b.Axis[i].System = sys + s
		}
		return b
	}
	s0 := 𐌈(十, sysPlus(十, 3), sysPlus(十, 6))
	s1 := 𐌈(一, sysPlus(一, 3), sysPlus(一, 6))

	z0 := 𐌈(σz, q.Sys(σz, 2))
	if err := q.Dot(z0, s0).Equal(s0, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
	if err := q.Dot(z0, s1).Equal(s1, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}

	x9Ops := make([]*q.Dense, 0, 9)
	for i := range 9 {
		x9Ops = append(x9Ops, q.SysReplace(σx, 1, i+1))
	}
	x := 𐌈(x9Ops...)
	if err := q.Dot(x, s0).Equal(s0, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
	if err := q.Dot(x.Mul(-1), s1).Equal(s1, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestQCLecBellStabilizer_252(t *testing.T) {
	ii := 𐌈(q.One, q.Sys(q.One, 2))
	xx := 𐌈(σx, q.Sys(σx, 2))
	yy := 𐌈(σy, q.Sys(σy, 2)).Mul(-1)
	zz := 𐌈(σz, q.Sys(σz, 2))

	x2 := q.Dot(xx, xx).Transpose(xx.Axis)
	if err := x2.Equal(ii, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
	xz := q.Dot(xx, zz).Transpose(xx.Axis)
	if err := xz.Equal(yy, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestQCLec27_3_239(t *testing.T) {
	s000 := 𐌈(q.Z0, q.Sys(q.Z0, 2), q.Sys(q.Z0, 3))
	s111 := 𐌈(q.Z1, q.Sys(q.Z1, 2), q.Sys(q.Z1, 3))
	十 := s000.Add(1, s111).Mul(1 / sqrt(2))
	一 := s000.Add(-1, s111).Mul(1 / sqrt(2))

	一2 := q.PartialDot(q.Sys(σz, 2), 2, 十).Transpose(一.Axis)
	if err := 一.Equal(一2, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

var (
	σx = q.Aσx
	σy = q.Aσy
	σz = q.Aσz
)

func 𐌈(as ...*q.Dense) *q.Dense  { return q.A𐌈(as...) }
func sqrt(x complex64) complex64 { return complex64(cmplx.Sqrt(complex128(x))) }

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}

func fanout(control int, nots ...int) *q.Dense {
	gates := make([]*q.Dense, 0, len(nots))
	for _, x := range nots {
		cnot := 𐌈(𐌈(q.Z0, q.Z0.H()), q.Sys(q.One, 2)).Add(1, 𐌈(𐌈(q.Z1, q.Z1.H()), q.Sys(σx, 2)))
		var g *q.Dense
		if x == 1 {
			g = q.SysReplace(q.SysReplace(cnot, 1, control), 2, x)
		} else {
			g = q.SysReplace(q.SysReplace(cnot, 2, x), 1, control)
		}
		gates = append(gates, g)
	}
	return q.Dot(gates...)
}

func swap(i, j int) *q.Dense {
	swap := q.FromMat([][]complex64{
		{1, 0, 0, 0},
		{0, 0, 1, 0},
		{0, 1, 0, 0},
		{0, 0, 0, 1},
	})
	if i == 2 {
		return q.SysReplace(swap, 1, j)
	}
	return q.SysReplace(q.SysReplace(swap, 1, i), 2, j)
}

func parity(n int) *q.Dense {
	one := q.Sys(q.One, n+1)
	σx := q.Sys(q.Aσx, n+1)
	return 𐌈(parityBit(n, 0), one).Add(1, 𐌈(parityBit(n, 1), σx))
}

func parityBit(n, bit int) *q.Dense {
	if n == 1 {
		if bit == 0 {
			return 𐌈(q.Z0, q.Z0.H())
		}
		return 𐌈(q.Z1, q.Z1.H())
	}

	o0 := q.Sys(𐌈(q.Z0, q.Z0.H()), n)
	o1 := q.Sys(𐌈(q.Z1, q.Z1.H()), n)

	b0, b1 := 0, 1
	if bit == 1 {
		b0, b1 = 1, 0
	}
	return 𐌈(o0, parityBit(n-1, b0)).Add(1, 𐌈(o1, parityBit(n-1, b1)))
}

func bigEndian(bits []int, i int) {
	if i < 0 {
		panic("negative")
	}

	bstr := strconv.FormatInt(int64(i), 2)
	for j := len(bstr) - 1; j >= 0; j-- {
		idx := j + len(bits) - len(bstr)
		bits[idx] = int(bstr[j] - '0')
	}
	for i := range len(bits) - len(bstr) {
		bits[i] = 0
	}
}

func littleEndian(bits []int, i int) {
	bigEndian(bits, i)
	slices.Reverse(bits)
}

func littleEndianState(bits []int) *q.Dense {
	var ψ *q.Dense
	for j, b := range bits {
		ψj := q.Z0
		if b == 1 {
			ψj = q.Z1
		}

		if ψ == nil {
			ψ = q.Sys(ψj, j+1)
		} else {
			ψ = 𐌈(ψ, q.Sys(ψj, j+1))
		}
	}
	return ψ
}

type stateAmplitude struct {
	state     []int
	amplitude complex64
}

func stateLittleEndian(ψ *q.Dense, tol float64) []stateAmplitude {
	// Compute n, the number of particles.
	vol := 1
	for _, s := range ψ.D.Shape() {
		vol *= s
	}
	n, exp2n := 0, vol>>1
	for exp2n != 0 {
		n++
		exp2n >>= 1
	}

	state, at := make([]int, n), make([]int, n)
	sa := make([]stateAmplitude, 0)
	for j := range 1 << n {
		littleEndian(state, j)

		for k := range at {
			sys := ψ.Axis[k].System.(int)
			at[k] = state[sys-1]
		}
		if c := ψ.D.At(at...); cmplx.Abs(complex128(c)) > tol {
			s := stateAmplitude{state: make([]int, n), amplitude: c}
			copy(s.state, state)
			sa = append(sa, s)
		}
	}
	return sa
}

func getCol[E any](m [][]E, j int) []E {
	col := make([]E, len(m))
	for i := range col {
		col[i] = m[i][j]
	}
	return col
}

func newBitMat(mi [][]int) [][]*field.PrimeExt {
	k := field.NewPrimeExtDeg(big.NewInt(2), 1)
	m := make([][]*field.PrimeExt, len(mi))
	for i := range m {
		m[i] = make([]*field.PrimeExt, len(mi[i]))
		for j := range m[i] {
			m[i][j] = k.NewZero().SetCoeffs(big.NewInt(int64(mi[i][j])))
		}
	}
	return m
}

func invMat(x [][]*field.PrimeExt) ([][]*field.PrimeExt, error) {
	k := x[0][0]
	n := len(x)
	xy := make([][]*field.PrimeExt, n)
	for i := range n {
		xy[i] = make([]*field.PrimeExt, 2*n)
		for j := range xy[i] {
			if j < n {
				xy[i][j] = k.NewZero().Set(x[i][j])
			} else {
				if j-n == i {
					xy[i][j] = k.NewOne()
				} else {
					xy[i][j] = k.NewZero()
				}
			}
		}
	}

	rre := nag.GaussElim(xy)
	one := k.NewOne()
	for i := range n {
		if !rre[i][i].Equal(one) {
			return nil, errors.Errorf("rre[%d] = %v want %v", i, rre[i][i], one)
		}
	}

	y := make([][]*field.PrimeExt, n)
	for i := range y {
		y[i] = make([]*field.PrimeExt, n)
		for j := range y[i] {
			y[i][j] = rre[i][n+j]
		}
	}
	return y, nil
}
