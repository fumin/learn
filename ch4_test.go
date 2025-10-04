package learn

import (
	"fmt"
	"math/cmplx"
	"slices"
	"testing"

	"github.com/fumin/tensor"
)

func TestBeamSplitter_121(t *testing.T) {
	var θ complex64 = π / 4
	beamSplitter := T2([][]complex64{
		{cos(θ), -sin(θ)},
		{sin(θ), cos(θ)},
	})
	photons := make([]*Dense, 0)
	photons = append(photons, sys(T1([]complex64{1, 0}).H(), len(photons)+1))
	photons = append(photons, sys(T1([]complex64{0, 1}).H(), len(photons)+1))
	pstate := 𐌈(photons...)

	// Apply beam splitter.
	for _, ax := range pstate.Axis {
		pstate = dot(pstate, sys(beamSplitter, ax.System))
	}

	// Check result for question (a).
	state, ops := toFock(pstate)
	for i := range ops {
		var want complex64
		switch {
		case slices.Equal([]int{2, 0}, ops[i]):
			want = 1. / sqrt(2)
		case slices.Equal([]int{1, 1}, ops[i]):
			want = 0
		case slices.Equal([]int{0, 2}, ops[i]):
			want = -1. / sqrt(2)
		}
		if cmplx.Abs(complex128(state.D.At(i)-want)) > 1e-6 {
			t.Errorf("%v %v %v", ops[i], state.D.At(i), want)
		}
	}

	photons = photons[:0]
	photons = append(photons, sys(T1([]complex64{1, 0}).H(), len(photons)+1))
	photons = append(photons, sys(T1([]complex64{1, 0}).H(), len(photons)+1))
	pstate = 𐌈(photons...).Mul(1. / sqrt(2))
	for _, ax := range pstate.Axis {
		pstate = dot(pstate, sys(beamSplitter, ax.System))
	}
	// Check result for question (b).
	state, ops = toFock(pstate)
	for i := range ops {
		var want complex64
		switch {
		case slices.Equal([]int{2, 0}, ops[i]):
			want = 1. / 2
		case slices.Equal([]int{1, 1}, ops[i]):
			want = -1. / sqrt(2)
		case slices.Equal([]int{0, 2}, ops[i]):
			want = 1. / 2
		}
		if cmplx.Abs(complex128(state.D.At(i)-want)) > 1e-6 {
			t.Errorf("%v %v %v", ops[i], state.D.At(i), want)
		}
	}

	// a0H a1V
	beamSplitter1 := T2([][]complex64{
		{cos(θ), 0, -sin(θ), 0},
		{0, cos(θ), 0, -sin(θ)},
		{sin(θ), 0, cos(θ), 0},
		{0, sin(θ), 0, cos(θ)},
	})
	photons = photons[:0]
	photons = append(photons, sys(T1([]complex64{1, 0, 0, 0}).H(), len(photons)+1))
	photons = append(photons, sys(T1([]complex64{0, 0, 0, 1}).H(), len(photons)+1))
	pstate = 𐌈(photons...)
	for _, ax := range pstate.Axis {
		pstate = dot(pstate, sys(beamSplitter1, ax.System))
	}
	state0, ops := toFock(pstate)

	// a0V a1H
	photons = photons[:0]
	photons = append(photons, sys(T1([]complex64{0, 1, 0, 0}).H(), len(photons)+1))
	photons = append(photons, sys(T1([]complex64{0, 0, 1, 0}).H(), len(photons)+1))
	pstate = 𐌈(photons...)
	for _, ax := range pstate.Axis {
		pstate = dot(pstate, sys(beamSplitter1, ax.System))
	}
	state1, ops := toFock(pstate)

	ΨPlus := Copy(T1([]complex64{0}), state0).Mul(1./sqrt(2)).Add(1./sqrt(2), Copy(T1([]complex64{0}), state1))
	ΨMinus := Copy(T1([]complex64{0}), state0).Mul(1./sqrt(2)).Add(-1./sqrt(2), Copy(T1([]complex64{0}), state1))
	// Check ΨPlus.
	for i := range ops {
		var want complex64
		switch {
		case slices.Equal([]int{1, 1, 0, 0}, ops[i]):
			want = 1. / sqrt(2)
		case slices.Equal([]int{0, 0, 1, 1}, ops[i]):
			want = -1. / sqrt(2)
		case slices.Equal([]int{0, 2}, ops[i]):
			want = 0
		}
		if cmplx.Abs(complex128(ΨPlus.D.At(i)-want)) > 1e-6 {
			t.Errorf("%v %v %v", ops[i], ΨPlus.D.At(i), want)
		}
	}
	// Check ΨMinus.
	for i := range ops {
		var want complex64
		switch {
		case slices.Equal([]int{1, 0, 0, 1}, ops[i]):
			want = 1. / sqrt(2)
		case slices.Equal([]int{0, 1, 1, 0}, ops[i]):
			want = -1. / sqrt(2)
		case slices.Equal([]int{0, 2}, ops[i]):
			want = 0
		}
		if cmplx.Abs(complex128(ΨMinus.D.At(i)-want)) > 1e-6 {
			t.Errorf("%v %v %v", ops[i], ΨMinus.D.At(i), want)
		}
	}
	t.Logf("%v", ops)
	t.Logf("Ψ+ %v", ΨPlus.D.ToSlice1())
	t.Logf("Ψ- %v", ΨMinus.D.ToSlice1())
}

// TestFig4_10_110 proves that the following circuits are equivalent:
// .===========‖
// .           ‖
// .========‖  ‖
// .   __   ‖  ‖
// .--|  |--X--Z---------
// .  |CZ|
// .--|__|---------------
// .
// .
// .===========‖
// .           ‖
// .========‖  ‖
// .        ‖  ‖   __
// .--------X--Z--|  |---
// .        ‖     |CZ|
// .--------Z-----|__|---
func TestFig4_10_110(t *testing.T) {
	circuit1 := func(bits [2]bool) *Dense {
		u := 𐌈(one, sys(one, 2))
		u = dot(u, cz)
		if bits[1] {
			u = dot(u, σx)
		}
		if bits[0] {
			u = dot(u, σz)
		}
		u = u.Transpose(cz.Axis)
		return u
	}
	circuit2 := func(bits [2]bool) *Dense {
		u := 𐌈(one, sys(one, 2))
		if bits[1] {
			u = dot(u, σx)
			u = dot(u, sys(σz, 2))
		}
		if bits[0] {
			u = dot(u, σz)
		}
		u = dot(u, cz)
		u = u.Transpose(cz.Axis)
		return u
	}
	classicalBits := [][2]bool{{false, false}, {false, true}, {true, false}, {true, true}}
	for _, bits := range classicalBits {
		u1 := circuit1(bits)
		u2 := circuit2(bits)
		if err := u1.D.Equal(u2.D, 1e-6); err != nil {
			t.Errorf("%v %v", bits, err)
		}
	}
}

func TestTeleportation_108(t *testing.T) {
	ψ1s := make([]*Dense, 0)
	for range 16 {
		ψ1s = append(ψ1s, T1(randC(2)))
	}
	for _, ψ1 := range ψ1s {
		b0023 := sysReplace(sysReplace(b00, 2, 3), 1, 2)
		state := 𐌈(ψ1, b0023)

		gates := []*Dense{one, σz, σx, dot(σz, σx)}
		for k, bij := range []*Dense{b00, b01, b10, b11} {
			gate := Copy(T1([]complex64{0}), gates[k])

			// Measure and project to bij.
			ψ3 := pdot(bij.H(), []int{1, 2}, state)
			ψ3 = ψ3.Mul(1. / sqrt(dot(ψ3.H(), ψ3).D.At(0)))

			// Apply gate on ψ3.
			ψ3 = dot(sys(gate, 3), ψ3)
			if err := ψ3.D.Equal(ψ1.D, 1e-6); err != nil {
				t.Errorf("bell%d %+v", k, err)
			}
		}
	}
}

func TestEq4_61_106(t *testing.T) {
	const mode1, mode2, mode3 = 0, 1, 2
	ns := T2([][]complex64{
		{1 - sqrt(2), pow(2, -1./4), sqrt(3./sqrt(2) - 2)},
		{pow(2, -1./4), 1. / 2, (1 - sqrt(2)) / 2},
		{sqrt(3./sqrt(2) - 2), (1 - sqrt(2)) / 2, sqrt(2) - 1./2}})

	const maxN = 2
	outAmp := make([]complex64, maxN+1)
	for n := range maxN + 1 {
		// Add n mode1 photons.
		photons := make([]*Dense, 0, n)
		for range n {
			p := sys(T1([]complex64{0, 0, 0}).H(), len(photons)+1)
			p.D.SetAt([]int{mode1}, 1)
			photons = append(photons, p)
		}
		// Add 1 mode2 photon.
		p := sys(T1([]complex64{0, 0, 0}).H(), len(photons)+1)
		p.D.SetAt([]int{mode2}, 1)
		photons = append(photons, p)
		// Prepare input state and track operator counts.
		state := 𐌈(photons...)
		oc := operatorCount(len(photons), p.D.Shape()[0])

		// Apply gate to each photon.
		// Note that gates assume input operators are bras, <in|U = <out|.
		for _, ax := range state.Axis {
			state = dot(state, sys(ns, ax.System))
		}

		// Collect amplitudes when there is 1 mode2 photon and 0 mode3 photons.
		for i, amp := range state.D.All() {
			if opAt(oc, i, mode2) == 1 && opAt(oc, i, mode3) == 0 {
				n := opAt(oc, i, mode1)
				outAmp[n] += amp
			}
		}
	}

	// The expected result is |2> is sign shifted, and the amplitudes of all |0>, |1>, and |2> are halved.
	// This means the success probability is 1/4.
	outAmpTrue := []complex64{1. / 2, 1. / 2, -1. / 2}
	for i := range outAmp {
		if cmplx.Abs(complex128(outAmp[i]-outAmpTrue[i])) > 1e-6 {
			t.Errorf("%d %v %v", i, outAmp[i], outAmpTrue[i])
		}
	}
}

func TestFig4_7_106(t *testing.T) {
	const mode1, mode2, mode3 = 0, 1, 2
	one := func() *Dense {
		I := T2([][]complex64{{0}})
		I.D.Eye(3, 0)
		return I
	}
	phase := func(m int, ϕ complex64) *Dense {
		u := one()
		u.D.SetAt([]int{m, m}, exp(1i*ϕ))
		return u
	}
	beam := func(m0, m1 int, cosθ, sinθ complex64) *Dense {
		u := one()
		u.D.SetAt([]int{m0, m0}, cosθ)
		u.D.SetAt([]int{m0, m1}, -sinθ)
		u.D.SetAt([]int{m1, m0}, sinθ)
		u.D.SetAt([]int{m1, m1}, cosθ)
		return u
	}
	elements := []*Dense{
		beam(mode2, mode3, 1./2*sqrt(2+sqrt(2)), 1./2*sqrt(2-sqrt(2))),
		phase(mode1, π),
		beam(mode1, mode2, sqrt(2)-1, sqrt(2*(sqrt(2)-1))),
		beam(mode2, mode3, 1./2*sqrt(2+sqrt(2)), -1./2*sqrt(2-sqrt(2))),
	}
	// Multiply from left to right, since gate matrices assume input operators are bras.
	// Using the beam splitter as an example:
	// U(a_{1, in}) = <1, 0|U = <1, 0|{{cos, -sin}, {sin, cos}} = <cos, -sin| = cos*a_{1, out} - sin*{2, out}.
	ns := dot(append([]*Dense{one()}, elements...)...)

	nsTrue := T2([][]complex64{{1 - sqrt(2), pow(2, -1./4), sqrt(3./sqrt(2) - 2)}, {pow(2, -1./4), 1. / 2, (1 - sqrt(2)) / 2}, {sqrt(3./sqrt(2) - 2), (1 - sqrt(2)) / 2, sqrt(2) - 1./2}})
	if err := ns.D.Equal(nsTrue.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func toFock(pstate *Dense) (*Dense, [][]int) {
	numSys := len(pstate.D.Shape())
	modes := pstate.D.Shape()[0]
	oc := operatorCount(numSys, modes)
	fockOps, fockM := opUniq(oc)

	// Sum amplitudes for each fock basis.
	state := make([]complex64, len(fockOps))
	basis := make([]int, 0, modes)
	for i, v := range pstate.D.All() {
		basis = basis[:0]
		for m := range modes {
			basis = append(basis, opAt(oc, i, m))
		}
		state[fockM(basis)] += v
	}

	// Multiply by sqrt(n) since a*|n> = sqrt(n+1)|n+1>.
	for i := range state {
		var denom complex64 = 1
		for _, opN := range fockOps[i] {
			for j := range opN {
				denom *= (complex(float32(j), 0) + 1)
			}
		}
		state[i] *= sqrt(denom)
	}

	return T1(state), fockOps
}

// operatorCount tracks the number of operators for each mode in the product space.
// For example, let there be 3 modes, and a_{i} the creation operator of the i'th mode.
// The product state of two systems is:
// (c11*a1+c12*a2+c13*a3)(c21*a1+c22*a2+c23*a3)
// = c11*c21*a1*a1 + c11*c22*a1*a2 + c11*c23*a1*a3 + c12*c21*a2*a1 + ... + c13*c23*a3*a3
// We track the operator counts of each mode:
// c11*c21: {2, 0, 0} meaning 2 a0's, 0 a1's and 0 a2's
// c11*c22: {1, 1, 0} meaning 1 a0's, 1 a1's and 0 a2's
// ...
// c13*c23: {0, 0, 2} meaning 0 a0's, 0 a1's and 2 a3's
func operatorCount(systemNum, modeNum int) *tensor.Dense {
	opCnt := make([]*tensor.Dense, 0, systemNum)
	for range systemNum {
		oc := tensor.Zeros(1).Eye(modeNum, 0)
		opCnt = append(opCnt, oc)
	}
	oc := op𐌈(opCnt...)
	return oc
}

func op𐌈(as ...*tensor.Dense) *tensor.Dense {
	buf0, buf1 := tensor.Zeros(1), tensor.Zeros(1)
	buf0.Reset(as[0].Shape()...).Set(nil, as[0])
	for _, a := range as[1:] {
		op𐌈2(buf1, buf0, a)
		buf0, buf1 = buf1, buf0
	}
	return buf0
}

func op𐌈2(c, a, b *tensor.Dense) *tensor.Dense {
	cShape := make([]int, 0, len(a.Shape())+len(b.Shape())-1)
	cShape = append(cShape, a.Shape()[:len(a.Shape())-1]...)
	cShape = append(cShape, b.Shape()[:len(b.Shape())-1]...)
	cShape = append(cShape, a.Shape()[len(a.Shape())-1])
	c.Reset(cShape...)

	abd := make([][2]int, len(a.Shape()))
	abd[len(abd)-1] = [2]int{0, 1}
	bbd := make([][2]int, len(b.Shape()))
	bbd[len(bbd)-1] = [2]int{0, 1}

	ik := make([]int, len(a.Shape()))
	jk := make([]int, len(b.Shape()))
	ijk := make([]int, len(cShape))
	for i := range a.Slice(abd).All() {
		for j := range b.Slice(bbd).All() {
			for k := range c.Shape()[1] {
				ik = ik[:0]
				ik = append(ik, i[:len(i)-1]...)
				ik = append(ik, k)

				jk = jk[:0]
				jk = append(jk, j[:len(j)-1]...)
				jk = append(jk, k)

				ijk = ijk[:0]
				ijk = append(ijk, i[:len(i)-1]...)
				ijk = append(ijk, j[:len(j)-1]...)
				ijk = append(ijk, k)

				v := a.At(ik...) + b.At(jk...)
				c.SetAt(ijk, v)
			}
		}
	}
	return c
}

func opAt(counts *tensor.Dense, at []int, mode int) int {
	return int(real(counts.At(append(at, mode)...)))
}

func opUniq(counts *tensor.Dense) ([][]int, func([]int) int) {
	modes := counts.Shape()[len(counts.Shape())-1]
	at := make([]int, len(counts.Shape())-1)
	uniqs := make([][]int, 0)
	m := make(map[string]int)
	for i := range counts.All() {
		if i[len(i)-1] != 0 {
			continue
		}
		copy(at, i)

		ops := make([]int, 0, modes)
		for mode := range modes {
			ops = append(ops, opAt(counts, at, mode))
		}

		k := fmt.Sprintf("%v", ops)
		if _, ok := m[k]; ok {
			continue
		}
		uniqs = append(uniqs, ops)
		m[k] = len(uniqs) - 1
	}

	mFunc := func(ops []int) int {
		k := fmt.Sprintf("%v", ops)
		i, ok := m[k]
		if !ok {
			return -1
		}
		return i
	}
	return uniqs, mFunc
}
