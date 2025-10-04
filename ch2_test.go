package learn

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"testing"

	"github.com/fumin/tensor"
)

func TestMeasurement_46(t *testing.T) {
	ψ := &Dense{
		Axis: []Axis{
			{System: 1, Braket: Ket},
			{System: 2, Braket: Ket}},
		D: tensor.T2([][]complex64{
			{1 / sqrt(3), 1. / 3},
			{1 / sqrt(3), sqrt(2) / 3}})}

	ψM := pdot(z0.H(), 1, ψ)
	ψM = ψM.Mul(1 / sqrt(dot(ψM.H(), ψM).D.At(0)))
	t.Logf("ψM %v %v", ψM.Axis, ψM.D.ToSlice1())
	amplitude := dot(sys(z1, 2).H(), ψM).D.At(0)
	prob := conj(amplitude) * amplitude
	if cmplx.Abs(complex128(prob-0.25)) > 1e-6 {
		t.Errorf("%v", prob)
	}

	// Do the same calculation in X basis.
	xZ0 := T1([]complex64{1 / math.Sqrt2, 1 / math.Sqrt2})
	xZ1 := T1([]complex64{1 / math.Sqrt2, -1 / math.Sqrt2})
	ψ = 𐌈(xZ0, sys(xZ0, 2)).Mul(1 / sqrt(3))
	ψ = ψ.Add(1./3, 𐌈(xZ0, sys(xZ1, 2)))
	ψ = ψ.Add(1./sqrt(3), 𐌈(xZ1, sys(xZ0, 2)))
	ψ = ψ.Add(math.Sqrt2/3, 𐌈(xZ1, sys(xZ1, 2)))

	ψM = pdot(xZ0.H(), 1, ψ)
	ψM = ψM.Mul(1 / sqrt(dot(ψM.H(), ψM).D.At(0)))
	t.Logf("ψM %v %v", ψM.Axis, ψM.D.ToSlice1())
	amplitude = dot(sys(xZ1, 2).H(), ψM).D.At(0)
	prob = conj(amplitude) * amplitude
	if cmplx.Abs(complex128(prob-0.25)) > 1e-6 {
		t.Errorf("%v", prob)
	}
}

func TestPhaseDamping_45(t *testing.T) {
	type System struct {
		alpha complex64
		beta  complex64
	}
	systems := make([]System, 0)
	for range 1 {
		c := randC(2)
		systems = append(systems, System{alpha: c[0], beta: c[1]})
	}
	env := sys(x0, "e")
	thetas := make([]complex64, 0)
	const nTheta = 32
	for i := range nTheta {
		thetas = append(thetas, complex(float32(i), 0)*2*π/nTheta)
	}
	basis := []*Dense{sys(z0, "e"), sys(z1, "e")}

	for _, s := range systems {
		system := sys(T1([]complex64{s.alpha, s.beta}), "s")
		ψ := 𐌈(system, system.H())
		for _, theta := range thetas {
			u := 𐌈(sys(σz, "s"), sys(σz, "e"))
			u = u.Mul(-1i * theta)
			// Exponential of U.
			diags := [][]int{
				{0, 0, 0, 0},
				{0, 0, 1, 1},
				{1, 1, 0, 0},
				{1, 1, 1, 1},
			}
			for _, at := range diags {
				v := u.D.At(at...)
				u.D.SetAt(at, exp(v))
			}

			// Kraus operators.
			ks := make([]*Dense, 0, len(basis))
			for _, b := range basis {
				k := pdot(b.H(), "e", u, "e", env)
				ks = append(ks, k)
			}

			// System after evolution.
			ρ := pdot(ks[0], "s", ψ, "s", ks[0].H())
			for _, k := range ks[1:] {
				ρ.Add(1, pdot(k, "s", ψ, "s", k.H()))
			}
			ρExpected := Copy(T1([]complex64{0}), ψ)
			ρExpected.D.SetAt([]int{0, 1}, cos(2*theta)*ρExpected.D.At(0, 1))
			ρExpected.D.SetAt([]int{1, 0}, cos(2*theta)*ρExpected.D.At(1, 0))
			if err := ρ.D.Equal(ρExpected.D, 1e-6); err != nil {
				t.Errorf("%+v", err)
			}
		}
	}
}

func TestGateRot_45(t *testing.T) {
	hRot := R(1/math.Sqrt2, 0, 1/math.Sqrt2, π)
	if err := hRot.D.Equal(h.D.Mul(-1i), 1e-6); err != nil {
		t.Errorf("%+v", err)
	}

	phase := T2([][]complex64{{1, 0}, {0, 1i}})
	phaseRot := R(0, 0, 1, π/2)
	if err := phaseRot.D.Equal(phase.D.Mul(1/math.Sqrt2*(1-1i)), 1e-6); err != nil {
		t.Errorf("%+v", err)
	}

	pi8 := T2([][]complex64{{1, 0}, {0, exp(1i * π / 4)}})
	pi8Rot := R(0, 0, 1, π/4)
	if err := pi8Rot.D.Equal(pi8.D.Mul(exp(-1i*π/8)), 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestPhaseKickback_44(t *testing.T) {
	ψI1 := 𐌈(z1, sys(x1, 2))
	ψI2 := 𐌈(x0, sys(x1, 2))
	ψF1 := dot(cnot, ψI1)
	ψF2 := dot(cnot, ψI2)
	t.Logf("ψF1 %v %v", ψF1.Axis, ψF1.D.ToSlice2())
	t.Logf("ψF2 %v %v", ψF2.Axis, ψF2.D.ToSlice2())
	if err := ψF1.D.Equal(𐌈(dot(σz, z1), sys(x1, 2)).D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
	if err := ψF2.D.Equal(𐌈(dot(σz, x0), sys(x1, 2)).D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestCnot21_44(t *testing.T) {
	cnot21 := dot(𐌈(h, sys(h, 2)), cnot, 𐌈(h, sys(h, 2)))
	cnot21Mat := toMat(cnot21)
	t.Logf("cnot21 %v %v", cnot21.Axis, cnot21Mat.D.ToSlice2())
	if err := cnot21Mat.D.Equal(tensor.T2([][]complex64{{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}}), 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestSwap_44(t *testing.T) {
	cnot21 := 𐌈(sys(z0, 2), sys(z0, 2).H(), one).Add(1, 𐌈(sys(z1, 2), sys(z1, 2).H(), σx))
	cnot21Mat := toMat(cnot21)
	t.Logf("cnot21 %v %v", cnot21.Axis, cnot21Mat.D.ToSlice2())
	if err := cnot21Mat.D.Equal(tensor.T2([][]complex64{{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}}), 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
	swap := dot(cnot, cnot21, cnot)
	swapMat := toMat(swap)
	t.Logf("swap %v %v", swap.Axis, swapMat.D.ToSlice2())
	if err := swapMat.D.Equal(tensor.T2([][]complex64{{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}), 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestKraus_31(t *testing.T) {
	// System.
	type System struct {
		alpha complex64
		beta  complex64
	}
	systems := []System{
		{alpha: 1 + 3i, beta: 2 - 5i},
	}
	for range 3 {
		c := randC(2)
		systems = append(systems, System{alpha: c[0], beta: c[1]})
	}
	// Environment.
	env := sys(z0, "e")
	// System environment interaction.
	u := sysReplace(cnot, 1, "s")
	u = sysReplace(u, 2, "e")

	// Basis.
	bases := [][]*Dense{
		{sys(z0, "e"), sys(z1, "e")},
		{sys(x0, "e"), sys(x1, "e")},
		{sys(y0, "e"), sys(y1, "e")},
		{sys(h0, "e"), sys(h1, "e")},
	}

	// Compute Kraus.
	for _, stm := range systems {
		ψ := sys(T1([]complex64{stm.alpha, stm.beta}), "s")
		ψ = 𐌈(ψ, ψ.H())
		ρ := T1([]complex64{0})
		for i, basis := range bases {
			var ks []*Dense
			for j, b := range basis {
				k := pdot(b.H(), "e", u, "e", env)
				ks = append(ks, k)
				t.Logf("k%d: %v %v", j, k.Axis, k.D.ToSlice2())
			}
			ρB := pdot(ks[0], "s", ψ, "s", ks[0].H())
			for _, k := range ks[1:] {
				ρB.Add(1, pdot(k, "s", ψ, "s", k.H()))
			}

			if i != 0 {
				if err := ρ.D.Equal(ρB.D, 4e-6); err != nil {
					panic(fmt.Sprintf("err %+v %d %v ρB %v", err, i, ρ.D.ToSlice2(), ρB.D.ToSlice2()))
				}
			}
			Copy(ρ, ρB)
		}
		if err := ρ.D.Equal(tensor.T2([][]complex64{{stm.alpha * conj(stm.alpha), 0}, {0, stm.beta * conj(stm.beta)}}), 2e-6); err != nil {
			panic(fmt.Sprintf("%+v", err))
		}
		t.Logf("ρ: %v %v", ρ.Axis, ρ.D.ToSlice2())
	}
}

func TestQGates_29(t *testing.T) {
	if err := toMat(cz).D.Equal(tensor.T2([][]complex64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}}), 0); err != nil {
		t.Errorf("%+v", err)
	}
	if err := toMat(cnot).D.Equal(tensor.T2([][]complex64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}), 0); err != nil {
		t.Errorf("%+v", err)
	}
	t.Logf("cz: %v %v", cz.Axis, toMat(cz).D.ToSlice2())
	t.Logf("cnot: %v %v", cnot.Axis, toMat(cnot).D.ToSlice2())
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	m.Run()
}
