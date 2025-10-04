package learn

import (
	"bytes"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"strconv"
	"testing"

	"github.com/fumin/diffeq"
	"github.com/fumin/tensor"
)

func TestInitialStateNMR_82(t *testing.T) {
	hwkt := ℏ * 500 * 1e6 / (k * 300)
	t.Logf("hwkt %f", hwkt)

	hkt := Copy(T1([]complex64{0}), σz).Mul(-1. / 2 * complex(float32(hwkt), 0))
	exph := Copy(T1([]complex64{0}), hkt)
	for i := range 2 {
		ax := []int{i, i}
		v := exph.D.At(ax...)
		exph.D.SetAt(ax, exp(-v))
	}
	z := exph.D.At(0, 0) + exph.D.At(1, 1)
	rho := exph.Mul(1. / z)
	t.Logf("rho %v", rho.D.ToSlice2())

	rhoH := Copy(T1([]complex64{0}), one).Add(-1, hkt).Mul(1. / 2)
	if err := rhoH.D.Equal(rho.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestConjugatingZRot_81(t *testing.T) {
	// Verify equation 3.32.
	u := 𐌈(R(0, 0, 1, π/2), sys(𐌈(z0.H(), z0), 1))
	u = u.Add(1, 𐌈(R(0, 0, 1, -π/2), sys(𐌈(z1.H(), z1), 1)))
	t.Logf("u %v %v", u.Axis, toMat(u).D.ToSlice2())
	R1 := func(x, y, z, theta complex64) *Dense { return 𐌈(R(x, y, z, theta), sys(one, 1)) }
	R2 := func(x, y, z, theta complex64) *Dense { return 𐌈(one, sys(R(x, y, z, theta), 1)) }
	rhs := Copy(T1([]complex64{0}), cnot).Mul(exp(-1i * π / 4))
	lhs := dot(R1(0, 0, 1, π/2), R2(1, 0, 0, π/2), R2(0, 1, 0, -π/2), u, R2(0, 1, 0, π/2)).Transpose(rhs.Axis)
	if err := lhs.D.Equal(rhs.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}

	// c3_72 is the circuit in equation 3.72.
	c3_72 := dot(R2(0, 0, 1, π/2), R2(0, 1, 0, π/2), R1(0, 0, 1, π/2), u)
	// Similar to equation 3.23:
	// Rz(θ)Ry(π/2) = Rn(π/2)Rz(θ), where n = cos(-θ)y + sin(-θ)x.
	zBeginning := dot(R2(1, 0, 0, -π/2), u, R2(0, 0, 1, π/2), R1(0, 0, 1, π/2)).Transpose(c3_72.Axis)
	if err := zBeginning.D.Equal(c3_72.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestRWA_81(t *testing.T) {
	records := [][]string{{"omega", "t", "mx", "my", "mz"}}
	omegas := []float64{10, 20, 40}
	for _, omega := range omegas {
		dydx := func(dmdt []float64, t float64, m []float64) {
			const x, y, z = 0, 1, 2
			dmdt[x] = omega * m[y]
			dmdt[y] = 2*math.Cos(omega*t)*m[z] - omega*m[x]
			dmdt[z] = -2 * math.Cos(omega*t) * m[y]
		}
		xspan := [2]float64{0, 2}
		y0 := []float64{0, 0, 1}
		xs, ys, err := diffeq.DormandPrince(dydx, xspan, y0)
		if err != nil {
			t.Fatalf("%+v", err)
		}

		// Collect results.
		for i, x := range xs {
			y := ys[i]

			line := []string{
				strconv.FormatFloat(omega, 'f', -1, 64),
				strconv.FormatFloat(x, 'f', -1, 64),
				strconv.FormatFloat(y[0], 'f', -1, 64),
				strconv.FormatFloat(y[1], 'f', -1, 64),
				strconv.FormatFloat(y[2], 'f', -1, 64),
			}
			records = append(records, line)
		}
	}

	fname := "/dev/null"
	buf := bytes.NewBuffer(nil)
	w := csv.NewWriter(buf)
	w.WriteAll(records)
	if err := w.Error(); err != nil {
		t.Fatalf("%+v", err)
	}
	if err := os.WriteFile(fname, buf.Bytes(), 0644); err != nil {
		t.Fatalf("%+v", err)
	}
}

func TestCompositePulses_80(t *testing.T) {
	psi0 := z0
	psi := dot(R(0, 1, 0, π), psi0)
	var epsN, epsD int = 16, 64
	records := [][]string{{"epsilon", "Fa", "Fb"}}
	for i := -epsN; i <= epsN; i++ {
		eps := complex(float32(i)*π/float32(epsD), 0)
		psiA := dot(R(0, 1, 0, π/2+eps), R(0, 1, 0, π/2+eps), psi0)
		fad := dot(psiA.H(), psi).D.At(0)
		fa := real(conj(fad) * fad)
		psiB := dot(R(0, 1, 0, π/2+eps), R(1, 0, 0, π+eps), R(0, 1, 0, π/2+eps), psi0)
		fbd := dot(psiB.H(), psi).D.At(0)
		fb := real(conj(fbd) * fbd)

		l := []string{
			strconv.FormatFloat(float64(real(eps)), 'f', -1, 64),
			strconv.FormatFloat(float64(fa), 'f', -1, 64),
			strconv.FormatFloat(float64(fb), 'f', -1, 64),
		}
		records = append(records, l)
	}
	buf := bytes.NewBuffer(nil)
	w := csv.NewWriter(buf)
	w.WriteAll(records)
	if err := w.Error(); err != nil {
		t.Errorf("%+v", err)
	}
	fname := "/dev/null"
	if err := os.WriteFile(fname, buf.Bytes(), 0644); err != nil {
		t.Errorf("%+v", err)
	}

	want := R(0, 1, 0, π)
	t.Logf("want %v", want.D.ToSlice2())
	pulse := dot(R(0, 1, 0, π/2), R(1, 0, 0, π), R(0, 1, 0, π/2), R(0, 0, 1, -π))
	t.Logf("pulse %v", pulse.D.ToSlice2())
	if err := pulse.D.Equal(want.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestMeasurement_46(t *testing.T) {
	psi := &Dense{
		Axis: []Axis{
			{System: 0, Braket: Ket},
			{System: 1, Braket: Ket}},
		D: tensor.T2([][]complex64{
			{1 / sqrt(3), 1. / 3},
			{1 / sqrt(3), sqrt(2) / 3}})}

	psiM := pdot(z0.H(), 0, psi)
	psiM = psiM.Mul(1 / sqrt(dot(psiM.H(), psiM).D.At(0)))
	t.Logf("psiM %v %v", psiM.Axis, psiM.D.ToSlice1())
	amplitude := dot(sys(z1, 1).H(), psiM).D.At(0)
	prob := conj(amplitude) * amplitude
	if cmplx.Abs(complex128(prob-0.25)) > 1e-6 {
		t.Errorf("%v", prob)
	}

	// Do the same calculation in X basis.
	xZ0 := T1([]complex64{1 / math.Sqrt2, 1 / math.Sqrt2})
	xZ1 := T1([]complex64{1 / math.Sqrt2, -1 / math.Sqrt2})
	psi = 𐌈(xZ0, sys(xZ0, 1)).Mul(1 / sqrt(3))
	psi = psi.Add(1./3, 𐌈(xZ0, sys(xZ1, 1)))
	psi = psi.Add(1./sqrt(3), 𐌈(xZ1, sys(xZ0, 1)))
	psi = psi.Add(math.Sqrt2/3, 𐌈(xZ1, sys(xZ1, 1)))

	psiM = pdot(xZ0.H(), 0, psi)
	psiM = psiM.Mul(1 / sqrt(dot(psiM.H(), psiM).D.At(0)))
	t.Logf("psiM %v %v", psiM.Axis, psiM.D.ToSlice1())
	amplitude = dot(sys(xZ1, 1).H(), psiM).D.At(0)
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
		psi := 𐌈(system, system.H())
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
			rho := pdot(ks[0], "s", psi, "s", ks[0].H())
			for _, k := range ks[1:] {
				rho.Add(1, pdot(k, "s", psi, "s", k.H()))
			}
			rhoExpected := Copy(T1([]complex64{0}), psi)
			rhoExpected.D.SetAt([]int{0, 1}, cos(2*theta)*rhoExpected.D.At(0, 1))
			rhoExpected.D.SetAt([]int{1, 0}, cos(2*theta)*rhoExpected.D.At(1, 0))
			if err := rho.D.Equal(rhoExpected.D, 1e-6); err != nil {
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
	psiI1 := 𐌈(z1, sys(x1, 1))
	psiI2 := 𐌈(x0, sys(x1, 1))
	psiF1 := dot(cnot, psiI1)
	psiF2 := dot(cnot, psiI2)
	t.Logf("psiF1 %v %v", psiF1.Axis, psiF1.D.ToSlice2())
	t.Logf("psiF2 %v %v", psiF2.Axis, psiF2.D.ToSlice2())
	if err := psiF1.D.Equal(𐌈(dot(σz, z1), sys(x1, 1)).D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
	if err := psiF2.D.Equal(𐌈(dot(σz, x0), sys(x1, 1)).D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestCnot21_44(t *testing.T) {
	cnot21 := dot(𐌈(h, sys(h, 1)), cnot, 𐌈(h, sys(h, 1)))
	cnot21Mat := toMat(cnot21)
	t.Logf("cnot21 %v %v", cnot21.Axis, cnot21Mat.D.ToSlice2())
	if err := cnot21Mat.D.Equal(tensor.T2([][]complex64{{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}}), 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestSwap_44(t *testing.T) {
	cnot21 := 𐌈(sys(z0, 1), sys(z0, 1).H(), one).Add(1, 𐌈(sys(z1, 1), sys(z1, 1).H(), σx))
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
	u := sysReplace(cnot, 0, "s")
	u = sysReplace(u, 1, "e")

	// Basis.
	bases := [][]*Dense{
		{sys(z0, "e"), sys(z1, "e")},
		{sys(x0, "e"), sys(x1, "e")},
		{sys(y0, "e"), sys(y1, "e")},
		{sys(h0, "e"), sys(h1, "e")},
	}

	// Compute Kraus.
	for _, stm := range systems {
		psi := sys(T1([]complex64{stm.alpha, stm.beta}), "s")
		psi = 𐌈(psi, psi.H())
		rho := T1([]complex64{0})
		for i, basis := range bases {
			var ks []*Dense
			for j, b := range basis {
				k := pdot(b.H(), "e", u, "e", env)
				ks = append(ks, k)
				t.Logf("k%d: %v %v", j, k.Axis, k.D.ToSlice2())
			}
			rhoB := pdot(ks[0], "s", psi, "s", ks[0].H())
			for _, k := range ks[1:] {
				rhoB.Add(1, pdot(k, "s", psi, "s", k.H()))
			}

			if i != 0 {
				if err := rho.D.Equal(rhoB.D, 4e-6); err != nil {
					panic(fmt.Sprintf("err %+v %d %v rhoB %v", err, i, rho.D.ToSlice2(), rhoB.D.ToSlice2()))
				}
			}
			Copy(rho, rhoB)
		}
		if err := rho.D.Equal(tensor.T2([][]complex64{{stm.alpha * conj(stm.alpha), 0}, {0, stm.beta * conj(stm.beta)}}), 2e-6); err != nil {
			panic(fmt.Sprintf("%+v", err))
		}
		t.Logf("rho: %v %v", rho.Axis, rho.D.ToSlice2())
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
