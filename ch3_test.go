package learn

import (
	"bytes"
	"encoding/csv"
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/fumin/diffeq"
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
	ρ := exph.Mul(1. / z)
	t.Logf("ρ %v", ρ.D.ToSlice2())

	ρH := Copy(T1([]complex64{0}), one).Add(-1, hkt).Mul(1. / 2)
	if err := ρH.D.Equal(ρ.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestConjugatingZRot_81(t *testing.T) {
	// Verify equation 3.32.
	u := 𐌈(R(0, 0, 1, π/2), sys(𐌈(z0.H(), z0), 2))
	u = u.Add(1, 𐌈(R(0, 0, 1, -π/2), sys(𐌈(z1.H(), z1), 2)))
	t.Logf("u %v %v", u.Axis, toMat(u).D.ToSlice2())
	R1 := func(x, y, z, theta complex64) *Dense { return 𐌈(R(x, y, z, theta), sys(one, 2)) }
	R2 := func(x, y, z, theta complex64) *Dense { return 𐌈(one, sys(R(x, y, z, theta), 2)) }
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
	ψ0 := z0
	ψ := dot(R(0, 1, 0, π), ψ0)
	var epsN, epsD int = 16, 64
	records := [][]string{{"eψlon", "Fa", "Fb"}}
	for i := -epsN; i <= epsN; i++ {
		eps := complex(float32(i)*π/float32(epsD), 0)
		ψA := dot(R(0, 1, 0, π/2+eps), R(0, 1, 0, π/2+eps), ψ0)
		fad := dot(ψA.H(), ψ).D.At(0)
		fa := real(conj(fad) * fad)
		ψB := dot(R(0, 1, 0, π/2+eps), R(1, 0, 0, π+eps), R(0, 1, 0, π/2+eps), ψ0)
		fbd := dot(ψB.H(), ψ).D.At(0)
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
