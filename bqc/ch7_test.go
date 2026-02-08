package bqc

import (
	"bytes"
	"encoding/csv"
	"os"
	"strconv"
	"testing"

	"github.com/fumin/tensor"
)

func TestQuantumProcess_232(t *testing.T) {
	// The 3 degrees of freedom corresponding to unitary transformations, that were mentioned in the text refers to rotations in the Bloch sphere.
	// The other nine correspond to:
	// 3 for translation, 3 for scaling, 3 for shears.
	//
	// This means a quantum process maps a sphere to an ellipsoid.
	// See One-to-one parametrization of quantum channels, Akio Fujiwara and Paul Algoet, https://doi.org/10.1103/PhysRevA.59.3290.
}

func TestRabiOscillations_232(t *testing.T) {
	十ket := Copy(T1([]complex64{0}), z0).Add(1, z1).Mul(1 / sqrt(2))
	十 := dot(十ket, 十ket.H())
	t.Logf("十ket %v", 十ket.D.ToSlice1())
	t.Logf("十 %f %f %f", real(tr(dot(σx, 十))), real(tr(dot(σy, 十))), real(tr(dot(σz, 十))))
	prob := func(εf, Ωf, t2f, tf float64) float64 {
		ε, Ω, t2, tm := complex(float32(εf), 0), complex(float32(Ωf), 0), complex(float32(t2f), 0), complex(float32(tf*π), 0)
		ρ := Copy(T1([]complex64{0}), one).Add((1 - ε), σx).Mul(1. / 2)
		u := Exp(Copy(T1([]complex64{0}), σz).Mul(Ω).Mul(-1i * tm))
		kraus := []*Dense{
			Copy(T1([]complex64{0}), one).Mul(sqrt((1 + exp(-tm/(2*t2))) / 2)),
			Copy(T1([]complex64{0}), σz).Mul(sqrt((1 - exp(-tm/(2*t2))) / 2)),
		}
		// Check kraus is normalized.
		kk := Copy(T1([]complex64{0}), kraus[0])
		kk.D.Reset(kk.D.Shape()...)
		for _, a := range kraus {
			kk.Add(1, dot(a.H(), a))
		}
		if err := kk.D.Equal(tensor.Zeros(0).Eye(kk.D.Shape()[0], 0), 1e-6); err != nil {
			t.Errorf("%+v", err)
		}

		// Evolve state.
		ρt := dot(u.H(), ρ, u)
		ρtk := Copy(T1([]complex64{0}), ρt)
		ρtk.D.Reset(ρtk.D.Shape()...)
		for _, a := range kraus {
			ρtk.Add(1, dot(a, ρt, a.H()))
		}

		prob := tr(dot(十, ρtk))
		return float64(real(prob))
	}

	records := make([][]string, 0)
	records = append(records, []string{"eps", "omega", "t2", "t", "p"})
	const εStart, εEnd, εn = 0, 0.9, 3
	const ΩStart, ΩEnd, Ωn = 0.1, 1, 3
	const tStart, tEnd = 0, 16
	for εi := range εn {
		ε := εStart + float64(εi)/float64(εn-1)*(εEnd-εStart)
		for Ωi := range Ωn {
			Ω := ΩStart + float64(Ωi)/float64(Ωn-1)*(ΩEnd-ΩStart)
			for _, t2 := range []float64{5, 10, 20, 1000} {
				tn := int(Ω * float64(1000))
				for ti := range tn {
					t := tStart + float64(ti)/float64(tn-1)*(tEnd-tStart)
					p := prob(ε, Ω, t2, t)
					line := []string{
						strconv.FormatFloat(ε, 'f', -1, 64),
						strconv.FormatFloat(Ω, 'f', -1, 64),
						strconv.FormatFloat(t2, 'f', -1, 64),
						strconv.FormatFloat(t, 'f', -1, 64),
						strconv.FormatFloat(p, 'f', -1, 64),
					}
					records = append(records, line)
				}
			}
		}
	}

	buf := bytes.NewBuffer(nil)
	w := csv.NewWriter(buf)
	w.WriteAll(records)
	if err := w.Error(); err != nil {
		t.Errorf("%+v", err)
	}
	fpath := "/dev/null"
	if err := os.WriteFile(fpath, buf.Bytes(), 0644); err != nil {
		t.Errorf("%+v", err)
	}
}

func tr(a *Dense) complex64 {
	var tr complex64 = 0
	for i := range a.D.Shape()[0] {
		tr += a.D.At(i, i)
	}
	return tr
}
