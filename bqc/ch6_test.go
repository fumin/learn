package bqc

import (
	"bytes"
	"encoding/csv"
	"math"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/fumin/tensor"
	"github.com/pkg/errors"
	"gonum.org/v1/exp/root"
	"gonum.org/v1/gonum/optimize"
)

func TestNativeSuperconducting_216(t *testing.T) {
	iswap := &Dense{Axis: cnot.Axis, D: tensor.T4([][][][]complex64{
		{{{1, 0}, {0, 0}}, {{0, 0}, {-1i, 0}}},
		{{{0, -1i}, {0, 0}}, {{0, 0}, {0, 1}}},
	})}
	t.Logf("iswap %v", toMat(iswap).D.ToSlice2())
	cnotf := func(x []float64) *Dense {
		a, b, c, d := complex(float32(x[0]*π), 0), complex(float32(x[1]*π), 0), complex(float32(x[2]*π), 0), complex(float32(x[3]*π), 0)
		gates := []*Dense{
			𐌈(one, sys(R(1, 0, 0, a), 2)),
			𐌈(R(0, 0, 1, b), sys(R(0, 0, 1, c), 2)),
			iswap,
			𐌈(R(1, 0, 0, π/2), sys(one, 2)),
			iswap,
			𐌈(one, sys(R(0, 0, 1, d), 2)),
		}
		slices.Reverse(gates)
		return dot(gates...).Transpose(cnot.Axis)
	}
	problem := optimize.Problem{
		Func: func(x []float64) float64 {
			cnotV := cnotf(x)
			phase := complex64(complex(x[4], x[5]))
			return float64(cnotV.Mul(phase).Add(-1, cnot).D.FrobeniusNorm())
		},
	}
	result, err := optimize.Minimize(problem, []float64{0, 0, 0, 0, 0, 0}, nil, nil)
	if err != nil {
		t.Errorf("%+v", err)
	}
	t.Logf("result %v diff %f", result.X, result.F)
	cnotiswap := cnotf(result.X).Mul(complex64(complex(result.X[4], result.X[5])))
	t.Logf("cnotiswap %v", toMat(cnotiswap).D.ToSlice2())

	ψ0 := 𐌈(z0, sys(z0, 2), sys(z0, 3))
	gates := []*Dense{
		𐌈(h, sys(one, 2), sys(one, 3)),
		𐌈(cnot, sys(one, 3)),
		𐌈(sysReplace(cnot, 2, 3), sys(one, 2)),
	}
	slices.Reverse(gates)
	ψ := dot(append(gates, ψ0)...).Transpose(ψ0.Axis)
	ghz := Copy(T1([]complex64{0}), ψ0)
	ghz.D.SetAt([]int{0, 0, 0}, 1/sqrt(2))
	ghz.D.SetAt([]int{1, 1, 1}, 1/sqrt(2))
	t.Logf("ψ %v", Copy(T1([]complex64{0}), ψ).D.Reshape(-1).ToSlice1())
	if err := ψ.D.Equal(ghz.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}

	// Question (b)
	cphase := 𐌈(one, sys(one, 2))
	cphase.D.SetAt([]int{1, 1, 1, 1}, exp(1i*π))
	t.Logf("cphase %v", toMat(cphase).D.ToSlice2())
	ψ0 = 𐌈(z0, sys(z0, 2))
	gates = []*Dense{
		𐌈(R(0, 1, 0, π/2), sys(R(0, 1, 0, -π/2), 2)),
		cphase,
		𐌈(one, sys(R(0, 1, 0, π/2), 2)),
	}
	slices.Reverse(gates)
	ψ = dot(append(gates, ψ0)...).Transpose(ψ0.Axis)
	t.Logf("ψ %v", ψ.D.Reshape(-1).ToSlice1())
	if err := ψ.D.Equal(b00.D, 2e-6); err != nil {
		t.Errorf("%+v", err)
	}

	// Question (c)
	cr := func(θ complex64) *Dense {
		return &Dense{Axis: cnot.Axis, D: tensor.T4([][][][]complex64{
			{{{cos(θ / 2), -1i * sin(θ/2)}, {-1i * sin(θ/2), cos(θ / 2)}}, {{0, 0}, {0, 0}}},
			{{{0, 0}, {0, 0}}, {{cos(θ / 2), 1i * sin(θ/2)}, {1i * sin(θ/2), cos(θ / 2)}}},
		})}
	}
	cnotf = func(x []float64) *Dense {
		a, b, c := complex(float32(x[0]*π), 0), complex(float32(x[1]*π), 0), complex(float32(x[2]*π), 0)
		gates := []*Dense{
			𐌈(R(0, 0, 1, a), sys(R(1, 0, 0, b), 2)),
			cr(c),
		}
		slices.Reverse(gates)
		return dot(gates...).Transpose(cnot.Axis)
	}
	problem = optimize.Problem{
		Func: func(x []float64) float64 {
			cnotV := cnotf(x)
			phase := complex64(complex(x[3], x[4]))
			return float64(cnotV.Mul(phase).Add(-1, cnot).D.FrobeniusNorm())
		},
	}
	result, err = optimize.Minimize(problem, []float64{0.5, 0, 0, 0, 0}, nil, nil)
	if err != nil {
		t.Errorf("%+v", err)
	}
	t.Logf("result %v diff %f", result.X, result.F)
	cnotcr := cnotf(result.X).Mul(complex64(complex(result.X[3], result.X[4])))
	t.Logf("cnotcr %v", toMat(cnotcr).D.ToSlice2())
}

func TestTransmonSpectrum_214(t *testing.T) {
	// Question (a)
	// E_{n}^{(0)} = \sqrt{8E_{J}E_{C}}(n+\frac{1}{2}) \\
	// E_{n}^{(1)} = \langle n^{(0)}| -\frac{E_{C}}{12}(\hat{b}^{\dagger}+\hat{b})^{4} |n^{(0)}\rangle \\
	// \text{Select terms with the same number of } b^{\dagger} \text{and } b \text{, since the rest do not return } |n^{(0)}\rangle \\
	// E_{n}^{(1)} = -\frac{E_{C}}{12}\langle n^{(0)}| \left(bbb^{\dagger}b^{\dagger}+bb^{\dagger}bb^{\dagger}+b^{\dagger}bbb^{\dagger}+bb^{\dagger}b^{\dagger}b+b^{\dagger}bb^{\dagger}b+b^{\dagger}b^{\dagger}bb\right) |n^{(0)}\rangle \\
	// E_{n}^{(1)} = -\frac{E_{C}}{12}\langle n^{(0)}|\left( (b^{\dagger}b^{\dagger}bb+4b^{\dagger}b+2)+(b^{\dagger}b^{\dagger}bb+3b^{\dagger}b+1)+(b^{\dagger}b^{\dagger}bb+2b^{\dagger}b)+(b^{\dagger}b^{\dagger}bb+2b^{\dagger}b)+(b^{\dagger}b^{\dagger}bb+b^{\dagger}b)+(b^{\dagger}b^{\dagger}bb) \right)|n^{(0)}\rangle \\
	// E_{n}^{(1)} = -\frac{E_{C}}{12}\langle n^{(0)}| 6b^{\dagger}b^{\dagger}bb+12b^{\dagger}b+3 |n^{(0)}\rangle \\
	// E_{n}^{(1)} = -\frac{E_{C}}{12}(6n^{2}+6n+3) \\
	// E_{n}=\sqrt{8E_{J}E_{C}}(n+\frac{1}{2})-\frac{E_{C}}{12}(6n^{2}+6n+3)

	// Question (b)
	// E_{0}=\sqrt{2E_{J}E_{C}}-\frac{E_{C}}{4} \\
	// E_{1}=3\sqrt{2E_{J}E_{C}}-\frac{5E_{C}}{4} \\
	// E_{2}=5\sqrt{2E_{J}E_{C}}-\frac{13E_{C}}{4} \\
	// \omega_{01}=\omega_{q}=\frac{1}{\hbar}(\sqrt{8E_{J}E_{C}}-E_{C}) \\
	// \omega_{12}=\frac{1}{\hbar}(\sqrt{8E_{J}E_{C}}-2E_{C}) \\
	// \alpha=\omega_{12}-\omega_{01}=-\frac{E_{C}}{\hbar} \\
	// \frac{\alpha}{\omega_{q}}=\frac{-E_{C}}{\sqrt{8E_{J}E_{C}}-E_{C}}=\frac{\frac{E_C}{E_{J}}}{\frac{E_C}{E_{J}}-\sqrt{\frac{8E_C}{E_{J}}}}\simeq -\frac{1}{\sqrt{8}}\sqrt{\frac{E_{C}}{E_{J}}}
	const ej = 13
	const ec = 0.6
	ω01 := math.Sqrt(8*ej*ec) - ec
	if diff := math.Abs(ω01 - 7.3); diff > 1e-3 {
		t.Errorf("%f", diff)
	}
	// T is the decoherence time, and 1/α is the minimum gate time.
	const T = 1e-3
	α := ec * 1e9
	if !(1/α*1e5 < T) {
		t.Errorf("%f %f", α, T)
	}
}

func TestCircuitHamiltonian_214(t *testing.T) {
	// Question (a)
	// Note that the second equation in Eq. 6.59 misses a negative sign,
	// it should be n = i*nzpf*(b - b.H).
	// More explicitly, n = i*∂/∂x, which is negative of p = -i∂/∂x.
	// With ϕ = x, Eq. 6.61 becomes [ϕ, n] = -i.
	// Nonetheless, [b, bH] = 1 remains the same.
	// A further relation is [exp(iϕ), n] = [exp(ix), i*∂/∂x] = exp(iϕ), which leads to Eq. 6.54 exp(iϕ)|n> = |n-1>.
	//
	// The key is to keep the ng=Cg*Vg/(2e) term in Eq. 6.58.
	// In this case, Eq 6.58 has an extra term -8*Ec*ng*n.
	// This term becomes -8*(e*e/(2(Cg+Cs+Cj)))*(Cg*Vg/(2e))*n = 2e*(Cg/Cg+Cs+Cj) * Vg * (-n).
	// Divide the hamiltonian by ℏ, this term becomes
	// nzpf/Φ0(Cg/Cg+Cs+Cj) * Vg * (i(b.H-b))
	// = Ω*Vg*σy, where Ω = nzpf/Φ0(Cg/Cg+Cs+Cj).
	// In other words, we actually do not need an additional waveguide resonator Eq. 6.93 for single qubit rotation, since vanilla transmon already provides σy.

	// Question (b)
	// L = -Cc*dot(Φr)*dot(Φt)
	// H = dot(Φr)∂L/∂dot(Φr) + dot(Φt)∂L/∂dot(Φt) - L = -Cc*dot(Φr)*dot(Φt)
	// Since dot(Φ) = V = Q/C, H = -Cc*(Qr/Cr)*(Qt/CΣ).
	// Since ωr*Zr = 1/Cr, H = -ωr*Cc/CΣ*(ZrQr)*Qt.
	// Since Qt = -2e*n, H = -ωr*(Cc/CΣ)*(ZrQr)*(-2e*n).
	// H = ℏ*ωr*(Cc/CΣ)*(Zr/ℏ*Qr)*e*(2n).
	// H = ℏ*ωr*(Cc/CΣ)*(Zr/ℏ*i*sqrt(ℏ/2Zr)(aH-a))*e*(2*i/sqrt(2)*pow(Ej/(8Ec), 1/4)(b-bH)).
	// H = -ℏ*ωr*(Cc/CΣ)*sqrt(Zr*e*e/(2ℏ))*pow(Ej/(2Ec), 1/4)*(bH-b)*(aH-a).
	// H = -ℏ*g*(bH-b)*(aH-a).
}

func TestCooperPairBox_214(t *testing.T) {
	const maxN = 3
	hamiltonian := func(eqf, ejf, ngf float64) *tensor.Dense {
		eq, ej := complex(float32(eqf), 0), complex(float32(ejf), 0)
		ng := complex(float32(ngf), 0)
		m := 2*maxN + 1
		h := tensor.Zeros(m, m)
		for ni := -maxN; ni < maxN; ni++ {
			n := ni + maxN
			nc := complex(float32(ni), 0)
			h.SetAt([]int{n, n}, eq*(nc-ng)*(nc-ng))
			h.SetAt([]int{n, n + 1}, -ej/2)
			h.SetAt([]int{n + 1, n}, -ej/2)
		}
		h.SetAt([]int{2 * maxN, 2 * maxN}, eq*(maxN-ng)*(maxN-ng))
		return h
	}
	type band struct {
		ng       float64
		energies []float64
	}
	calcBand := func(eq, ej float64) ([]band, error) {
		var bands []band
		bufs := [3]*tensor.Dense{tensor.Zeros(1), tensor.Zeros(1), tensor.Zeros(1)}
		eigvals := tensor.Zeros(1)
		start, end := -1., 1.
		numIntervals := 40
		for i := range numIntervals + 1 {
			ng := start + (end-start)*float64(i)/float64(numIntervals)
			h := hamiltonian(eq, ej, ng)
			if err := tensor.Eig(eigvals, nil, h, bufs); err != nil {
				return nil, errors.Wrap(err, "")
			}
			b := band{ng: ng}
			for _, v := range eigvals.ToSlice1() {
				b.energies = append(b.energies, float64(real(v)))
			}
			bands = append(bands, b)
		}
		return bands, nil
	}
	var records [][]string
	header := []string{"Eq/Ej", "ng", "E0", "E1", "E2"}
	records = append(records, header)
	cases := []struct {
		EqEj float64
		Ej   float64
	}{
		{EqEj: 5, Ej: 5},
		{EqEj: 1, Ej: 5},
		{EqEj: 1. / 5, Ej: 5. / (math.Sqrt(8*1./4*1./5) - 1./4*1./5)},
	}
	for _, x := range cases {
		band, err := calcBand(x.EqEj*x.Ej, x.Ej)
		if err != nil {
			t.Errorf("%+v", err)
		}
		for _, b := range band {
			line := make([]string, 0, 2+len(b.energies))
			line = append(line, strconv.FormatFloat(x.EqEj, 'f', -1, 64))
			line = append(line, strconv.FormatFloat(b.ng, 'f', -1, 64))
			for _, e := range b.energies[:3] {
				line = append(line, strconv.FormatFloat(e, 'f', -1, 64))
			}
			records = append(records, line)
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

func TestEnergyScales_213(t *testing.T) {
	// Junction length by capacitance.
	capacitancePerArea := math.Log10(50) - 15 + 2*6
	energyK := math.Log10(30)
	area := math.Log10(2) + 2*e - (capacitancePerArea + energyK + k)
	lengthNanoM := math.Pow(10, area/2+9)
	t.Logf("lengthNanoM %v", lengthNanoM)

	// Junction length by current.
	Φ0 := math.Log10(π) + ℏ - e
	jc := math.Log10(100) + 2*2
	ejh := math.Log10(5) + 9
	area = math.Log10(2*π) + ejh + math.Log10(2*π) + ℏ - (Φ0 + jc)
	lengthNanoM = math.Pow(10, area/2+9)
	t.Logf("lengthNanoM %v", lengthNanoM)

	// CPB temperature.
	tp := 250e-3
	ekt := func(tp float64) float64 {
		return math.Pow(10, ejh+math.Log10(2*π)+ℏ-math.Log10(2)-(k+math.Log10(tp)))
	}
	groundPopulation := func(tp float64) float64 {
		pg := math.Exp(ekt(tp))
		pe := math.Exp(-ekt(tp))
		z := pg + pe
		return pg / z
	}
	t.Logf("excited population %v", 1-groundPopulation(tp))
	f := func(tp float64) float64 { return groundPopulation(tp) - 0.99 }
	tc, err := root.Brent(f, tp, 1e-3, 1e-6)
	if err != nil {
		t.Errorf("%+v", err)
	}
	tcMilliK := tc * 1e3
	t.Logf("tcMilliK %v", tcMilliK)
}
