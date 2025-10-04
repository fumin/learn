package learn

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"github.com/fumin/tensor"
	"gonum.org/v1/gonum/optimize"
)

func TestCNOTMolmer_159(t *testing.T) {
	ums := func(t float64) *Dense {
		var jms complex64 = 1
		u := 𐌈(σx, sys(σx, 2)).Mul(-1i * jms * complex(float32(t), 0))
		return Exp(u)
	}
	cnott := func(t float64) *Dense {
		gates := []*Dense{
			𐌈(R(0, 1, 0, π/2), sys(one, 2)),
			ums(t),
			𐌈(R(1, 0, 0, -π/2), sys(one, 2)),
			𐌈(R(0, 1, 0, -π/2), sys(R(1, 0, 0, -π/2), 2)),
		}
		slices.Reverse(gates)
		return dot(gates...).Transpose(cnot.Axis)
	}
	abs := func(x complex64) float64 { return cmplx.Abs(complex128(x)) }
	problem := optimize.Problem{
		Func: func(x []float64) float64 {
			cn := toMat(cnott(x[0])).D
			return abs(cn.At(2, 2)) + abs(cn.At(3, 3))
		},
	}
	result, err := optimize.Minimize(problem, []float64{0}, nil, nil)
	if err != nil {
		t.Errorf("%+v", err)
	}
	tcnot := result.X[0]
	if diff := math.Abs(tcnot - π/4); diff > 1e-6 {
		t.Errorf("%f %v", diff, tcnot)
	}
	mscnot := cnott(tcnot)
	phase := (1 + 1i) / sqrt(2)
	if err := mscnot.D.Equal(cnot.Mul(phase).D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestSideband_159(t *testing.T) {
	// n is the number of motional states.
	n := 3
	// m0 is the motional ground state.
	m0 := T1(make([]complex64, n))
	m0.D.SetAt([]int{0}, 1)
	const motional = "motional"
	m0 = sys(m0, motional)
	// a is the creation operator
	a := sys(T2(tensor.Zeros(n, n).ToSlice2()), motional)
	for i := range n - 1 {
		ic := complex(float32(i+1), 0)
		a.D.SetAt([]int{i, i + 1}, sqrt(ic))
	}
	red := func(θ, ϕ complex64) *Dense {
		aeg := 𐌈(a, dot(z1, z0.H()))
		aHge := 𐌈(a.H(), dot(z0, z1.H()))
		u := aeg.Mul(exp(1i*ϕ)).Add(-exp(-1i*ϕ), aHge).Mul(θ / 2)
		return Exp(u)
	}

	rπ2 := red(π/sqrt(2), π/2)
	// red4 is in the basis |e, 0>, |g, 1>, |e, 1>, |g, 2>.
	red4 := func(θ, ϕ complex64) *tensor.Dense {
		return tensor.T2([][]complex64{
			{cos(θ / 2), 1i * exp(1i*ϕ) * sin(θ/2), 0, 0},
			{1i * exp(-1i*ϕ) * sin(θ/2), cos(θ / 2), 0, 0},
			{0, 0, cos(θ / sqrt(2)), 1i * exp(1i*ϕ) * sin(θ/sqrt(2))},
			{0, 0, 1i * exp(-1i*ϕ) * sin(θ/sqrt(2)), cos(θ / sqrt(2))},
		})
	}
	rπ24 := red4(π/sqrt(2), 0)
	rπ2Mat := toMat(rπ2).D.Slice([][2]int{{1, 5}, {1, 5}})
	if err := rπ2Mat.Equal(rπ24, 1e-6); err != nil {
		t.Errorf("%+v %v %v", err, rπ2Mat.ToSlice2(), rπ24.ToSlice2())
	}

	// Compute the swap operator.
	ϕs := complex(float32(math.Acos(math.Pow(1./math.Tan(π/math.Sqrt(2)), 2))), 0) + π/2
	swap2 := dot(rπ2, red(2*π/sqrt(2), ϕs), rπ2)
	for i, v := range swap2.D.All() {
		if cmplx.Abs(complex128(v)) < 1e-6 {
			swap2.D.SetAt(i, 0)
		}
		if cmplx.Abs(complex128(v-1)) < 1e-6 {
			swap2.D.SetAt(i, 1)
		}
	}
	// swap is in the basis |g, 0>, |e, 0>, |g, 1>, |e, 1>.
	swap := toMat(swap2).D.Slice([][2]int{{0, 4}, {0, 4}})
	cπ := cot(π / sqrt(2))
	swapExact := tensor.T2([][]complex64{
		{1, 0, 0, 0},
		{0, 0, -sqrt(1-pow(cπ, 2)) + 1i*cπ, 0},
		{0, sqrt(1-pow(cπ, 2)) + 1i*cπ, 0, 0},
		{0, 0, 0, 1},
	})
	if err := swap.Equal(swapExact, 5e-6); err != nil {
		t.Errorf("%+v", err)
	}
	t.Logf("swap %v", swap.ToSlice2())

	// Define the entangling operator.
	blue := func(θ, ϕ complex64) *Dense {
		u1 := 𐌈(a.H(), dot(z1, z0.H()))
		u2 := 𐌈(a, dot(z0, z1.H()))
		u := u1.Mul(exp(1i*ϕ)).Add(-exp(-1i*ϕ), u2).Mul(θ / 2)
		return Exp(u)
	}
	blue1 := 𐌈(blue(π/2, π/2), sys(one, 2))
	red2 := 𐌈(sysReplace(red(π, π/2), 1, 2), one)
	entangle := dot(red2, blue1).Transpose(blue1.Axis)
	gg0 := 𐌈(z0, sys(z0, 2), m0)
	ent := dot(entangle, gg0).Transpose(gg0.Axis)
	ee0 := 𐌈(z1, sys(z1, 2), m0)
	entTrue := gg0.Mul(1/sqrt(2)).Add(-1/sqrt(2), ee0)
	if err := ent.D.Equal(entTrue.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestIonPhononCoupling_158(t *testing.T) {
	var ωq complex64 = 13
	var ωz complex64 = 7
	var ti complex64 = 1
	// n is the number of motional states.
	n := 6
	// m1 is the motional identity operator.
	m1 := sys(T2(tensor.Zeros(1).Eye(n, 0).ToSlice2()), 2)
	// a is the creation operator
	a := sys(T2(tensor.Zeros(n, n).ToSlice2()), 2)
	for i := range n - 1 {
		ic := complex(float32(i+1), 0)
		a.D.SetAt([]int{i, i + 1}, sqrt(ic))
	}
	hmo := dot(a.H(), a).Add(1./2, m1)
	h0 := 𐌈(σz, m1).Mul(-1./2*ωq).Add(ωz, 𐌈(one, hmo))
	// u0 is exp(-i*h0*t).
	u0 := Copy(T1([]complex64{0}), h0)
	for i, v := range u0.D.All() {
		if i[0] == i[1] && i[2] == i[3] {
			u0.D.SetAt(i, exp(-1i*v*ti))
		}
	}

	// Check first equation of 5.54.
	uau := dot(u0.H(), 𐌈(one, a), u0).Transpose(u0.Axis)
	ea := 𐌈(one, a).Mul(exp(-1i * ωz * ti))
	if err := uau.D.Equal(ea.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}

	// Check second equation of 5.54.
	eg := 𐌈(dot(z1, z0.H()), m1)
	uegu := dot(u0.H(), eg, u0).Transpose(u0.Axis)
	eeg := Copy(T1([]complex64{0}), eg).Mul(exp(1i * ωq * ti))
	if err := uegu.D.Equal(eeg.D, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}

func TestTrappingParam_158(t *testing.T) {
	u0 := math.Log(20)
	v0 := math.Log(100)
	Ωm := math.Log(2 * π * 22 * 1e6)
	d0 := math.Log(2.7 / 2 * 1e-3)
	s0 := math.Log(0.46 * 1e-3)
	m := math.Log(2.873 * 1e-25)
	q := math.Log(e)

	a := math.Exp(math.Log(4) + q + u0 - (Ωm + Ωm + m + d0 + d0))
	b := math.Exp(math.Log(2) + q + v0 - (Ωm + Ωm + m + s0 + s0))
	if diff := math.Abs(b - 0.028); diff > 1e-3 {
		t.Errorf("%f %f", b, diff)
	}
	if diff := math.Abs(a/b - 0.046); diff > 1e-3 {
		t.Errorf("%f %f", a/b, diff)
	}
}
