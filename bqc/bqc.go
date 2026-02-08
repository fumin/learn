package bqc

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"reflect"
	"slices"

	"github.com/fumin/tensor"
)

const (
	π = math.Pi
)

var (
	ℏ = math.Log10(1.054571817) - 34
	k = math.Log10(1.380649) - 23
	e = math.Log10(1.602176634) - 19

	one = T2([][]complex64{{1, 0}, {0, 1}})

	// Pauli matrices.
	σx = T2([][]complex64{{0, 1}, {1, 0}})
	σy = T2([][]complex64{{0, -1i}, {1i, 0}})
	σz = T2([][]complex64{{1, 0}, {0, -1}})
	h  = T2([][]complex64{{1 / math.Sqrt2, 1 / math.Sqrt2}, {1 / math.Sqrt2, -1 / math.Sqrt2}})
	z0 = T1([]complex64{1, 0})
	z1 = T1([]complex64{0, 1})
	x0 = T1([]complex64{1 / math.Sqrt2, 1 / math.Sqrt2})
	x1 = T1([]complex64{1 / math.Sqrt2, -1 / math.Sqrt2})
	y0 = T1([]complex64{1 / math.Sqrt2, 1i / math.Sqrt2})
	y1 = T1([]complex64{1 / math.Sqrt2, -1i / math.Sqrt2})
	h0 = T1([]complex64{cos(π / 8), sin(π / 8)})
	h1 = T1([]complex64{-sin(π / 8), cos(π / 8)})

	// R is the rotation along the n = (x, y, z) axis with θ angle.
	// R = exp(-iθ/2 n*σ)
	R = func(x, y, z, θ complex64) *Dense {
		r := Copy(T1([]complex64{0}), one).Mul(cos(θ / 2))
		r = r.Add(-1i*sin(θ/2)*x, σx)
		r = r.Add(-1i*sin(θ/2)*y, σy)
		r = r.Add(-1i*sin(θ/2)*z, σz)
		return r
	}

	// Bell states.
	b00  = 𐌈(z0, sys(z0, 2)).Add(1, 𐌈(z1, sys(z1, 2))).Mul(1. / math.Sqrt2)
	b01  = 𐌈(z0, sys(z0, 2)).Add(-1, 𐌈(z1, sys(z1, 2))).Mul(1. / math.Sqrt2)
	b10  = 𐌈(z0, sys(z1, 2)).Add(1, 𐌈(z1, sys(z0, 2))).Mul(1. / math.Sqrt2)
	b11  = 𐌈(z0, sys(z1, 2)).Add(-1, 𐌈(z1, sys(z0, 2))).Mul(1. / math.Sqrt2)
	bell = 𐌈(𐌈(z0, sys(z0, 2)), b00.H()).Add(1, 𐌈(𐌈(z0, sys(z1, 2)), b01.H())).Add(1, 𐌈(𐌈(z1, sys(z0, 2)), b10.H())).Add(1, 𐌈(𐌈(z1, sys(z1, 2)), b11.H()))

	cz   = 𐌈(z0, z0.H(), sys(one, 2)).Add(1, 𐌈(z1, z1.H(), sys(σz, 2)))
	cnot = 𐌈(z0, z0.H(), sys(one, 2)).Add(1, 𐌈(z1, z1.H(), sys(σx, 2)))
)

const (
	Bra Braket = iota
	Ket
)

type Braket int

func (b Braket) String() string {
	if b == Bra {
		return "bra"
	}
	return "ket"
}

type Axis struct {
	System interface{}
	Braket Braket
}

type Dense struct {
	Axis []Axis
	D    *tensor.Dense
}

func T1(s []complex64) *Dense {
	return &Dense{Axis: []Axis{{System: 1, Braket: Ket}}, D: tensor.T1(s)}
}

func T2(s [][]complex64) *Dense {
	return &Dense{Axis: []Axis{{System: 1, Braket: Ket}, {System: 1, Braket: Bra}}, D: tensor.T2(s)}
}

func (a *Dense) Index(ax Axis, idx int) *Dense {
	axIdx := slices.Index(a.Axis, ax)
	if axIdx == -1 {
		panic(fmt.Sprintf("axis not found"))
	}

	shape := make([]int, 0, len(a.Axis))
	axes := make([][2]int, 0, len(a.Axis))
	for i := range len(a.Axis) {
		if i == axIdx {
			axes = append(axes, [2]int{idx, idx + 1})
			continue
		}
		shape = append(shape, a.D.Shape()[i])
		axes = append(axes, [2]int{0, a.D.Shape()[i]})
	}
	sliced := a.D.Slice(axes)
	buf := tensor.Zeros(1).Reset(sliced.Shape()...).Set(nil, sliced)
	d := buf.Reshape(shape...)

	b := &Dense{Axis: make([]Axis, len(a.Axis)), D: d}
	copy(b.Axis, a.Axis)
	b.Axis = slices.Delete(b.Axis, axIdx, axIdx+1)
	return b
}

func (a *Dense) Conj() *Dense {
	return &Dense{Axis: a.Axis, D: a.D.Conj()}
}

func (a *Dense) H() *Dense {
	allSameBK := true
	for i := range len(a.Axis) - 1 {
		if a.Axis[i].Braket != a.Axis[i+1].Braket {
			allSameBK = false
			break
		}
	}
	if allSameBK {
		axes := make([]Axis, len(a.Axis))
		copy(axes, a.Axis)
		newBK := (axes[0].Braket + 1) % 2
		for i := range axes {
			axes[i].Braket = newBK
		}
		return &Dense{Axis: axes, D: a.D.Conj()}
	}
	return &Dense{Axis: a.Axis, D: a.D.H()}
}

func (a *Dense) Mul(c complex64) *Dense {
	a.D = a.D.Mul(c)
	return a
}

func (a *Dense) Add(c complex64, b *Dense) *Dense {
	return &Dense{Axis: a.Axis, D: a.D.Add(c, b.Transpose(a.Axis).D)}
}

func (a *Dense) Transpose(axis []Axis) *Dense {
	axisIndices := make([]int, 0, len(axis))
	for _, ax := range axis {
		found := slices.Index(a.Axis, ax)
		if found == -1 {
			panic(fmt.Sprintf("axis %v not found in %v", ax, a.Axis))
		}
		axisIndices = append(axisIndices, found)
	}
	return &Dense{Axis: axis, D: a.D.Transpose(axisIndices...)}
}

func Product(c, a, b *Dense, axes [][2]Axis) *Dense {
	var aRemoves, bRemoves []int
	axesIndices := make([][2]int, 0, len(axes))
	for i, ai := range axes {
		aIdx := slices.Index(a.Axis, ai[0])
		if aIdx == -1 {
			panic(fmt.Sprintf("axis not found in tensor A at %d %v among %v", i, ai[0], a.Axis))
		}
		aRemoves = append(aRemoves, aIdx)
		bIdx := slices.Index(b.Axis, ai[1])
		if bIdx == -1 {
			panic(fmt.Sprintf("axis not found in tensor B at %d %v among %v", i, ai[1], b.Axis))
		}
		bRemoves = append(bRemoves, bIdx)
		axesIndices = append(axesIndices, [2]int{aIdx, bIdx})
	}

	c.Axis = c.Axis[:0]
	for i, ax := range a.Axis {
		if slices.Contains(aRemoves, i) {
			continue
		}

		if slices.Contains(c.Axis, ax) {
			panic(fmt.Sprintf("duplicate A axis %v", ax))
		}
		c.Axis = append(c.Axis, ax)
	}
	for i, ax := range b.Axis {
		if slices.Contains(bRemoves, i) {
			continue
		}

		if slices.Contains(c.Axis, ax) {
			panic(fmt.Sprintf("duplicate B axis %v", ax))
		}
		c.Axis = append(c.Axis, ax)
	}

	tensor.Product(c.D, a.D, b.D, axesIndices)
	return c
}

func Copy(b, a *Dense) *Dense {
	b.Axis = b.Axis[:0]
	for _, ax := range a.Axis {
		b.Axis = append(b.Axis, ax)
	}

	b.D.Reset(a.D.Shape()...).Set(nil, a.D)
	return b
}

func Exp(a *Dense) *Dense {
	axes := toMatAxes(a)
	shape := make([]int, len(a.Axis))
	for i := range len(shape) {
		shape[i] = a.D.Shape()[axes[i]]
	}

	aMat := toMat(a)
	eAMat := tensor.Exp(aMat.D, [4]*tensor.Dense{tensor.Zeros(1), tensor.Zeros(1), tensor.Zeros(1), tensor.Zeros(1)})
	eAMat = eAMat.Reshape(shape...)
	eA := eAMat.Transpose(invPerm(axes)...)

	bAxis := make([]Axis, len(a.Axis))
	copy(bAxis, a.Axis)
	return &Dense{Axis: bAxis, D: eA}
}

func pdot(a0 *Dense, args ...interface{}) *Dense {
	systems := make([]interface{}, 0, len(args)/2)
	as := make([]*Dense, 0, len(args)/2)
	for i, arg := range args {
		if i%2 == 0 {
			systems = append(systems, arg)
		} else {
			as = append(as, arg.(*Dense))
		}
	}

	buf0, buf1 := T1([]complex64{0}), T1([]complex64{0})
	Copy(buf0, a0)
	for i, a := range as {
		s := systems[i]

		var axes [][2]Axis
		if s == nil {
			for _, ax := range a.Axis {
				if ax.Braket == Ket {
					axes = append(axes, [2]Axis{{System: ax.System, Braket: Bra}, {System: ax.System, Braket: Ket}})
				}
			}
		} else if reflect.TypeOf(s).Kind() == reflect.Slice {
			slc := reflect.ValueOf(s)
			for i := range slc.Len() {
				si := slc.Index(i).Interface()
				axes = append(axes, [2]Axis{{System: si, Braket: Bra}, {System: si, Braket: Ket}})
			}
		} else {
			axes = append(axes, [2]Axis{{System: s, Braket: Bra}, {System: s, Braket: Ket}})
		}

		Product(buf1, buf0, a, axes)
		buf0, buf1 = buf1, buf0
	}
	return buf0
}

func dot(as ...*Dense) *Dense {
	a0 := as[0]
	args := make([]interface{}, 0, 2*len(as)-1)
	for _, a := range as[1:] {
		args = append(args, nil)
		args = append(args, a)
	}
	return pdot(a0, args...)
}

// This is the "Old Italic Letter The" U+10308.
// It should be rendered as an 'x' enclosed by a circle, as handled by the "GNU FreeFont" and "Noto Sans Old Italic" fonts.
// Note that some fonts such as "Dejavu" wrongly renders it as a '+'.
func 𐌈(as ...*Dense) *Dense {
	buf0, buf1 := T1([]complex64{0}), T1([]complex64{0})

	Copy(buf0, as[0])
	for _, a := range as[1:] {
		Product(buf1, buf0, a, nil)
		buf0, buf1 = buf1, buf0
	}

	return buf0
}

func sys(a *Dense, s interface{}) *Dense {
	b := &Dense{Axis: make([]Axis, len(a.Axis)), D: a.D}
	copy(b.Axis, a.Axis)
	for i := range b.Axis {
		b.Axis[i].System = s
	}
	return b
}

func sysReplace(a *Dense, oldS, newS interface{}) *Dense {
	b := &Dense{Axis: make([]Axis, len(a.Axis)), D: a.D}
	copy(b.Axis, a.Axis)
	for i, ax := range b.Axis {
		if ax.System == oldS {
			b.Axis[i].System = newS
		}
	}
	return b
}

func toMatAxes(a *Dense) []int {
	systems := make([]interface{}, 0, len(a.Axis)/2)
	for _, ax := range a.Axis {
		if ax.Braket == Ket {
			systems = append(systems, ax.System)
		}
	}
	compare := func(a, b interface{}) int {
		ai := slices.Index(systems, a)
		bi := slices.Index(systems, b)
		return cmp.Compare(ai, bi)
	}

	// Move all kets to the front.
	type AxisIdx struct {
		Axis Axis
		Idx  int
	}
	ais := make([]AxisIdx, 0, len(a.Axis))
	for i, ax := range a.Axis {
		ais = append(ais, AxisIdx{Axis: ax, Idx: i})
	}
	slices.SortFunc(ais, func(a, b AxisIdx) int {
		if a.Axis.Braket != b.Axis.Braket {
			if a.Axis.Braket == Ket {
				return -1
			}
			return 1
		}
		return compare(a.Axis.System, b.Axis.System)
	})
	// Double capacity so that users can invPerm.
	axes := make([]int, 0, len(a.D.Shape())*2)
	for _, a := range ais {
		axes = append(axes, a.Idx)
	}
	return axes
}

func toMat(a *Dense) *Dense {
	// Compute matrix shape.
	nElems := 1
	for _, d := range a.D.Shape() {
		nElems *= d
	}
	n := int(math.Sqrt(float64(nElems)))

	axes := toMatAxes(a)

	aT := a.D.Transpose(axes...)
	buf := tensor.Zeros(1).Reset(aT.Shape()...).Set(nil, aT)
	buf = buf.Reshape(n, n)

	return &Dense{Axis: []Axis{{System: 0, Braket: Ket}, {System: 0, Braket: Bra}}, D: buf}
}

func randC(n int) []complex64 {
	c := make([]complex64, 0, n)
	for range n {
		c = append(c, complex64(complex(rand.Float64(), rand.Float64())))
	}
	var sum complex64
	for _, ci := range c {
		sum += ci * conj(ci)
	}
	for i := range c {
		c[i] /= sqrt(sum)
	}
	return c
}

func invPerm(p []int) []int {
	n := len(p)
	for range n {
		p = append(p, -1)
	}
	invp := p[n:]
	p = p[:n]

	for i, v := range p {
		invp[v] = i
	}

	copy(p, invp)
	return p
}

func conj(x complex64) complex64 {
	return complex64(cmplx.Conj(complex128(x)))
}

func sqrt(x complex64) complex64 {
	return complex64(cmplx.Sqrt(complex128(x)))
}

func sin(x complex64) complex64 {
	return complex64(cmplx.Sin(complex128(x)))
}

func cos(x complex64) complex64 {
	return complex64(cmplx.Cos(complex128(x)))
}

func cot(x complex64) complex64 {
	return complex64(cmplx.Cot(complex128(x)))
}

func exp(x complex64) complex64 {
	return complex64(cmplx.Exp(complex128(x)))
}

func pow(x, y complex64) complex64 {
	return complex64(cmplx.Pow(complex128(x), complex128(y)))
}
