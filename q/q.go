package q

import (
	"cmp"
	"fmt"
	"math"
	"reflect"
	"slices"

	"github.com/fumin/tensor"
	"github.com/pkg/errors"
)

var (
	One = T2([][]complex64{{1, 0}, {0, 1}})
	Aσx = T2([][]complex64{{0, 1}, {1, 0}})
	Aσy = T2([][]complex64{{0, -1i}, {1i, 0}})
	Aσz = T2([][]complex64{{1, 0}, {0, -1}})
	Z0  = T1([]complex64{1, 0})
	Z1  = T1([]complex64{0, 1})
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

func (a *Dense) Equal(b *Dense, tol float32) error {
	if !slices.Equal(a.Axis, b.Axis) {
		return errors.Errorf("axis not equal")
	}
	if err := a.D.Equal(b.D, tol); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (a *Dense) String() string {
	var d interface{}
	m := ToMat(a)
	switch {
	case len(m.Axis) == 1:
		d = m.D.ToSlice1()
	default:
		d = m.D.ToSlice2()
	}
	return fmt.Sprintf("{Axis:%v, D:%v}", a.Axis, d)
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
	a2 := T1([]complex64{0})
	Copy(a2, a)

	a2.D = a2.D.Mul(c)
	return a2
}

func (a *Dense) Add(c complex64, b *Dense) *Dense {
	a2 := T1([]complex64{0})
	Copy(a2, a)

	a2.D.Add(c, b.Transpose(a2.Axis).D)
	return a2
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

	aMat := ToMat(a)
	eAMat := tensor.Exp(aMat.D, [4]*tensor.Dense{tensor.Zeros(1), tensor.Zeros(1), tensor.Zeros(1), tensor.Zeros(1)})
	eAMat = eAMat.Reshape(shape...)
	eA := eAMat.Transpose(invPerm(axes)...)

	bAxis := make([]Axis, len(a.Axis))
	copy(bAxis, a.Axis)
	return &Dense{Axis: bAxis, D: eA}
}

func PartialDot(a0 *Dense, args ...interface{}) *Dense {
	if len(args)%2 != 0 {
		panic(fmt.Sprintf("len(args) not even number %d", len(args)))
	}
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
		switch {
		case s == nil:
			for _, ax := range a.Axis {
				bax := Axis{System: ax.System, Braket: Bra}
				if ax.Braket == Ket && slices.Contains(buf0.Axis, bax) {
					axes = append(axes, [2]Axis{bax, ax})
				}
			}
		case reflect.TypeOf(s).Kind() == reflect.Slice:
			slc := reflect.ValueOf(s)
			for i := range slc.Len() {
				si := slc.Index(i).Interface()
				axes = append(axes, [2]Axis{{System: si, Braket: Bra}, {System: si, Braket: Ket}})
			}
		default:
			axes = append(axes, [2]Axis{{System: s, Braket: Bra}, {System: s, Braket: Ket}})
		}

		Product(buf1, buf0, a, axes)
		buf0, buf1 = buf1, buf0
	}
	return buf0
}

func Dot(as ...*Dense) *Dense {
	a0 := as[0]
	args := make([]interface{}, 0, 2*len(as)-1)
	for _, a := range as[1:] {
		args = append(args, nil)
		args = append(args, a)
	}
	return PartialDot(a0, args...)
}

// This is the "Old Italic Letter The" U+10308.
// It should be rendered as an 'x' enclosed by a circle, as handled by the "GNU FreeFont" and "Noto Sans Old Italic" fonts.
// Note that some fonts such as "Dejavu" wrongly renders it as a '+'.
func A𐌈(as ...*Dense) *Dense {
	buf0, buf1 := T1([]complex64{0}), T1([]complex64{0})

	Copy(buf0, as[0])
	for _, a := range as[1:] {
		Product(buf1, buf0, a, nil)
		buf0, buf1 = buf1, buf0
	}

	return buf0
}

func Sys(a *Dense, s interface{}) *Dense {
	b := &Dense{Axis: make([]Axis, len(a.Axis)), D: a.D}
	copy(b.Axis, a.Axis)
	for i := range b.Axis {
		b.Axis[i].System = s
	}
	return b
}

func SysReplace(a *Dense, oldS, newS interface{}) *Dense {
	b := &Dense{Axis: make([]Axis, len(a.Axis)), D: a.D}
	copy(b.Axis, a.Axis)
	for i, ax := range b.Axis {
		if ax.System == oldS {
			b.Axis[i].System = newS
		}
	}
	return b
}

func ToMat(a *Dense) *Dense {
	if !slices.ContainsFunc(a.Axis, func(x Axis) bool { return x.Braket == Bra }) {
		return toMat1D(a)
	}

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

func toMat1D(a *Dense) *Dense {
	buf := tensor.Zeros(1).Reset(a.D.Shape()...).Set(nil, a.D)
	buf = buf.Reshape(-1)
	return &Dense{Axis: []Axis{{System: 0, Braket: Ket}}, D: buf}
}
