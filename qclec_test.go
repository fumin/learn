package learn

import (
	"math/cmplx"
	"testing"

	"github.com/fumin/learn/q"
)

var (
	σx = q.Aσx
	σy = q.Aσy
	σz = q.Aσz
)

func 𐌈(as ...*q.Dense) *q.Dense  { return q.A𐌈(as...) }
func sqrt(x complex64) complex64 { return complex64(cmplx.Sqrt(complex128(x))) }

func Test28_9_258(t *testing.T) {
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

func TestBellStabilizer_252(t *testing.T) {
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

func Test27_3_239(t *testing.T) {
	s000 := 𐌈(q.Z0, q.Sys(q.Z0, 2), q.Sys(q.Z0, 3))
	s111 := 𐌈(q.Z1, q.Sys(q.Z1, 2), q.Sys(q.Z1, 3))
	十 := s000.Add(1, s111).Mul(1 / sqrt(2))
	一 := s000.Add(-1, s111).Mul(1 / sqrt(2))

	一2 := q.PartialDot(q.Sys(σz, 2), 2, 十).Transpose(一.Axis)
	if err := 一.Equal(一2, 1e-6); err != nil {
		t.Errorf("%+v", err)
	}
}
