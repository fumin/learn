package learn

import (
	"flag"
	"fmt"
	"log"
	"slices"
	"testing"

	"github.com/fumin/tensor"
)

func TestExp(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		b *Dense
	}{
		{
			a: &Dense{
				Axis: []Axis{{System: 1, Braket: Ket}, {System: 1, Braket: Bra}, {System: 2, Braket: Ket}, {System: 2, Braket: Bra}},
				D: tensor.T4([][][][]complex64{
					{{{0, 1}, {0, 1i}}, {{0, 0}, {1, 0}}},
					{{{0, 0}, {1, 0}}, {{-1, 1}, {0, -1i}}},
				})},
			b: &Dense{
				Axis: []Axis{{System: 1, Braket: Ket}, {System: 1, Braket: Bra}, {System: 2, Braket: Ket}, {System: 2, Braket: Bra}},
				D: tensor.T4([][][][]complex64{
					{{{1.03337495, 0.84824894 + 0.460866848i}, {0.12532526, 0.5725081 + 0.848248942i}}, {{0.33554159 + 0.125325261i, 0.12532526}, {0.51270736 + 0.335541587i, 0.33554159}}},
					{{{0.33554159 - 0.125325261i, 0.12532526}, {0.84824894 - 0.460866848i, 0.46086685}}, {{0.39534234, 0.51270736 - 0.335541587i}, {0.12532526, 0.5725081 - 0.848248942i}}},
				})},
		},
		{
			a: &Dense{
				Axis: []Axis{{System: 2, Braket: Bra}, {System: 1, Braket: Ket}, {System: 1, Braket: Bra}, {System: 2, Braket: Ket}},
				D: tensor.T4([][][][]complex64{
					{{{0, 0}, {0, 1}}, {{0, 1}, {-1, 0}}},
					{{{1, 1i}, {0, 0}}, {{0, 0}, {1, -1i}}},
				})},
			b: &Dense{
				Axis: []Axis{{System: 2, Braket: Bra}, {System: 1, Braket: Ket}, {System: 1, Braket: Bra}, {System: 2, Braket: Ket}},
				D: tensor.T4([][][][]complex64{
					{{{1.03337495, 0.12532526}, {0.33554159 + 0.125325261i, 0.51270736 + 0.335541587i}}, {{0.33554159 - 0.125325261i, 0.84824894 - 0.460866848i}, {0.39534234, 0.12532526}}},
					{{{0.84824894 + 0.460866848i, 0.5725081 + 0.848248942i}, {0.12532526, 0.33554159}}, {{0.12532526, 0.46086685}, {0.51270736 - 0.33554159i, 0.5725081 - 0.84824894i}}},
				})},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			b := Exp(test.a)
			if !slices.Equal(b.Axis, test.b.Axis) {
				t.Errorf("%v %v", b.Axis, test.b.Axis)
			}
			if err := b.D.Equal(test.b.D, 1e-6); err != nil {
				t.Errorf("%+v", err)
			}
		})
	}
}

func TestInvPerm(t *testing.T) {
	t.Parallel()
	tests := []struct {
		p    []int
		invp []int
	}{
		{
			p:    []int{0, 3, 2, 1},
			invp: []int{0, 3, 2, 1},
		},
		{
			p:    []int{1, 2, 3, 4, 0},
			invp: []int{4, 0, 1, 2, 3},
		},
		{
			p:    []int{4, 8, 0, 7, 1, 5, 3, 6, 2},
			invp: []int{2, 4, 8, 6, 0, 5, 7, 3, 1},
		},
		{
			p:    []int{2, 7, 4, 9, 8, 3, 5, 0, 6, 1},
			invp: []int{7, 9, 0, 5, 2, 6, 8, 1, 4, 3},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			p := make([]int, len(test.p))
			copy(p, test.p)
			invp := invPerm(p)
			if !slices.Equal(invp, test.invp) {
				t.Errorf("%v %v", invp, test.invp)
			}
			p = invPerm(test.invp)
			if !slices.Equal(p, test.p) {
				t.Errorf("%v %v", p, test.p)
			}
		})
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}
