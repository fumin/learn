package ecc

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/fumin/nag"
	"github.com/fumin/nag/field"
)

func TestDivide(t *testing.T) {
	tests := []struct {
		p    int64
		vars []string
		f    string
		g    string
		q    string
		r    string
	}{
		{
			p:    23,
			vars: []string{"x", "y"},
			f:    "x^3+3xy+2x^2+2y+2x+5",
			g:    "x-7",
			q:    "x^2+9x+3y-4",
			r:    "0",
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExtDeg(big.NewInt(test.p), 1)
			vs := make(map[string]nag.Symbol)
			for _, v := range test.vars {
				vs[v] = nag.Symbol(len(vs))
			}
			f := parse(vs, k, test.f)
			g := parse(vs, k, test.g)
			qWant := parse(vs, k, test.q)
			rWant := parse(vs, k, test.r)
			q, r := Divide(f, g)
			if !q.Equal(qWant) {
				t.Errorf("Divide(%v, %v) = %v want %v", f, g, q, r)
			}
			if !r.Equal(rWant) {
				t.Errorf("Divide(%v, %v) = %v want %v", f, g, q, r)
			}
		})
	}
}

func TestReduceFraction(t *testing.T) {
	tests := []struct {
		vars    []string
		d       [2]string
		reduced [2]string
	}{
		{
			vars:    []string{"x", "y"},
			d:       [2]string{"x^2y", "xy^2"},
			reduced: [2]string{"x", "y"},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			vs := make(map[string]nag.Symbol)
			for _, v := range test.vars {
				vs[v] = nag.Symbol(len(vs))
			}
			d := [2]*nag.Polynomial[*nag.Rat]{}
			d[0], _ = nag.Parse(vs, nag.Deglex, test.d[0])
			d[1], _ = nag.Parse(vs, nag.Deglex, test.d[1])
			reducedWant := [2]*nag.Polynomial[*nag.Rat]{}
			reducedWant[0], _ = nag.Parse(vs, nag.Deglex, test.reduced[0])
			reducedWant[1], _ = nag.Parse(vs, nag.Deglex, test.reduced[1])
			reduced := reduceFraction(d)
			if !reduced[0].Equal(reducedWant[0]) {
				t.Errorf("reduceFraction(%v) = %v want %v", test.d, reduced, reducedWant)
			}
			if !reduced[1].Equal(reducedWant[1]) {
				t.Errorf("reduceFraction(%v) = %v want %v", test.d, reduced, reducedWant)
			}
		})
	}
}

func TestGcd(t *testing.T) {
	tests := []struct {
		p    int64
		vars []string
		f    string
		g    string
		lcm  string
		gcd  string
	}{
		{
			p:    23,
			vars: []string{"x", "y"},
			f:    "x^3+4xy+4x^2+15y+x+22",
			g:    "x^2+9x+3",
			lcm:  "x^5-10x^4+4x^3y-6x^3+5x^2y-3x^2+9xy-6x-y-3",
			gcd:  "1",
		},
		{
			p:    23,
			vars: []string{"x", "y"},
			f:    "xy+11x^2+21y+14x+20",
			g:    "x+21",
			lcm:  "11x^2+xy-9x-2y-3",
			gcd:  "x-2",
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			k := field.NewPrimeExtDeg(big.NewInt(test.p), 1)
			vs := make(map[string]nag.Symbol)
			for _, v := range test.vars {
				vs[v] = nag.Symbol(len(vs))
			}
			f := parse(vs, k, test.f)
			g := parse(vs, k, test.g)
			lcmWant := parse(vs, k, test.lcm)
			if lcm := Lcm(f, g); !lcm.Equal(lcmWant) {
				t.Errorf("Lcm(%v, %v) = %v want %v", f, g, lcm, lcmWant)
			}
			gcdWant := parse(vs, k, test.gcd)
			if gcd := Gcd(f, g); !gcd.Equal(gcdWant) {
				t.Errorf("Gcd(%v, %v) = %v want %v", f, g, gcd, gcdWant)
			}
		})
	}
}

func TestGcdQ(t *testing.T) {
	tests := []struct {
		vars []string
		f    string
		g    string
		lcm  string
		gcd  string
	}{
		{
			vars: []string{"x", "y"},
			f:    "x^2y^2+7x^2y+3xy^2+12x^2+21xy+2y^2+36x+14y+24",
			g:    "x^2y^2+6x^2y+4xy^2+8x^2+24xy+3y^2+32x+18y+24",
			lcm:  "x^3y^3+9x^3y^2+6x^2y^3+26x^3y+54x^2y^2+11xy^3+24x^3+156x^2y+99xy^2+6y^3+144x^2+286xy+54y^2+264x+156y+144",
			gcd:  "xy+4x+y+4",
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			vs := make(map[string]nag.Symbol)
			for _, v := range test.vars {
				vs[v] = nag.Symbol(len(vs))
			}
			f, _ := nag.Parse(vs, nag.Deglex, test.f)
			g, _ := nag.Parse(vs, nag.Deglex, test.g)
			lcmWant, _ := nag.Parse(vs, nag.Deglex, test.lcm)
			gcdWant, _ := nag.Parse(vs, nag.Deglex, test.gcd)
			if lcm := Lcm(f, g); !lcm.Equal(lcmWant) {
				t.Errorf("Lcm(%v, %v) = %v want %v", f, g, lcm, lcmWant)
			}
			if gcd := Gcd(f, g); !gcd.Equal(gcdWant) {
				t.Errorf("Gcd(%v, %v) = %v want %v", f, g, gcd, gcdWant)
			}
		})
	}
}
