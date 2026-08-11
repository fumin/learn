package bn128

import (
	"flag"
	"log"
	"math/big"
	"testing"

	"github.com/fumin/learn/zk/bn128/gfp"
	"github.com/fumin/learn/zk/ecc"
	"github.com/fumin/nag"
)

func TestG1(t *testing.T) {
	x := add(G1, G1)
	y := x.NewOne().Inv(x)
	if z := add(x, y); !z.Equal(x.NewOne()) {
		t.Errorf("%v", z)
	}
	if z := add(add(x, y), x); !z.Equal(x) {
		t.Errorf("%v", z)
	}

	// The below tests are from test_G1_object in
	// https://github.com/ethereum/py_ecc/blob/v8.0.0/tests/core/test_bn128_and_bls12_381.py
	x = add(add(double(G1), G1), G1)
	y = double(double(G1))
	if !x.Equal(y) {
		t.Errorf("%v %v", x, y)
	}
	if x := double(G1); x.Equal(G1) {
		t.Errorf("%v %v", x, G1)
	}
	x = add(multiply(9, G1), multiply(5, G1))
	y = add(multiply(12, G1), multiply(2, G1))
	if !x.Equal(y) {
		t.Errorf("%v %v", x, y)
	}
	exponent := new(big.Int).Set(curveOrder)
	if x := nag.Pow(G1.NewOne().Set(G1), exponent); !x.Equal(G1.NewOne()) {
		t.Errorf("%v", x)
	}
}

func TestG2(t *testing.T) {
	x := add(G2, G2)
	y := x.NewOne().Inv(x)
	if z := add(x, y); !z.Equal(x.NewOne()) {
		t.Errorf("%v", z)
	}
	if z := add(add(x, y), x); !z.Equal(x) {
		t.Errorf("%v", z)
	}

	// The below tests are from test_G1_object in
	// https://github.com/ethereum/py_ecc/blob/v8.0.0/tests/core/test_bn128_and_bls12_381.py
	x = add(add(double(G2), G2), G2)
	y = double(double(G2))
	if !x.Equal(y) {
		t.Errorf("%v %v", x, y)
	}
	if x := double(G2); x.Equal(G2) {
		t.Errorf("%v", x)
	}
	x = add(multiply(9, G2), multiply(5, G2))
	y = add(multiply(12, G2), multiply(2, G2))
	if !x.Equal(y) {
		t.Errorf("%v %v", x, y)
	}
	exponent := new(big.Int).Set(curveOrder)
	if x := nag.Pow(G2.NewOne().Set(G2), exponent); !x.Equal(G2.NewOne()) {
		t.Errorf("%v", x)
	}
	exponent = new(big.Int).Sub(new(big.Int).Mul(big.NewInt(2), fieldChar), curveOrder)
	if x := nag.Pow(G2.NewOne().Set(G2), exponent); x.Equal(G2.NewOne()) {
		t.Errorf("%v", x)
	}
}

func TestG12(t *testing.T) {
	// Check u^2 = -1.
	u2want := fp12.NewZero().SetCoeffs(big.NewInt(-1), big.NewInt(0))
	if u2 := nag.Pow(fp2.NewZero().Set(u), big.NewInt(2)); !u2.Equal(u2want) {
		t.Errorf("%v^2 = %v want %v", u, u2, u2want)
	}
	b2want := fp2.NewZero().SetCoeffs(ecc.Int10("19485874751759354771024239261021720505790618469301721065564631296452457478373"), ecc.Int10("266929791119991161246907387137283842545076965332900288569378510910307636690"))
	if !b2.Equal(b2want) {
		t.Errorf("%v != %v", b2, b2want)
	}

	g12want := g12.NewOne().SetCoords([]*FiniteExt[*gfp.Gfp]{
		fp12.NewZero().SetCoeffs(ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("16260673061341949275257563295988632869519996389676903622179081103440260644990"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("11559732032986387107991004021392285783925812861821192530917403151452391805634"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0")),
		fp12.NewZero().SetCoeffs(ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("15530828784031078730107954109694902500959150953518636601196686752670329677317"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("0"), ecc.Int10("4082367875863433681332203403145435568316851327593401208105741076214120093531"), ecc.Int10("0"), ecc.Int10("0")),
	})
	if !g12.Equal(g12want) {
		t.Errorf("%v != %v", g12, g12want)
	}
	exponent := new(big.Int).Set(curveOrder)
	if x := nag.Pow(g12.NewOne().Set(g12), exponent); !x.Equal(g12.NewOne()) {
		t.Errorf("%v", x)
	}

	// Check that the twist map is an isomorphism.
	p, q := multiply(5, G2), multiply(12, G2)
	pq := add(p, q)
	tpq := twist(u, w, g12.NewOne(), pq)
	tp, tq := twist(u, w, g12.NewOne(), p), twist(u, w, g12.NewOne(), q)
	tptq := add(tp, tq)
	if !tpq.Equal(tptq) {
		t.Errorf("%v != %v", tpq, tptq)
	}

	// Check that g12 is the G2 group which is sent to O by the trace map.
	p = multiply(2834283373991, g12)
	o := ecc.Trace(p)
	if !o.Equal(g12.NewOne()) {
		t.Errorf("Trace(%v) = %v want O", p, o)
	}
}

func TestPairing(t *testing.T) {
	p1 := Pairing(G2, G1)
	wantP1 := fp12.NewZero().SetCoeffs(
		ecc.Int10("18443897754565973717256850119554731228214108935025491924036055734000366132575"),
		ecc.Int10("10734401203193558706037776473742910696504851986739882094082017010340198538454"),
		ecc.Int10("5985796159921227033560968606339653189163760772067273492369082490994528765680"),
		ecc.Int10("4093294155816392700623820137842432921872230622290337094591654151434545306688"),
		ecc.Int10("642121370160833232766181493494955044074321385528883791668868426879070103434"),
		ecc.Int10("4527449849947601357037044178952942489926487071653896435602814872334098625391"),
		ecc.Int10("3758435817766288188804561253838670030762970764366672594784247447067868088068"),
		ecc.Int10("18059168546148152671857026372711724379319778306792011146784665080987064164612"),
		ecc.Int10("14656606573936501743457633041048024656612227301473084805627390748872617280984"),
		ecc.Int10("17918828665069491344039743589118342552553375221610735811112289083834142789347"),
		ecc.Int10("19455424343576886430889849773367397946457449073528455097210946839000147698372"),
		ecc.Int10("7484542354754424633621663080190936924481536615300815203692506276894207018007"))
	if !p1.Equal(wantP1) {
		t.Errorf("%v != %v", p1, wantP1)
	}

	pn1 := Pairing(G2, G1.NewOne().Inv(G1))
	p1pn1 := p1.NewZero().Mul(p1, pn1)
	if !p1pn1.Equal(fp12.NewOne()) {
		t.Errorf("%v", p1pn1)
	}

	p1CO := nag.Pow(p1.NewZero().Set(p1), new(big.Int).Set(curveOrder))
	if !p1CO.Equal(fp12.NewOne()) {
		t.Errorf("%v", p1CO)
	}

	p3 := Pairing(multiply(27, G2), multiply(37, G1))
	po3 := Pairing(G2, multiply(999, G1))
	if !p3.Equal(po3) {
		t.Errorf("%v != %v", p3, po3)
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}

func add[G nag.Group[G]](x, y G) G {
	return x.NewOne().Mul(x, y)
}

func double[G nag.Group[G]](x G) G {
	return x.NewOne().Mul(x, x)
}

func multiply[G nag.Group[G]](n int64, x G) G {
	return nag.Pow(x.NewOne().Set(x), big.NewInt(n))
}
