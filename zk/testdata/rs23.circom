// This circom script corresponds to the TestRS2_3 test.
pragma circom 2.2.3;

// Power performs out = in^n, for a compile-time constant exponent n >= 0.
// Uses exponentiation by squaring.
template Power(n) {
    signal input in;
    signal output out;

    if (n == 0) {
        out <== 1;
    } else if (n == 1) {
        out <== in;
    } else {
        component half = Power(n \ 2);
        half.in <== in;

        signal sq;
        sq <== half.out * half.out;

        if (n % 2 == 1) {
            out <== sq * in;
        } else {
            out <== sq;
        }
    }
}

// RS2_3 encodes the constraint system:
//   x^3 + 5*x - 2 == y
//   y^3 == 3241792
// where x and y are both private witnesses supplied by the prover.
//
// It is based on Chapter 3, Module 2 of the RareSkills Zero-Knowledge book.
// https://rareskills.io/post/r1cs-zkp
template RS2_3() {
    signal input x;
    signal input y;

    component xCubed = Power(3);
    xCubed.in <== x;

    xCubed.out + 5 * x - 2 === y;

    component yCubed = Power(3);
    yCubed.in <== y;

    yCubed.out === 3241792;
}

component main = RS2_3();
