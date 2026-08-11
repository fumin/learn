pragma circom 2.2.3;

include "circomlib/bitify.circom";
include "circomlib/gates.circom";

template BitwiseAnd32() {
    signal input in[2];
    signal output out;

    // range check
    component n2ba = Num2Bits(32);
    component n2bb = Num2Bits(32);
    n2ba.in <== in[0];
    n2bb.in <== in[1];

    component b2n = Bits2Num(32);
    component Ands[32];
    for (var i = 0; i < 32; i++) {
        Ands[i] = AND();
        Ands[i].a <== n2ba.out[i];
        Ands[i].b <== n2bb.out[i];
        Ands[i].out ==> b2n.in[i];
    }

    b2n.out ==> out;
}

template BitwiseOr32() {
    signal input in[2];
    signal output out;

    // range check
    component n2ba = Num2Bits(32);
    component n2bb = Num2Bits(32);
    n2ba.in <== in[0];
    n2bb.in <== in[1];

    component b2n = Bits2Num(32);
    component Ors[32];
    for (var i = 0; i < 32; i++) {
        Ors[i] = OR();
        Ors[i].a <== n2ba.out[i];
        Ors[i].b <== n2bb.out[i];
        Ors[i].out ==> b2n.in[i];
    }

    b2n.out ==> out;
}

template BitwiseXor32() {
    signal input in[2];
    signal output out;

    // range check
    component n2ba = Num2Bits(32);
    component n2bb = Num2Bits(32);
    n2ba.in <== in[0];
    n2bb.in <== in[1];

    component b2n = Bits2Num(32);
    component Xors[32];
    for (var i = 0; i < 32; i++) {
        Xors[i] = XOR();
        Xors[i].a <== n2ba.out[i];
        Xors[i].b <== n2bb.out[i];
        Xors[i].out ==> b2n.in[i];
    }

    b2n.out ==> out;
}

template BitwiseNot32() {
    signal input in;
    signal output out;

    // range check
    component n2ba = Num2Bits(32);
    n2ba.in <== in;

    component b2n = Bits2Num(32);
    component Nots[32];
    for (var i = 0; i < 32; i++) {
        Nots[i] = NOT();
        Nots[i].in <== n2ba.out[i];
        Nots[i].out ==> b2n.in[i];
    }

    b2n.out ==> out;
}

// n is the number of bytes
template ToBytes(n) {
    signal input in;
    signal output out[n];

    component n2b = Num2Bits(n * 8);
    n2b.in <== in;

    component b2ns[n];
    for (var i = 0; i < n; i++) {
        b2ns[i] = Bits2Num(8);
        for (var j = 0; j < 8; j++) {
            b2ns[i].in[j] <== n2b.out[8*i + j];
        }
        out[i] <== b2ns[i].out;
    }
}

// n is the number of bytes
template Padding(n) {
    // 56 bytes = 448 bits
    assert(n < 56);

    signal input in[n];

    // 64 bytes = 512 bits
    signal output out[64];

    for (var i = 0; i < n; i++) {
        out[i] <== in[i];
    }

    // add 128 = 0x80 to pad the 1 bit (0x80 = 10000000b)
    out[n] <== 128;

    // pad the rest with zeros
    for (var i = n + 1; i < 56; i++) {
        out[i] <== 0;
    }

    var lenBits = n * 8;
    if (lenBits < 256) {
        out[56] <== lenBits;
    }
    else {
        var lowOrderBytes = lenBits % 256;
        var highOrderBytes = lenBits \ 256;
        out[56] <== lowOrderBytes;
        out[57] <== highOrderBytes;
    }
}
template Overflow32() {
    signal input in;
    signal output out;

    component n2b = Num2Bits(252);
    component b2n = Bits2Num(32);

    n2b.in <== in;
    for (var i = 0; i < 32; i++) {
        n2b.out[i] ==> b2n.in[i];
    }

    b2n.out ==> out;
}

template LeftRotate(s) {
    signal input in;
    signal output out;

    component n2b = Num2Bits(32);
    component b2n = Bits2Num(32);

    n2b.in <== in;

    for (var i = 0; i < 32; i++) {
        b2n.in[(i + s) % 32] <== n2b.out[i];
    }

    out <== b2n.out;
}

template Func(i) {
    assert(i <= 64);
    signal input b;
    signal input c;
    signal input d;

    signal output out;

    if (i < 16) {
        component a1 = BitwiseAnd32();
        a1.in[0] <== b;
        a1.in[1] <== c;

        component a2 = BitwiseAnd32();
        component n1 = BitwiseNot32();
        n1.in <== b;
        a2.in[0] <== n1.out;
        a2.in[1] <== d;

        component o1 = BitwiseOr32();
        o1.in[0] <== a1.out;
        o1.in[1] <== a2.out;

        out <== o1.out;
    }
    else if (i >= 16 && i < 32) {
        // (D & B) | (~D & C)
        component a1 = BitwiseAnd32();
        a1.in[0] <== d;
        a1.in[1] <== b;

        component n1 = BitwiseNot32();
        n1.in <== d;
        component a2 = BitwiseAnd32();
        a2.in[0] <== n1.out;
        a2.in[1] <== c;

        component o1 = BitwiseOr32();
        o1.in[0] <== a1.out;
        o1.in[1] <== a2.out;

        out <== o1.out;
    }
    else if (i >= 32 && i < 48) {
        component x1 = BitwiseXor32();
        component x2 = BitwiseXor32();

        x1.in[0] <== b;
        x1.in[1] <== c;
        x2.in[0] <== x1.out;
        x2.in[1] <== d;

        out <== x2.out;
    }
    // i must be < 64 by the assert statement above
    else {
        component o1 = BitwiseOr32();
        component n1 = BitwiseNot32();
        n1.in <== d;
        o1.in[0] <== n1.out;
        o1.in[1] <== b;

        component x1 = BitwiseXor32();
        x1.in[0] <== o1.out;
        x1.in[1] <==c;

        out <== x1.out;
    }
}

// n is the number of bytes
template MD5(n) {

    var s[64] = [7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
     5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
     4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21];

    var K[64] = [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
     0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
     0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
     0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
     0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
     0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
     0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
     0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
     0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
     0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
     0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
     0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
     0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
     0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
     0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
     0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391];

    var iter_to_index[64] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12,
     5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2,
    0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9];

    signal input in[n];

    signal inp[64];
    component Pad = Padding(n);

    for (var i = 0; i < n; i++) {
        Pad.in[i] <== in[i];
    }
    for (var i = 0; i < 64; i++) {
        Pad.out[i] ==> inp[i];
    }

    signal data32[16];
    for (var i = 0; i < 16; i++) {
        data32[i] <== inp[4 * i] + inp[4 * i + 1] * 2**8 + inp[4 * i + 2] * 2**16 + inp[4 * i + 3] * 2**24;
    }

    var A = 0;
    var B = 1;
    var C = 2;
    var D = 3;
    signal buffer[65][4];
    buffer[0][A] <== 1732584193;
    buffer[0][B] <== 4023233417;
    buffer[0][C] <== 2562383102;
    buffer[0][D] <== 271733878;

    component Funcs[64];
    signal toRotates[64];
    component SelectInputWords[64];
    component LeftRotates[64];
    component Overflow32s[64];
    component Overflow32s2[64];
    for (var i = 0; i < 64; i++) {
        Funcs[i] = Func(i);
        Funcs[i].b <== buffer[i][B];
        Funcs[i].c <== buffer[i][C];
        Funcs[i].d <== buffer[i][D];
        
        Overflow32s[i] = Overflow32();
        Overflow32s[i].in <== buffer[i][A] + Funcs[i].out + K[i] + data32[iter_to_index[i]];
        
        // rotated = rotate(to_rotate, s[i])
        toRotates[i] <== Overflow32s[i].out;
        LeftRotates[i] = LeftRotate(s[i]);
        LeftRotates[i].in <== toRotates[i];

        // new_B = rotated + B
        Overflow32s2[i] = Overflow32();
        Overflow32s2[i].in <== LeftRotates[i].out + buffer[i][B];

        // store into the next state
        buffer[i + 1][A] <== buffer[i][D];
        buffer[i + 1][B] <== Overflow32s2[i].out;
        buffer[i + 1][C] <== buffer[i][B];
        buffer[i + 1][D] <== buffer[i][C];
    }

    component addA = Overflow32();
    component addB = Overflow32();
    component addC = Overflow32();
    component addD = Overflow32();

    // we hardcode initial state because we only
    // process one 512 bit block
    addA.in <== 1732584193 + buffer[64][A];
    addB.in <== 4023233417 + buffer[64][B];
    addC.in <== 2562383102 + buffer[64][C];
    addD.in <== 271733878 + buffer[64][D];

    signal littleEndianMd5;
    littleEndianMd5 <== addA.out + addB.out * 2**32 + addC.out * 2**64 + addD.out * 2**96;

    // convert the answer to bytes and reverse
    // the bytes order to make it big endian
    component Tb = ToBytes(16);
    Tb.in <== littleEndianMd5;

    // sum the bytes in reverse
    var acc;
    for (var i = 0; i < 16; i++) {
        acc += Tb.out[15 - i] * 2**(i * 8);
    }
    signal output out;
    out <== acc;
}

template MD5PreimageCheck(n) {
    signal input preimage[n];
    signal input image;

    component md5 = MD5(n);
    md5.in <== preimage;

    md5.out === image;
}

component main = MD5PreimageCheck(10);

// The result out = 
// "RareSkills" in ascii to decimal
/* INPUT = {"in": [82, 97, 114, 101, 83, 107, 105, 108, 108, 115]} */

// The result is 246193259845151292174181299259247598493

// The MD5 hash of "RareSkills" is 0xb93718dd21d2f5081239d7a16cf69b9d when converted to decimal is 246193259845151292174181299259247598493
