from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) + const_k, name="C")

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) * const_k, name="C")

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape,
        lambda *i: tvm.max(tvm.const(0, A.dtype), A(*i)), name="C")
    s = tvm.create_schedule(C.op)
    return tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)

def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape,
        lambda *i: B(*i) * tvm.select(A(*i) > 0, tvm.const(1, A.dtype), tvm.const(0, A.dtype)),
        "C")
    s = tvm.create_schedule(C.op)
    return tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    sa = shapeA[::-1] if transposeA else shapeA
    sb = shapeB[::-1] if transposeB else shapeB
    m = sa[0]
    n = sb[1]
    K = sa[1]

    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")
    k = tvm.reduce_axis((0, K), "k")

    if transposeA and transposeB:
        func = lambda x, y: tvm.sum(A[k, x] * B[y, k], axis=k)
    elif not transposeA and transposeB:
        func = lambda x, y: tvm.sum(A[x, k] * B[y, k], axis=k)
    elif transposeA and not transposeB:
        func = lambda x, y: tvm.sum(A[k, x] * B[k, y], axis=k)
    else: # neither.
        func = lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k)

    C = tvm.compute((m, n), func, name="C")
    s = tvm.create_schedule(C.op)

    SPLIT_FACTOR = 16
    xo, xi = s[C].split(C.op.axis[1], factor=SPLIT_FACTOR)

    s[C].reorder(xi, xo, k)
    s[C].vectorize(xi)
    s[C].parallel(xo)

    # print(transposeA, transposeB)
    # print(tvm.lower(s, [A, B, C], simple_mode=True))

    return tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1]) # Same number of channels.
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    X = tvm.placeholder(shapeX, dtype=dtype, name="X")
    Filter = tvm.placeholder(shapeF, dtype=dtype, name="Filter")

    dchannel = tvm.reduce_axis((0, C), "dchannel")
    di = tvm.reduce_axis((0, S), "di")
    dj = tvm.reduce_axis((0, R), "dj")

    out_shape = (shapeX[0], shapeF[0], H - R + 1, W - S + 1)

    Y = tvm.compute(
        out_shape,
        lambda n, f, i, j: tvm.sum(
            X[n, dchannel, i + di, j + dj] * Filter[f, dchannel, di, dj],
            axis=[dchannel, di, dj]),
        name="Y"
        )

    s = tvm.create_schedule(Y.op)
    return tvm.build(s, [X, Filter, Y], tgt, target_host=tgt_host, name=func_name)

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="X")
    j1 = tvm.reduce_axis((0, shape[1]), "j1")

    maxX = tvm.compute((shape[0],),
        lambda i: tvm.max(X[i, j1], axis=j1),
        name="maxX")

    numerator = tvm.compute(shape,
        lambda i, j: tvm.exp(X[i, j] - maxX[i]),
        name="numerator")

    j2 = tvm.reduce_axis((0, shape[1]), "j2")

    denominator = tvm.compute((shape[0],),
        lambda i: tvm.sum(numerator[i, j2], axis=j2),
        name="denominator")

    Y = tvm.compute(shape,
        lambda i, j: numerator[i, j] / denominator[i],
        name="Y")
    s = tvm.create_schedule(Y.op)
    return tvm.build(s, [X, Y], tgt, target_host=tgt_host, name=func_name)

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="X")
    Y_orig = tvm.placeholder(shape, dtype=dtype, name="Y_orig")

    j1 = tvm.reduce_axis((0, shape[1]), "j1")
    j2 = tvm.reduce_axis((0, shape[1]), "j2")

    maxX = tvm.compute((shape[0],),
        lambda i: tvm.max(X[i, j1], axis=j1),
        name="maxX")

    numerator = tvm.compute(shape,
        lambda i, j: tvm.exp(X[i, j] - maxX[i]),
        name="numerator")

    denominator = tvm.compute((shape[0],),
        lambda i: tvm.sum(numerator[i, j2], axis=j2),
        name="denominator")

    m1 = tvm.reduce_axis((0, shape[0]), "m1")
    m2 = tvm.reduce_axis((0, shape[1]), "m2")

    cross_entropy_sum = tvm.compute((1,),
        lambda i: tvm.sum(
            Y_orig[m1, m2] * tvm.log(numerator[m1, m2] / denominator[m1]),
            axis=[m1, m2]),
        name="cross_entropy_sum")

    negated = tvm.compute((1,),
        lambda i: -cross_entropy_sum[i] / shape[0],
        name="negated")


    s = tvm.create_schedule(negated.op)

    # print(tvm.lower(s, [X, Y_orig, negated], simple_mode=True))

    return tvm.build(s, [X, Y_orig, negated], tgt,
        target_host=tgt_host, name=func_name)

def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
