"""
Auxiliary functions used in Julia tutorial notebooks.
"""
module TutorialFunctions

using LinearAlgebra
using TensorOperations
using TensorKit
using KrylovKit

#############
# Chapter 1 #
#############

"""
Returns a random complex MPS tensor.

### Arguments

- `D::Int`: bond dimension for MPS.
- `d::Int`: physical dimension for MPS.

### Returns

`A::TensorMap{CartesianSpace}`: normalized MPS tensor with 3 legs, ordered left-bottom-right.
"""
function createMPS(D, d)
    A = Tensor(randn, ComplexF64, ℝ^D ⊗ ℝ^d ⊗ ℝ^D)
    return normalizeMPS(A)
end

"""
Normalize an MPS tensor.

### Arguments

- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.

### Returns

- `Anew::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.

### Complexity

O(D^3) algorithm, D^3 contraction for transfer matrix handle.
"""
function normalizeMPS(A)
    vals, _, _ =
        eigsolve(TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), 1, :LM) do v
            @tensor vout[-1; -2] := A[-1 2 1] * conj(A[-2 2 3]) * v[1; 3]
        end

    Anew = A / sqrt(vals[1])

    return Anew
end

"""
Find left fixed point.

### Arguments

- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.

### Returns

- `l::TensorMap{CartesianSpace, 1, 1}`: left fixed point with 2 legs of dimension (D, D), ordered bottom-top.

### Complexity

O(D^3) algorithm, D^3 contraction for transfer matrix handle.
"""
function leftFixedPoint(A)
    # calculate fixed point
    _, vecs, _ =
        eigsolve(TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), 1, :LM) do v
            @tensor vout[-1; -2] := A[1 2 -2] * conj(A[3 2 -1]) * v[3; 1]
        end
    l = vecs[1]

    # make left fixed point hermitian and positive semidefinite explicitly
    tracel = tr(l)
    l /= (tracel / abs(tracel)) # remove possible phase
    l = (l + l') / 2 # force hermitian

    return l
end

"""
Find right fixed point.

### Arguments

- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.

### Returns

- `r::TensorMap{CartesianSpace, 1, 1}`: right fixed point with 2 legs of dimension (D, D), ordered top-bottom.

### Complexity

O(D^3) algorithm, D^3 contraction for transfer matrix handle.
"""
function rightFixedPoint(A)
    # calculate fixed point
    _, vecs, _ =
        eigsolve(TensorMap(randn, eltype(A), space(A, 1) ← space(A, 1)), 1, :LM) do v
            @tensor vout[-1; -2] := A[-1 2 1] * conj(A[-2 2 3]) * v[1; 3]
        end
    r = vecs[1]

    # make right fixed point hermitian and positive semidefinite explicitly
    tracer = tr(r)
    r /= (tracer / abs(tracer)) # remove possible phase
    r = (r + r') / 2 # force hermitian

    return r
end

"""
Find normalized fixed points.

### Arguments

- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.

### Returns

- `l::TensorMap{CartesianSpace, 1, 1}`: left fixed point with 2 legs of dimension (D, D), bottom-top.
- `r::TensorMap{CartesianSpace, 1, 1}`: right fixed point with 2 legs of dimension (D, D), top-bottom.

### Complexity

O(D^3) algorithm, D^3 contraction for transfer matrix handle.
"""
function fixedPoints(A)
    # find fixed points
    l, r = leftFixedPoint(A), rightFixedPoint(A)

    # calculate trace
    trace = tr(l * r)

    return l / trace, r
end

"""
Transform A to right-orthonormal gauge.

### Arguments

- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.
- `R0::TensorMap{CartesianSpace, 1, 1}`: right gauge tensor with 2 legs of dimension (D, D), initial guess.
- `tol::Float64=1e-14`: convergence criterium, `norm(R - Rnew) < tol`.
- `maxIter::Int=1e5`: maximum amount of iterations.

### Returns

- `R::TensorMap{CartesianSpace, 1, 1}`: right gauge tensor with 2 legs of dimension (D, D), ordered left-right.
- `Ar::TensorMap{CartesianSpace, 1, 2}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, right orthonormal.
"""
function rightOrthonormalize(
    A, R0=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 3)); tol=1e-14, maxIter=1e5
)
    tol = max(tol, 1e-14)
    i = 1

    # Normalize R0
    R0 /= norm(R0)

    # Initialize loop
    @tensor Ai[-1 -2 -3] := A[-1 -2 1] * R0[1; -3]
    R, Ar = rightorth(Ai, (1,), (2, 3); alg=LQpos())
    R /= norm(R)
    convergence = norm(R - R0)

    # Decompose A*R until R converges
    while convergence > tol
        # calculate AR and decompose
        @tensor Ai[-1 -2 -3] = A[-1 -2 1] * R[1; -3]
        Rnew, Ar = rightorth(Ai, (1,), (2, 3); alg=LQpos())

        # normalize new R
        Rnew /= norm(Rnew)

        # calculate convergence criterium
        convergence = norm(Rnew - R)
        R = Rnew

        # check if iterations exceeds maxIter
        if i > maxIter
            println("Warning, right decomposition has not converged ", convergence)
            break
        end
        i += 1
    end

    return R, Ar
end

"""
Transform A to left-orthonormal gauge.

### Arguments

- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.
- `L0::TensorMap{CartesianSpace, 1, 1}`: left gauge tensor with 2 legs of dimension (D, D), initial guess.
- `tol::Float64=1e-14`: convergence criterium, `norm(R - Rnew) < tol`.
- `maxIter::Int=1e5`: maximum amount of iterations.

### Returns

- `L::TensorMap{CartesianSpace, 1, 1}`: left gauge tensor with 2 legs of dimension (D, D), ordered left-right.
- `Al::TensorMap{CartesianSpace, 2, 1}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, left orthonormal.
"""
function leftOrthonormalize(
    A, L0=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 3)); tol=1e-14, maxIter=1e5
)
    tol = max(tol, 1e-14)
    i = 1

    # Normalize L0
    L0 /= norm(L0)

    # Initialize loop
    @tensor Ai[-1 -2 -3] := L0[-1; 1] * A[1 -2 -3]
    Al, L = leftorth(Ai, (1, 2), (3,); alg=QRpos())
    L /= norm(L)
    convergence = norm(L - L0)

    # Decompose L*A until L converges
    while convergence > tol
        # calculate LA and decompose
        @tensor Ai[-1 -2 -3] = L[-1; 1] * A[1 -2 -3]
        Al, Lnew = leftorth(Ai, (1, 2), (3,); alg=QRpos())

        # normalize new L
        Lnew /= norm(Lnew)

        # calculate convergence criterium
        convergence = norm(Lnew - L)
        L = Lnew

        # check if iterations exceeds maxIter
        if i > maxIter
            println("Warning, left decomposition has not converged ", convergence)
            break
        end
        i += 1
    end
    return L, Al
end

"""
Bring MPS tensor into mixed gauge, such that -Al-C- = -C-Ar- = Ac.

### Arguments

- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.

### Returns

- `Al::TensorMap{CartesianSpace, 2, 1}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, left orthonormal.
- `Ac::TensorMap{CartesianSpace, 2, 1}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, center gauge.
- `Ar::TensorMap{CartesianSpace, 1, 2}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, right orthonormal.
- `C::TensorMap{CartesianSpace, 1, 1}`: center gauge tensor with 2 legs of dimension (D, D), ordered left-right, diagonal.

### Complexity

O(D^3) algorithm.
"""
function mixedCanonical(
    A;
    L0=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 3)),
    R0=TensorMap(randn, eltype(A), space(A, 1) ← space(A, 3)),
    tol=1e-14,
    maxIter=1e5,
)
    tol = max(tol, 1e-14)

    # Compute left and right orthonormal forms
    L, Al = leftOrthonormalize(A, L0; tol, maxIter)
    R, Ar = rightOrthonormalize(A, R0; tol, maxIter)

    # center matrix C is matrix multiplication of L and R
    C = L * R

    # singular value decomposition to diagonalize C
    U, C, V = tsvd(C, (1,), (2,))

    # absorb corresponding unitaries in Al and Ar
    @tensor Al[-1 -2; -3] = U'[-1; 1] * Al[1 -2; 2] * U[2; -3]
    @tensor Ar[-1; -2 -3] = V[-1; 1] * Ar[1; -2 2] * V'[2; -3]

    # normalize center matrix
    norm = tr(C * C')
    C /= sqrt(norm)

    # compute center MPS tensor
    @tensor Ac[-1 -2; -3] := Al[-1 -2; 1] * C[1; -3]

    return Al, Ac, Ar, C
end

"""
Calculate the expectation value of a 2-site operator in uniform gauge.

### Arguments

- `O::TensorMap{CartesianSpace, 2, 2}`: two-site operator with 4 legs of dimension (d, d, d, d), ordered topLeft-topRight-bottomLeft-bottomRight.
- `A::TensorMap{CartesianSpace}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.
- `fpts::Tuple=fixedPoints(A)`: left and right fixed points of transfermatrix, normalized.

### Returns

- `o::ComplexF64`: expectation value of `O`.
"""
function expVal2Uniform(O, A, fpts=fixedPoints(A))
    l, r = fpts
    # contract expectation value network
    @tensor o =
        l[6; 1] *
        r[5; 10] *
        A[1 2 3] *
        A[3 4 5] *
        conj(A[6 7 8]) *
        conj(A[8 9 10]) *
        O[2 4; 7 9]

    return o
end

"""
Calculate the expectation value of a 2-site operator in mixed gauge.

### Arguments

- `O::TensorMap{CartesianSpace, 2, 2}`: two-site operator with 4 legs of dimension (d, d, d, d), ordered topLeft-topRight-bottomLeft-bottomRight.
- `Ac::TensorMap{CartesianSpace, 2, 1}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, center gauged.
- `Ar::TensorMap{CartesianSpace, 1, 2}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, right gauged.

### Returns

- `o::ComplexF64`: expectation value of `O`.
"""
function expVal2Mixed(O, Ac, Ar)
    # contract expectation value network
    @tensor o = Ac[4 2; 1] * Ar[1; 3 6] * conj(Ac[4 5; 8]) * conj(Ar[8; 7 6]) * O[2 3; 5 7]

    return o
end

#############
# Chapter 2 #
#############

"""
Find Al and Ar corresponding to Ãc and C̃, according to algorithm 5 in the lecture notes.

### Arguments

- `Ãc::TensorMap{CartesianSpace, 2, 1}`: new guess for center gauge MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right.
- `C̃::TensorMap{CartesianSpace, 1, 1}`: new guess for center gauge tensor with 2 legs of dimension (D, D), ordered left-right.
- `tol::Float64=1e-5`: canonicalization tolerance.

### Returns

- `Al::TensorMap{CartesianSpace, 2, 1}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, left orthonormal.
- `Ar::TensorMap{CartesianSpace, 1, 2}`: MPS tensor with 3 legs of dimension (D, d, D), ordered left-bottom-right, right orthonormal.
- `C::TensorMap{CartesianSpace, 1, 1}`: center gauge tensor with 2 legs of dimension (D, D), ordered left-right.
"""
function minAcC(Ãc, C̃; tol=1e-5)
    tol = max(tol, 1e-14)

    # polar decomposition of Ac
    UlAc, _ = leftorth(Ãc, (1, 2), (3,); alg=Polar())

    # polar decomposition of C
    UlC, _ = leftorth(C̃, (1,), (2,); alg=Polar())

    # construct Al
    Al = UlAc * UlC'

    # find corresponding Ar, C, and Ac through right orthonormalizing Al
    C, Ar = rightOrthonormalize(Al, C̃; tol)
    nrm = tr(C * C')
    C /= sqrt(nrm)
    @tensor Ac[-1 -2; -3] := Al[-1 -2; 1] * C[1; -3]

    return Al, Ac, Ar, C
end

####################
# External patches #
####################

# TODO: add proper overload of KrylovKit.eigsolve(::TensorMap, ...) to TensorKit.jl

# overload KrylovKit.eigsolve for TensorMaps
function KrylovKit.eigsolve(
    t::AbstractTensorMap,
    p1::TensorKit.IndexTuple,
    p2::TensorKit.IndexTuple,
    args...;
    kwargs...,
)
    return eigsolve(permute(t, p1, p2; copy=true), args...; kwargs...)
end

function KrylovKit.eigsolve(
    t::TensorMap,
    howmany::Int=1,
    which::KrylovKit.Selector=:LM,
    T::Type=eltype(t);
    kwargs...,
)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("eigsolve requires domain and codomain to be the same"))
    return eigsolve(x -> t * x, Tensor(randn, T, domain(t)), howmany, which; kwargs...)
end

end
