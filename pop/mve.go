package pop

import (
	"math"

	"github.com/gonum/blas/blas64"
	"github.com/gonum/matrix/mat64"
)

// InteriorPoint computes an interior point of the closed convex space defined
// by the equation "A * x <= b".
func InteriorPoint(A, b *mat64.Dense, tol float64, maxier int) []float64 {
	return nil
}

func newvec(n int) blas64.Vector {
	return mat64.NewVector(n, nil).RawVector()
}

func newveccopy(v blas64.Vector) blas64.Vector {
	data := make([]float64, len(v.Data))
	copy(data, v.Data)
	newv := v
	newv.Data = data
	return newv
}

func newmat(m, n int) blas64.General {
	return mat64.NewDense(m, n, nil).RawMatrix()
}

func newmatcopy(m blas64.General) blas64.General {
	data := make([]float64, len(m.Data))
	copy(data, m.Data)
	newm := m
	newm.Data = data
	return newm
}

//func MaxVolEllipsoid(A blas64.General, b, x0 blas64.Vector, maxiter int, tol float64) (center, rads *mat64.Dense) {
//	m, n := A.Rows, A.Cols
//	x := newvec(m)
//	E2 := mat64.NewDense(n, n, nil).RawMatrix()
//
//	if maxiter <= 0 {
//		maxiter = 50
//	}
//	if tol <= 0 {
//		tol = 1e-4
//	}
//
//	bnrm := blas64.Nrm2(1, b)
//
//	// b - A * x0
//	bmAx0 := newveccopy(b)
//	blas64.Gemv(blas.NoTrans, -1, A, x0, 1, bmAx0)
//
//	for i, val := range bmAx0.Data {
//		if val <= 0 {
//			panic(fmt.Sprintf("x0 is not an interior point - constraint %v violated", i))
//		}
//	}
//
//	lmywork := n
//	if m > n {
//		lmywork = m
//	}
//
//	tmpnm := newmat(n, m)
//	tmpmm := newmat(m, m)
//	tmpnn := newmat(n, n)
//	r1 := newvec(n)
//	r2 := newvec(m)
//	r3 := newvec(m)
//	dy := newvec(m)
//	dyDy := newvec(m)
//	dz := newvec(m)
//	R23 := newvec(m)
//	R3Dy := newvec(m)
//	yz := newvec(m)
//	yh := newvec(m)
//	y2h := newvec(m)
//	G := newmat(m, n)
//	YA := newmat(m, n)
//	T := newmat(m, n)
//	Q := newmat(m, m)
//	YQ := newmat(m, m)
//	YQQY := newmat(m, m)
//	mywork := newvec(lmywork)
//	myworkI := newvec(lmywork) // should be int vec
//	bmAx := newvec(m)
//	nA := newmat(m, n)
//	Adx := newvec(m)
//	nb := newvec(m)
//	h := newvec(m)
//	y := newvec(m)
//	z := newvec(m)
//
//	for i := 0; i < m; i++ {
//		for j := 0; j < n; j++ {
//			nA.Data[i+j*m] = A.Data[i+j*m] / bmAx0.Data[i]
//		}
//		nb.Data[i] = 1
//		bmAx.Data[i] = 1
//		y.Data[i] = 1
//	}
//
//	res := 1.0
//	astep := 0.0
//	for iter := 0; iter < maxiter; iter++ {
//		// flip astep sign and set "bmAx = astep * Adx + bmAx"
//		if iter > 0 {
//			astep *= -1
//			blas64.Axpy(m, astep, Adx, bmAx)
//		}
//
//		// Y is diag matrix using elements from y
//		Y := newmat(m, m)
//		for i := 0; i < m; i++ {
//			Y.Data[i*m+i] = y.Data[i]
//		}
//
//		blas64.Gemm(blas.Trans, blas.NoTrans, 1, nA, Y, 0, tmpnm)
//		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tmpnm, nA, 0, E2)
//	}
//
//	return nil, nil
//}

// Sqrt returns "m^(1/2)".
func Sqrt(m *mat64.Dense) *mat64.Dense {
	ef := mat64.Eigen(mat64.DenseCopyOf(m), 1e-6)
	Vinv, err := mat64.Inverse(ef.V)
	if err != nil {
		panic(err.Error())
	}

	D := ef.D()
	D.Apply(func(r, c int, v float64) float64 {
		if r == c {
			return math.Sqrt(v)
		}
		return 0
	}, D)

	tmpm := mat64.DenseCopyOf(m)
	tmpm.Mul(ef.V, D)
	tmpm.Mul(tmpm, Vinv)
	return tmpm
}

func uofy(A, y *mat64.Dense) *mat64.Dense {
	Y := diag(y)
	Y.Mul(Y, hofy(A, y))
	return Y
}

// diag returns Y = DIAG(y) for column vector y.
func diag(mat *mat64.Dense) *mat64.Dense {
	_, n := mat.Dims()
	Y := mat64.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		Y.Set(i, i, mat.At(i, 0))
	}
	return Y
}

// hofy returns h(y) = h(E(y)) or "hofE(A, eofy(A, y))".
func hofy(A, y *mat64.Dense) *mat64.Dense {
	return hofE(A, eofy(A, y))
}

// hofE returns h(E(y)) = {||Ea1||,||Ea2||,...} where ||x|| is euclid norm and a1
// is row 1 of A as a col vector, a2 is row 2, etc.
func hofE(A, E *mat64.Dense) *mat64.Dense {
	m, _ := A.Dims()
	tmpA := mat64.DenseCopyOf(A)
	tmpA.TCopy(tmpA)
	tmpA.Mul(E, tmpA)

	result := mat64.NewDense(m, 1, nil)
	for i := 0; i < m; i++ {
		col := tmpA.Col(nil, i)
		tot := 0.0
		for _, v := range col {
			tot += v * v
		}
		result.Set(i, 1, math.Sqrt(tot))
	}
	return result
}

// eofy calculates E(y) = transpose(A)*Y*A where Y is DIAG(y).
func eofy(A, y *mat64.Dense) *mat64.Dense {
	Y := diag(y)
	Atrans := mat64.DenseCopyOf(A)
	Atrans.TCopy(Atrans)

	Atrans.Mul(Atrans, Y)
	Atrans.Mul(Atrans, A)
	return Sqrt(Atrans)
}
