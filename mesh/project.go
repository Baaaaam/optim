package mesh

import "github.com/gonum/matrix/mat64"

// OrthoProj computes the orthogonal projection of x0 onto the affine subspace
// defined by Ax=b which is the intersection of affine hyperplanes that
// constitute the rows of A with associated shifts in b.  The equation is:
//
//    proj = [I - A^T * (A * A^T)^-1 * A]*x0 + A^T * (A * A^T)^-1 * b
//
// where x0 is the point being projected and I is the identity matrix.  A is
// an m by n matrix where m <= n. if m == n, the returned result is the
// solution to the system A*x0=b
func OrthoProj(x0 []float64, A, b *mat64.Dense) []float64 {
	x := mat64.NewDense(len(x0), 1, x0)

	m, n := A.Dims()
	if m == n {
		proj, err := mat64.Solve(A, b)
		if err != nil {
			panic(err.Error())
		}
		return proj.Col(nil, 0)
	}

	Atrans := &mat64.Dense{}
	Atrans.TCopy(A)

	AAtrans := &mat64.Dense{}
	AAtrans.Mul(A, Atrans)

	// B = A^T * (A*A^T)^-1
	B := &mat64.Dense{}
	inv, err := mat64.Inverse(AAtrans)
	if err != nil {
		panic(err.Error())
	}
	B.Mul(Atrans, inv)

	n, _ = B.Dims()

	tmp := &mat64.Dense{}
	tmp.Mul(B, A)
	tmp.Sub(eye(n), tmp)
	tmp.Mul(tmp, x)

	tmp2 := &mat64.Dense{}
	tmp2.Mul(B, b)
	tmp.Add(tmp, tmp2)

	return tmp.Col(nil, 0)
}

func eye(n int) *mat64.Dense {
	m := mat64.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		m.Set(i, i, 1)
	}
	return m
}

// Nearest returns the nearest point to x0 that doesn't violate constraints in
// the equation Ax <= b.
func Nearest(x0 []float64, A, b *mat64.Dense) []float64 {
	proj := x0
	var badA *mat64.Dense
	var badb *mat64.Dense
	for {
		Aviol, bviol := mostviolated(proj, A, b)

		if Aviol == nil { // projection is complete
			break
		} else {
			if badA == nil {
				badA, badb = Aviol, bviol
			} else {
				tmpA, tmpb := badA, badb
				badA, badb = &mat64.Dense{}, &mat64.Dense{}
				badA.Stack(tmpA, Aviol)
				badb.Stack(tmpb, bviol)
			}
		}

		proj = OrthoProj(x0, badA, badb)

		// we have projected to a single point
		if m, n := badA.Dims(); m == n {
			break
		}
	}
	return proj
}

// mostviolated returns the most violated constraint in the system Ax <= b.
// Aviol and b each have one row and len(x0) columns. It returns nil, nil if
// x0 violates no constraints.
func mostviolated(x0 []float64, A, b *mat64.Dense) (Aviol, bviol *mat64.Dense) {
	eps := 1e-10

	ax := &mat64.Dense{}
	xm := mat64.NewDense(len(x0), 1, x0)
	ax.Mul(A, xm)

	m, _ := ax.Dims()
	worst := eps
	worstRow := -1
	for i := 0; i < m; i++ {
		if diff := ax.At(i, 0) - b.At(i, 0); diff > worst {
			worst = diff
			worstRow = i
		}
	}
	if worstRow == -1 {
		return nil, nil
	}

	return mat64.NewDense(1, len(x0), A.Row(nil, worstRow)), mat64.NewDense(1, 1, b.Row(nil, worstRow))
}
