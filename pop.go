package optim

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/petar/GoLLRB/llrb"
)

// RandPop generates n randomly positioned points in the boxed bounds defined by
// low and up.  The number of dimensions is equal to len(low).  Returned
// points have their values initialized to +infinity.
func RandPop(n int, low, up []float64) []Point {
	if len(low) != len(up) {
		panic("low and up vectors are not same length")
	}

	ndims := len(low)

	points := make([]Point, n)
	for i := 0; i < n; i++ {
		pos := make([]float64, ndims)
		for j := range pos {
			pos[j] = low[j] + RandFloat()*(up[j]-low[j])
		}
		points[i] = NewPoint(pos, math.Inf(1))
	}
	return points
}

// populitem implements llrb.Item interface for use in RandPopConstr
type populitem struct {
	Point
	howbad float64
}

func (p1 populitem) Less(than llrb.Item) bool {
	p2 := than.(populitem)
	return p1.howbad < p2.howbad
}

// NewConstr tries to generate a random population of n feasible points
// satisfying the linear constraints "low <= Ax <= up". lb and ub define lower
// and upper box bounds for the variables.  NewConstr generates random points
// within the box bounds and keeps all feasible points.  It queues up the
// least unfavorable infeasible points in case n feasible ones cannot be found
// within maxiter.  Rand is used for random
// numbers.
func RandPopConstr(n, maxiter int, lb, ub []float64, low, A, up *mat64.Dense) (points []Point, nbad, iter int) {
	stackA, b, ranges := StackConstr(low, A, up)

	_, ndims := A.Dims()

	violaters := llrb.New()
	points = make([]Point, 0, n)
	for i := 0; i < maxiter; i++ {
		// create point
		pos := make([]float64, ndims)
		for j := range pos {
			l, u := lb[j], ub[j]
			pos[j] = l + RandFloat()*(u-l)
		}
		p := NewPoint(pos, math.Inf(1))

		// check for constraint violations
		ax := &mat64.Dense{}
		ax.Mul(stackA, p.Matrix())
		m, _ := ax.Dims()
		howbad := 0.0
		for i := 0; i < m; i++ {
			if diff := ax.At(i, 0) - b.At(i, 0); diff > 0 {
				howbad += diff / ranges[i]
				break
			}
		}

		if howbad == 0 {
			points = append(points, p)
			if len(points) == n {
				return points, 0, i
			}
		} else {
			// add to tree
			violaters.InsertNoReplace(populitem{p, howbad})
			for violaters.Len() > n-len(points) {
				violaters.DeleteMax()
			}
		}
	}

	nbad = n - len(points)
	for len(points) < n {
		p := violaters.DeleteMin().(populitem).Point
		points = append(points, p)
	}

	return points, nbad, maxiter
}
