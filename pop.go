package optim

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim/mesh"
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

// RandPopConstr generates a random population of n feasible points satisfying
// the linear constraints "low <= Ax <= up". lb and ub define lower and upper
// box bounds for the variables.
func RandPopConstr(n int, lb, ub []float64, low, A, up *mat64.Dense) []Point {
	stackA, b, _ := StackConstrBoxed(lb, ub, low, A, up)
	_, ndims := A.Dims()

	points := make([]Point, 0, n)
	for i := 0; i < n; i++ {
		pos := make([]float64, ndims)
		for j := range pos {
			l, u := lb[j], ub[j]
			pos[j] = l + RandFloat()*2*(u-l) - (u-l)/2
		}

		// project onto feasible region
		proj, _ := mesh.Nearest(pos, stackA, b)
		points = append(points, NewPoint(proj, math.Inf(1)))
	}

	return points
}
