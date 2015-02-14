package project

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

func TestProject(t *testing.T) {
	ndim := 30
	low := make([]float64, ndim)
	up := make([]float64, ndim)

	for i := range up {
		up[i] = float64(10 * ndim)
	}

	// box bound constraint coeffs
	coeffs := make([]float64, ndim*ndim)
	for i := range low {
		coeffs[ndim*i+i] = 1
	}

	// angled plane slice equality constraint
	plane := make([]float64, ndim)
	for i := range low {
		plane[i] = 1
	}
	coeffs = append(coeffs, plane...)
	low = append(low, float64(10*ndim))
	up = append(up, float64(10*ndim))

	l := mat64.NewDense(len(low), 1, low)
	u := mat64.NewDense(len(up), 1, up)
	A := mat64.NewDense(len(coeffs)/ndim, ndim, coeffs)

	poss := [][]float64{}

	pos := make([]float64, ndim)
	for i := range pos {
		pos[i] = up[len(up)-1] * 2
	}
	poss = append(poss, pos)

	for _, pos := range poss {
		p := optim.NewPoint(pos, math.Inf(1))
		var m mesh.Mesh = &mesh.Infinite{}
		m.SetStep((up[0] - low[0]) / 10)
		m = &mesh.Integer{m}
		projection, success := Project(m, p, l, A, u)
		t.Logf("%v projected to %v (%v)", p, projection, success)
	}
}
