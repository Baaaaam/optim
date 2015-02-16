package project

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

func TestProject(t *testing.T) {
	ndim := 500
	low := make([]float64, ndim)
	up := make([]float64, ndim)

	for i := range up {
		up[i] = float64(10 * ndim)
	}

	// angled plane slice equality constraint
	coeffs := make([]float64, ndim)
	for i := range coeffs {
		coeffs[i] = 1
	}

	A := mat64.NewDense(len(coeffs)/ndim, ndim, coeffs)
	l := mat64.NewDense(len(coeffs)/ndim, 1, nil)
	u := mat64.NewDense(len(coeffs)/ndim, 1, nil)
	l.Set(0, 0, 10*float64(ndim))
	u.Set(0, 0, 10*float64(ndim))

	poss := [][]float64{}

	pos := make([]float64, ndim)
	for i := range pos {
		pos[i] = up[len(up)-1]
	}
	poss = append(poss, pos)

	tmp := make([]float64, ndim)
	tmp[0] = 10 * float64(ndim)
	//interior := optim.NewPoint(tmp, math.Inf(1))

	for _, pos := range poss {
		p := optim.NewPoint(pos, math.Inf(1))
		var m mesh.Mesh = &mesh.Infinite{}
		m.SetStep((up[0] - low[0]) / 4)
		m = &mesh.Integer{mesh.NewBounded(m, low, up)}
		//projection, success := Project(m, p, l, A, u, interior)
		projection, success := Project(m, p, l, A, u)
		t.Logf("%v projected to %v (%v)", p, projection, success)
	}
}
