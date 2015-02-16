package project

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pattern"
)

func Project(m mesh.Mesh, p optim.Point, l, A, u *mat64.Dense, interior ...optim.Point) (best optim.Point, success bool) {
	stackA, b, _ := optim.StackConstr(l, A, u)

	prevorigin := m.Origin()
	if prevorigin != nil {
		defer m.SetOrigin(prevorigin)
	}
	m.SetOrigin(p.Pos())

	// This is important because the solver iteration below may modify the
	// step size.
	prevstep := m.Step()
	defer m.SetStep(prevstep)

	fn1 := func(v []float64) float64 {
		ax := &mat64.Dense{}
		x := mat64.NewDense(len(v), 1, v)
		ax.Mul(stackA, x)

		m, n := ax.Dims()
		penalty := 0.0
		for i := 0; i < m; i++ {
			if diff := ax.At(i, 0) - b.At(i, 0); diff > 0 {
				// normalize each constraint violation to the sum of the
				// constraint's coefficients - or 1.0 whichever is larger
				nNonzero := 0.0
				for j := 0; j < n; j++ {
					nNonzero += stackA.At(i, j)
				}
				nNonzero = math.Abs(nNonzero)
				if nNonzero == 0 {
					nNonzero = 1
				}
				penalty += diff / nNonzero
			}
		}
		return penalty
	}

	fn2 := func(v []float64) float64 {
		ax := &mat64.Dense{}
		x := mat64.NewDense(len(v), 1, v)
		ax.Mul(stackA, x)

		m, _ := ax.Dims()
		for i := 0; i < m; i++ {
			if diff := ax.At(i, 0) - b.At(i, 0); diff > 0 {
				return math.Inf(1)
			}
		}

		dist := 0.0
		for i := range v {
			diff := p.At(i) - v[i]
			dist += diff * diff
		}
		return math.Sqrt(dist)
	}

	// p is already feasible
	if fn2(m.Nearest(p.Pos())) < math.Inf(1) {
		return p, true
	}

	it := pattern.NewIterator(nil, p, pattern.NsuccessGrow(2))
	it.Poller = &pattern.CompassPoller{Nkeep: p.Len(), SpanFn: pattern.CompassNp1}

	// solve for an interior point
	s := &optim.Solver{
		Iter:    it,
		MaxIter: 19000,
		MaxEval: 19000,
		Mesh:    m,
	}

	if len(interior) == 0 {
		s.Obj = optim.Func(fn1)
		for s.Next() {
			// stop as soon as we find an interior point
			if s.Best().Val == 0 {
				break
			}
		}
		fmt.Println("niter neval", s.Niter(), s.Neval())
		interior = append(interior, s.Best())
	}

	// abort if we couldn't find interior point
	if fn2(interior[0].Pos()) == math.Inf(1) {
		return interior[0], false
	}

	// dont forget to set interior point val to according to new objective
	// function.
	interior[0].Val = optim.L2Dist(interior[0], p)

	// solve for an interior point closest to p
	s = &optim.Solver{
		Iter:         pattern.NewIterator(nil, interior[0]),
		MaxIter:      3000,
		MaxEval:      3000,
		MaxNoImprove: 100,
		Mesh:         m,
	}

	s.Obj = optim.Func(fn2)
	s.Run()
	projection := s.Best()

	if fn2(projection.Pos()) < math.Inf(1) {
		return projection, true
	} else {
		return projection, false
	}
}
