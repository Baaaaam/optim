package project

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pattern"
)

func Project(p optim.Point, lb, ub []float64, l, A, u *mat64.Dense, interior ...optim.Point) (best optim.Point, success bool) {
	stackA, b, _ := optim.StackConstr(l, A, u)

	fn1 := func(v []float64) float64 {
		ax := &mat64.Dense{}
		x := mat64.NewDense(len(v), 1, v)
		ax.Mul(stackA, x)

		m, _ := ax.Dims()
		penalty := 0.0
		for i := 0; i < m; i++ {
			if diff := ax.At(i, 0) - b.At(i, 0); diff > 0 {
				penalty += diff
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

	// prepare the mesh
	var m mesh.Mesh = &mesh.Infinite{}
	m.SetStep((ub[0] - lb[0]) / 4)
	m = &mesh.Integer{mesh.NewBounded(m, lb, ub)}
	m.SetOrigin(p.Pos())

	// check if p is already feasible - no work to do
	if fn2(m.Nearest(p.Pos())) < math.Inf(1) {
		return p, true
	}

	// solve for an interior point
	it := pattern.NewIterator(nil, p, pattern.NsuccessGrow(1))
	it.Poller = &pattern.CompassPoller{Nkeep: 5, SpanFn: pattern.CompassNp1}
	s := &optim.Solver{
		Iter:    it,
		MaxIter: 20000,
		MaxEval: 20000,
		Mesh:    m,
		Obj:     optim.Func(fn1),
	}

	if len(interior) == 0 {
		for s.Next() {
			// stop as soon as we find an interior point
			if s.Best().Val == 0 {
				break
			}
		}
		interior = append(interior, s.Best())
	}
	fmt.Println("niter neval", s.Niter(), s.Neval())

	// abort if we couldn't find interior point
	if fn2(interior[0].Pos()) == math.Inf(1) {
		return interior[0], false
	}

	// dont forget to set interior point val to according to new objective
	// function.
	interior[0].Val = optim.L2Dist(interior[0], p)
	fmt.Println(interior[0])

	// solve for an interior point closest to p
	m.SetStep((ub[0] - lb[0]) / 4)
	it = pattern.NewIterator(nil, interior[0], pattern.NsuccessGrow(1))
	it.Poller = &pattern.CompassPoller{Nkeep: 5, SpanFn: pattern.CompassNp1}
	s = &optim.Solver{
		Iter:         it,
		MaxIter:      300000,
		MaxEval:      300000,
		MaxNoImprove: 200,
		Mesh:         m,
		Obj:          optim.Func(fn2),
	}
	s.Run()
	fmt.Println("niter neval", s.Niter(), s.Neval())
	projection := s.Best()

	if fn2(projection.Pos()) < math.Inf(1) {
		return projection, true
	} else {
		return projection, false
	}
}
