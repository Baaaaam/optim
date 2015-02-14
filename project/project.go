package project

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pattern"
)

func Project(m mesh.Mesh, p optim.Point, l, A, u *mat64.Dense) (best optim.Point, success bool) {
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

		m, _ := ax.Dims()
		penalty := 0.0
		for i := 0; i < m; i++ {
			if diff := ax.At(i, 0) - b.At(i, 0); diff > 0 {
				// maybe use "*=" for compounding penalty buildup
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

	// p is already feasible
	if fn2(m.Nearest(p.Pos())) < math.Inf(1) {
		return p, true
	}

	// solve for an interior point
	s := &optim.Solver{
		Iter:    pattern.NewIterator(nil, p),
		MaxIter: 10000,
		MaxEval: 10000,
		Mesh:    m,
	}

	s.Obj = optim.Func(fn1)
	for s.Next() {
		// stop as soon as we find an interior point
		if s.Best().Val == 0 {
			break
		}
	}
	interior := s.Best()

	// abort if we couldn't find interior point
	if fn2(interior.Pos()) == math.Inf(1) {
		return interior, false
	}

	// dont forget to set interior point val to according to new objective
	// function.
	interior.Val = optim.L2Dist(interior, p)

	// solve for an interior point closest to p
	s = &optim.Solver{
		Iter:    pattern.NewIterator(nil, interior),
		MaxIter: 100000,
		MaxEval: 100000,
		Mesh:    m,
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
