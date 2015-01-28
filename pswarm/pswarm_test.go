package pswarm

import (
	"testing"

	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pop"
)

const maxiter = 50000

func TestSimple(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildIter(fn)

		best, niter, neval, err := bench.Benchmark(it, fn, .01, maxiter)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals: optimum is %v, got %v. %v", fn.Name(), neval, optimum, best.Val, err)
		} else if neval < maxiter {
			t.Logf("[pass:%v] %v evals: optimum is %v, got %v", fn.Name(), neval, optimum, best.Val)
		} else {
			t.Errorf("[FAIL:%v] %v evals: optimum is %v, got %v", fn.Name(), neval, optimum, best.Val)
		}
		t.Log("final inertia was", it.Mover.InertiaFn(niter))
	}
}

func buildIter(fn bench.Func) *Iterator {
	low, up := fn.Bounds()
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 6
		maxv[i] = minv[i] * 1.7
	}

	n := 10 + 7*len(low)
	if n > maxiter/1000 {
		n = maxiter / 1000
	}

	points := pop.New(n, low, up)
	pop := NewPopulation(points, minv, maxv)
	return NewIterator(nil, nil, pop, LinInertia(0.9, 0.4, maxiter/n))
}
