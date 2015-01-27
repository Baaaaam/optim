package pswarm_test

import (
	"testing"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pop"
	"github.com/rwcarlsen/optim/pswarm"
)

const maxiter = 50000

func TestSimple(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildIter(fn)

		best, n, err := bench.Benchmark(it, fn, .01, maxiter)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals: optimum is %v, got %v. %v", fn.Name(), n, optimum, best.Val, err)
		} else if n < maxiter {
			t.Logf("[pass:%v] %v evals: optimum is %v, got %v", fn.Name(), n, optimum, best.Val)
		} else {
			t.Errorf("[FAIL:%v] %v evals: optimum is %v, got %v", fn.Name(), n, optimum, best.Val)
		}
	}
}

func buildIter(fn bench.Func) optim.Iterator {
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
	pop := pswarm.NewPopulation(points, minv, maxv)

	return pswarm.NewIterator(pop, nil, nil)
}
