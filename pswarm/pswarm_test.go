package pswarm_test

import (
	"math/rand"
	"testing"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pswarm"
	"github.com/rwcarlsen/optim/pswarm/population"
)

func TestSimple(t *testing.T) {
	maxiter := 50000
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildIter(fn)

		best, n, _ := bench.Benchmark(it, fn, .01, maxiter)
		if n < maxiter {
			t.Logf("[pass:%v] %v evals: optimum is %v, got %v", fn.Name(), n, optimum, best.Val)
		} else {
			t.Errorf("[FAIL:%v] optimum is %v, got %v", fn.Name(), optimum, best.Val)
		}
	}
}

func buildIter(fn bench.Func) optim.Iterator {
	ev := optim.SerialEvaler{}
	mv := &pswarm.SimpleMover{
		Cognition: pswarm.DefaultCognition,
		Social:    pswarm.DefaultSocial,
	}

	low, up := fn.Bounds()
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 6
		maxv[i] = minv[i] * 1.7
	}

	rand.Seed(1)
	pop := population.NewRandom(15*len(low), low, up, minv, maxv)

	return pswarm.SimpleIter{
		Pop:    pop,
		Evaler: ev,
		Mover:  mv,
	}
}
