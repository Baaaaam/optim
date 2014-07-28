package pswarm_test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pswarm"
	"github.com/rwcarlsen/optim/pswarm/population"
)

const maxiter = 150000

func TestSimple(t *testing.T) {
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

	rand.Seed(time.Now().Unix())
	n := 8 * len(low)
	if n > maxiter/500 {
		n = maxiter / 500
	}
	pop := population.NewRandom(n, low, up, minv, maxv)

	return pswarm.SimpleIter{
		Pop:    pop,
		Evaler: ev,
		Mover:  mv,
	}
}
