package pattern_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pattern"
)

func TestSolverBench(t *testing.T) {
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
	low, up := fn.Bounds()
	max, min := up[0], low[0]

	ev := optim.SerialEvaler{}
	s := pattern.NullSearcher{}
	p := &pattern.CompassPoller{
		Step:     (max - min) / 5,
		Expand:   2.0,
		Contract: 0.5,
		NDims:    len(fn.Optima()[0].Pos),
	}

	rand.Seed(time.Now().Unix())
	point := optim.Point{Val: math.Inf(1)}
	for _ = range low {
		point.Pos = append(point.Pos, rand.Float64()*(max-min)+min)
	}
	return pattern.NewIterator(point, ev, p, s)
}
