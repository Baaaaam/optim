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
	maxiter := 10000
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		low, up := fn.Bounds()
		it := buildIter((up[0]-low[0])/5, low[0], up[0])

		best, n, _ := bench.Benchmark(it, fn, .01, maxiter)
		if n < maxiter {
			t.Logf("[%v:pass] %v evals: optimum is %v, got %v", fn.Name(), n, optimum, best.Val)
		} else {
			t.Errorf("[%v:FAIL] optimum is %v, got %v", fn.Name(), optimum, best.Val)
		}
	}
}

func buildIter(step, max, min float64) optim.Iterator {
	ev := optim.SerialEvaler{}
	s := pattern.NullSearcher{}
	p := &pattern.CompassPoller{
		Step:     step,
		Expand:   2.0,
		Contract: 0.5,
		Direcs: [][]float64{
			[]float64{1, 0},
			[]float64{-1, 0},
			[]float64{0, 1},
			[]float64{0, -1},
		},
	}

	rand.Seed(time.Now().Unix())
	x := rand.Float64()*(max-min) + min
	y := rand.Float64()*(max-min) + min
	point := optim.Point{Pos: []float64{x, y}, Val: math.Inf(1)}
	return pattern.NewIterator(point, ev, p, s)
}
