package pattern_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pattern"
	"github.com/rwcarlsen/optim/pswarm"
	"github.com/rwcarlsen/optim/pswarm/population"
)

func TestCompass(t *testing.T) {
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

func TestHybrid(t *testing.T) {
	maxiter := 50000
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildHybrid(fn)

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
	}

	rand.Seed(time.Now().Unix())
	point := optim.Point{Val: math.Inf(1)}
	for _ = range low {
		point.Pos = append(point.Pos, rand.Float64()*(max-min)+min)
	}
	return pattern.NewIterator(point, ev, p, s)
}

func buildHybrid(fn bench.Func) optim.Iterator {
	low, up := fn.Bounds()
	ev := optim.SerialEvaler{}

	// configure pswarm solver
	mv := &pswarm.SimpleMover{
		Cognition: pswarm.DefaultCognition,
		Social:    pswarm.DefaultSocial,
	}

	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 6
		maxv[i] = minv[i] * 1.7
	}

	rand.Seed(time.Now().Unix())
	pop := population.NewRandom(15*len(low), low, up, minv, maxv)

	swarmiter := pswarm.SimpleIter{
		Pop:    pop,
		Evaler: ev,
		Mover:  mv,
	}

	// configure pattern solver
	max, min := up[0], low[0]

	s := &pattern.WrapSearcher{Iter: swarmiter}
	p := &pattern.CompassPoller{
		Step:     (max - min) / 5,
		Expand:   2.0,
		Contract: 0.5,
	}

	rand.Seed(time.Now().Unix())
	point := optim.Point{Val: math.Inf(1)}
	for _ = range low {
		point.Pos = append(point.Pos, rand.Float64()*(max-min)+min)
	}
	return pattern.NewIterator(point, ev, p, s)
}
