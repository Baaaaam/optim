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

const maxiter = 150000

func TestCompass(t *testing.T) {
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

func TestHybridNocache(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildHybrid(fn, false)

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

func TestHybridCache(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildHybrid(fn, true)

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
	max, min := up[0], low[0]

	ev := optim.SerialEvaler{}
	s := pattern.NullSearcher{}
	p := &pattern.CompassPoller{
		Step: (max - min) / 5,
	}

	rand.Seed(time.Now().Unix())
	start := initialpoint(fn.Bounds())
	return pattern.NewIterator(start, ev, p, s)
}

func initialpoint(up, low []float64) optim.Point {
	max, min := up[0], low[0]
	pos := make([]float64, len(low))
	for i := range low {
		pos[i] = rand.Float64()*(max-min) + min
	}
	return optim.NewPoint(pos, math.Inf(1))
}

func buildHybrid(fn bench.Func, cache bool) optim.Iterator {
	rand.Seed(time.Now().Unix())
	start := initialpoint(fn.Bounds())

	low, up := fn.Bounds()
	var ev optim.Evaler = optim.SerialEvaler{}
	if cache {
		ev = optim.NewCacheEvaler(optim.SerialEvaler{})
	}

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

	n := 8 * len(low)
	if n > maxiter/500 {
		n = maxiter / 500
	}
	pop := population.NewRandom(n, low, up, minv, maxv)

	swarmiter := pswarm.SimpleIter{
		Pop:    pop,
		Evaler: ev,
		Mover:  mv,
	}

	// configure pattern solver
	max, min := up[0], low[0]

	s := &pattern.WrapSearcher{Iter: swarmiter}
	p := &pattern.CompassPoller{
		Step: (max - min) / 5,
	}

	pos := make([]float64, len(low))
	for i := range low {
		pos[i] = rand.Float64()*(max-min) + min
	}
	return pattern.NewIterator(start, ev, p, s)
}
