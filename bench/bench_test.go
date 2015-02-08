package bench_test

import (
	"database/sql"
	"math"
	"math/rand"
	"testing"
	"time"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pattern"
	"github.com/rwcarlsen/optim/swarm"
)

const (
	cache   = true
	nocache = false
)

const (
	maxeval      = 50000
	maxiter      = 5000
	maxnoimprove = 500
	minstep      = 1e-8
)

const seed = 7

func seedrng(seed int64) {
	if seed < 0 {
		seed = time.Now().Unix()
	}
	optim.Rand = rand.New(rand.NewSource(seed))
}

func testbench(t *testing.T, fn bench.Func, sfn func() *optim.Solver, successfrac, avgiter float64) {
	t.Logf("[%v] optimum == %v, expect <= %v", fn.Name(), fn.Optima()[0].Val, fn.Tol())

	nrun := 20
	neval := 0
	niter := 0
	nsuccess := 0
	sum := 0.0
	for i := 0; i < nrun; i++ {
		s := sfn()

		for s.Next() {
			if s.Best().Val < fn.Tol() {
				break
			}
		}
		if err := s.Err(); err != nil {
			t.Errorf("    %v", err)
		}

		neval += s.Neval()
		niter += s.Niter()
		sum += s.Best().Val
		if s.Best().Val < fn.Tol() {
			nsuccess++
		}
	}

	frac := float64(nsuccess) / float64(nrun)
	if frac < successfrac {
		t.Errorf("    only found good solutions in %v/%v runs - want >= %v/%v", nsuccess, nrun, math.Ceil(successfrac*float64(nrun)), nrun)
	} else {
		t.Logf("    found good solutions in %v/%v runs", nsuccess, nrun)
	}

	t.Logf("    average best objective is %v", sum/float64(nrun))

	gotavg := float64(niter) / float64(nrun)
	if gotavg > avgiter {
		t.Errorf("    took too many iterations: want %v, averaged %v", avgiter, gotavg)
	} else {
		t.Logf("    averaged %v iterations", gotavg)
	}
}

func TestBenchSwarmRosen(t *testing.T) {
	seedrng(seed)

	ndim := 30
	npar := 30
	maxiter := 10000

	fn := bench.Rosenbrock{ndim}
	sfn := func() *optim.Solver {
		it, m := swarmsolver(fn, nil, npar)
		return &optim.Solver{
			Iter:    it,
			Obj:     optim.Func(fn.Eval),
			Mesh:    m,
			MaxEval: maxiter * npar,
			MaxIter: maxiter,
		}
	}

	successfrac := 0.90
	avgiter := 4000.0
	testbench(t, fn, sfn, successfrac, avgiter)
}

func TestBenchPSwarmRosen(t *testing.T) {
	seedrng(seed)

	ndim := 30
	npar := 30
	maxiter := 10000

	fn := bench.Rosenbrock{ndim}
	sfn := func() *optim.Solver {
		it, m := pswarmsolver(fn, nil, npar, false)
		return &optim.Solver{
			Iter:    it,
			Obj:     optim.Func(fn.Eval),
			Mesh:    m,
			MaxEval: maxiter * npar,
			MaxIter: maxiter,
		}
	}

	successfrac := 0.90
	avgiter := 2500.0
	testbench(t, fn, sfn, successfrac, avgiter)
}
func TestPattern(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		//db, _ := sql.Open("sqlite3", fn.Name()+".sqlite")
		seedrng(seed)
		it, m := patternsolver(fn, nil)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
			MinStep:      minstep,
		}

		bench.Benchmark(t, solv, fn)
	}
}

func TestSwarm(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		seedrng(seed)
		it, m := swarmsolver(fn, nil, -1)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
		}

		bench.Benchmark(t, solv, fn)
	}
}

func TestPSwarm(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		seedrng(seed)
		it, m := pswarmsolver(fn, nil, -1, nocache)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
			MinStep:      minstep,
		}

		bench.Benchmark(t, solv, fn)
	}
}

func TestPSwarmCache(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		seedrng(seed)
		it, m := pswarmsolver(fn, nil, -1, cache)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
			MinStep:      minstep,
		}

		bench.Benchmark(t, solv, fn)
	}
}

func patternsolver(fn bench.Func, db *sql.DB) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := &mesh.Infinite{StepSize: (max - min) / 10}

	p := initialpoint(fn)
	m.SetOrigin(p.Pos())

	it := pattern.NewIterator(nil, p, pattern.DB(db))
	return it, m
}

func swarmsolver(fn bench.Func, db *sql.DB, n int) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	m := &mesh.Infinite{StepSize: 0}

	if n < 0 {
		n = 30 + 1*len(low)
		if n > maxeval/500 {
			n = maxeval / 500
		}
	}

	//c := 2.01
	//k := swarm.Constriction(c, c)

	pop := swarm.NewPopulationRand(n, low, up)
	it := swarm.NewIterator(nil, pop,
		swarm.VmaxBounds(fn.Bounds()),
		swarm.DB(db),
		//swarm.VelUpdParams(k*c, k*c),
		//swarm.FixedInertia(k),
	)
	return it, m
}

func pswarmsolver(fn bench.Func, db *sql.DB, n int, cache bool) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := &mesh.Infinite{StepSize: (max - min) / 10}

	var ev optim.Evaler = optim.NewCacheEvaler(optim.SerialEvaler{})
	if !cache {
		ev = optim.SerialEvaler{}
	}

	swarm, _ := swarmsolver(fn, db, n)

	p := initialpoint(fn)
	m.SetOrigin(p.Pos())

	it := pattern.NewIterator(ev, p,
		pattern.SearchIter(swarm),
		pattern.ContinuousSearch,
		pattern.DB(db),
	)
	return it, m
}

func initialpoint(fn bench.Func) optim.Point {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	pos := make([]float64, len(low))
	for i := range low {
		pos[i] = rand.Float64()*(max-min) + min
	}
	return optim.NewPoint(pos, math.Inf(1))
}
