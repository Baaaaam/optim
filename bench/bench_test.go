package bench_test

import (
	"database/sql"
	"math"
	"math/rand"
	"testing"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pattern"
	"github.com/rwcarlsen/optim/swarm"
)

const (
	maxeval      = 50000
	maxiter      = 5000
	maxnoimprove = 500
	minstep      = 1e-8
)

const seed = 7

func init() { bench.BenchSeed = seed }

func TestBenchSwarmRosen(t *testing.T) {
	ndim := 30
	npar := 30
	maxiter := 10000
	successfrac := 0.90
	avgiter := 4000.0

	fn := bench.Rosenbrock{ndim}
	sfn := func() *optim.Solver {
		return &optim.Solver{
			Iter:    swarmsolver(fn, nil, npar),
			Obj:     optim.Func(fn.Eval),
			MaxEval: maxiter * npar,
			MaxIter: maxiter,
		}
	}
	bench.Benchmark(t, fn, sfn, successfrac, avgiter)
}

func TestBenchPSwarmRosen(t *testing.T) {
	ndim := 30
	npar := 30
	maxiter := 10000
	successfrac := 0.90
	avgiter := 2500.0

	fn := bench.Rosenbrock{ndim}
	sfn := func() *optim.Solver {
		it, m := pswarmsolver(fn, nil, npar)
		return &optim.Solver{
			Iter:    it,
			Obj:     optim.Func(fn.Eval),
			Mesh:    m,
			MaxEval: maxiter * npar,
			MaxIter: maxiter,
		}
	}
	bench.Benchmark(t, fn, sfn, successfrac, avgiter)
}

func TestOverviewPattern(t *testing.T) {
	maxeval := 50000
	maxiter := 5000
	successfrac := 0.30
	avgiter := 3500.0

	// ONLY test plain pattern search on convex functions
	for _, fn := range []bench.Func{bench.Rosenbrock{NDim: 2}} {
		sfn := func() *optim.Solver {
			it, m := patternsolver(fn, nil)
			return &optim.Solver{
				Iter:    it,
				Obj:     optim.Func(fn.Eval),
				Mesh:    m,
				MaxIter: maxiter,
				MaxEval: maxeval,
			}
		}
		bench.Benchmark(t, fn, sfn, successfrac, avgiter)
	}
}

func TestOverviewSwarm(t *testing.T) {
	maxeval := 50000
	maxiter := 5000
	successfrac := 0.90
	avgiter := 500.0

	for _, fn := range bench.Basic {
		sfn := func() *optim.Solver {
			return &optim.Solver{
				Iter:    swarmsolver(fn, nil, -1),
				Obj:     optim.Func(fn.Eval),
				MaxEval: maxeval,
				MaxIter: maxiter,
			}
		}
		bench.Benchmark(t, fn, sfn, successfrac, avgiter)
	}
}

func TestOverviewPSwarm(t *testing.T) {
	maxeval := 50000
	maxiter := 5000
	successfrac := 0.90
	avgiter := 250.0

	for _, fn := range bench.Basic {
		sfn := func() *optim.Solver {
			it, m := pswarmsolver(fn, nil, -1)
			return &optim.Solver{
				Iter:    it,
				Obj:     optim.Func(fn.Eval),
				Mesh:    m,
				MaxEval: maxeval,
				MaxIter: maxiter,
			}
		}
		bench.Benchmark(t, fn, sfn, successfrac, avgiter)
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

func swarmsolver(fn bench.Func, db *sql.DB, n int, opts ...swarm.Option) optim.Iterator {
	low, up := fn.Bounds()

	if n < 0 {
		n = 30 + 1*len(low)
		if n > maxeval/500 {
			n = maxeval / 500
		}
	}

	opts = append(opts,
		swarm.VmaxBounds(fn.Bounds()),
		swarm.DB(db),
	)

	pop := swarm.NewPopulationRand(n, low, up)
	it := swarm.NewIterator(nil, pop, opts...)
	return it
}

func pswarmsolver(fn bench.Func, db *sql.DB, n int, opts ...pattern.Option) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := &mesh.Infinite{StepSize: (max - min) / 10}

	ev := optim.SerialEvaler{}

	swarm := swarmsolver(fn, db, n)

	p := initialpoint(fn)
	m.SetOrigin(p.Pos())

	opts = append(opts,
		pattern.SearchIter(swarm, pattern.NoShare),
		pattern.ContinuousSearch,
		pattern.DB(db),
	)

	it := pattern.NewIterator(ev, p, opts...)
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
