package bench

import (
	"database/sql"
	"math"
	"math/rand"
	"testing"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pattern"
	"github.com/rwcarlsen/optim/pop"
	"github.com/rwcarlsen/optim/pswarm"
)

const (
	cache   = true
	nocache = false
)

const (
	maxeval      = 50000
	maxiter      = 1000
	maxnoimprove = 200
	minstep      = 1e-6
)

func TestPattern(t *testing.T) {
	for _, fn := range AllFuncs {
		db, _ := sql.Open("sqlite3", fn.Name()+".sqlite")
		it, m := patternsolver(fn, db)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
			MinStep:      minstep,
		}

		Benchmark(t, solv, fn)
	}
}

func TestSwarm(t *testing.T) {
	for _, fn := range AllFuncs {
		it, m := swarmsolver(fn, nil)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
			MinStep:      -1, // needed because swarm doesn't operate on discrete mesh
		}

		Benchmark(t, solv, fn)
	}
}

func TestPSwarm(t *testing.T) {
	for _, fn := range AllFuncs {
		it, m := pswarmsolver(fn, nil, nocache)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
			MinStep:      minstep,
		}

		Benchmark(t, solv, fn)
	}
}

func TestPSwarmCache(t *testing.T) {
	for _, fn := range AllFuncs {
		it, m := pswarmsolver(fn, nil, cache)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      maxiter,
			MaxEval:      maxeval,
			MaxNoImprove: maxnoimprove,
			MinStep:      minstep,
		}

		Benchmark(t, solv, fn)
	}
}

func patternsolver(fn Func, db *sql.DB) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := mesh.NewBounded(&mesh.Infinite{StepSize: (max - min) / 10}, low, up)

	p := initialpoint(fn)
	m.SetOrigin(p.Pos())

	it := pattern.NewIterator(nil, p, pattern.DB(db))
	return it, m
}

func swarmsolver(fn Func, db *sql.DB) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	m := mesh.NewBounded(&mesh.Infinite{StepSize: 0}, low, up)

	n := 30 + 1*len(low)
	if n > maxeval/150 {
		n = maxeval / 150
	}

	pop := initialPop(fn, n)
	it := pswarm.NewIterator(nil, nil, pop,
		pswarm.VmaxBounds(fn.Bounds()),
		pswarm.DB(db),
	)
	return it, m
}

func pswarmsolver(fn Func, db *sql.DB, cache bool) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := mesh.NewBounded(&mesh.Infinite{StepSize: (max - min) / 10}, low, up)

	var ev optim.Evaler = optim.NewCacheEvaler(optim.SerialEvaler{})
	if !cache {
		ev = optim.SerialEvaler{}
	}

	n := 30 + 1*len(low)
	if n > maxeval/150 {
		n = maxeval / 150
	}

	pop := initialPop(fn, n)
	swarm := pswarm.NewIterator(ev, nil, pop,
		pswarm.VmaxBounds(fn.Bounds()),
		pswarm.DB(db),
	)

	p := initialpoint(fn)
	m.SetOrigin(p.Pos())

	it := pattern.NewIterator(ev, p,
		pattern.SearchIter(swarm),
		pattern.ContinuousSearch,
		pattern.DB(db),
	)
	return it, m
}

func initialpoint(fn Func) optim.Point {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	pos := make([]float64, len(low))
	for i := range low {
		pos[i] = rand.Float64()*(max-min) + min
	}
	return optim.NewPoint(pos, math.Inf(1))
}

func initialPop(fn Func, n int) pswarm.Population {
	low, up := fn.Bounds()
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 2
		maxv[i] = minv[i] * 2
	}

	points := pop.New(n, low, up)
	return pswarm.NewPopulation(points, minv, maxv)
}
