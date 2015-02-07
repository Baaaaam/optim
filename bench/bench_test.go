package bench_test

import (
	"database/sql"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	rand2 "bitbucket.org/MaVo159/rand"
	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pattern"
	"github.com/rwcarlsen/optim/pswarm"
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

func TestBenchSwarmRosen(t *testing.T) {
	optim.Rand = rand2.New(rand2.NewMersenneTwister(seed))
	seedrng(seed)

	ndim := 30
	npar := 30
	nrun := 20
	maxiter := 10000

	os.Remove("rosenbench.sqlite")
	db, _ := sql.Open("sqlite3", "rosenbench.sqlite")
	defer db.Close()
	db = nil

	fn := bench.Rosenbrock{ndim}

	nsuccess := 0
	neval := 0
	niter := 0
	sum := 0.0
	for i := 0; i < nrun; i++ {
		it, m := swarmsolver(fn, db, npar)
		solv := &optim.Solver{
			Iter:    it,
			Obj:     optim.Func(fn.Eval),
			Mesh:    m,
			MaxEval: maxiter * npar,
			MaxIter: maxiter,
		}

		for solv.Next() {
			if solv.Best().Val < 100 {
				break
			}
		}
		neval += solv.Neval()
		niter += solv.Niter()
		sum += solv.Best().Val
		if solv.Best().Val < 100 {
			nsuccess++
		}
	}

	t.Logf("[%v] optimum == %v, expect <= 100", fn.Name(), fn.Optima()[0].Val)
	t.Logf("  success rate is %v/%v (%v%%) - averaged %v", nsuccess, nrun, float64(nsuccess)/float64(nrun)*100, sum/float64(nrun))
	t.Logf("  averaged %v iter and %v evals", float64(niter)/float64(nrun), float64(neval)/float64(nrun))
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
		n = 1 * len(low)
		if n > maxeval/500 {
			n = maxeval / 500
		} else if n < 30 {
			n = 30
		}
	}

	//c := 2.01
	//k := pswarm.Constriction(c, c)

	pop := pswarm.NewPopulationRand(n, low, up)
	it := pswarm.NewIterator(nil, pop,
		pswarm.VmaxBounds(fn.Bounds()),
		pswarm.DB(db),
		//pswarm.VelUpdParams(k*c, k*c),
		//pswarm.FixedInertia(k),
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
