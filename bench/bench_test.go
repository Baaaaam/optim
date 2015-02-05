package bench_test

import (
	"database/sql"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

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
	maxiter      = 1000
	maxnoimprove = 300
	minstep      = 1e-8
)

//var seed int64 = time.Now().Unix()
var seed int64 = 1

func seedrng(seed int64) {
	if seed < 0 {
		seed = time.Now().Unix()
	}
	optim.Rand = rand.New(rand.NewSource(seed))
}

func TestSwarmBenchRosenbrock(t *testing.T) {
	seedrng(seed)
	fn := bench.Rosenbrock{30}

	n := 50

	os.Remove("rosenbench.sqlite")
	db, _ := sql.Open("sqlite3", "rosenbench.sqlite")
	defer db.Close()
	db = nil

	gots := make([]float64, n)
	nsuccess := 0
	sum := 0.0
	for i := 0; i < n; i++ {
		it, m := swarmsolver(fn, db, 60)
		solv := &optim.Solver{
			Iter:         it,
			Obj:          optim.Func(fn.Eval),
			Mesh:         m,
			MaxIter:      5000,
			MaxEval:      50000,
			MaxNoImprove: 0,
			MinStep:      -1,
		}

		solv.Run()
		gots[i] = solv.Best().Val
		sum += gots[i]
		if gots[i] < 100 {
			nsuccess++
		}
	}

	t.Logf("[%v] optimum == %v, expect <= %v", fn.Name(), fn.Optima()[0].Val, fn.Tol())
	t.Logf("  success rate is %v/%v (%v%%) - averaged %v", nsuccess, n, float64(nsuccess)/float64(n)*100, sum/float64(n))
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
			MinStep:      -1, // needed because swarm doesn't operate on discrete mesh
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
	m := mesh.NewBounded(&mesh.Infinite{StepSize: (max - min) / 10}, low, up)

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

	pop := pswarm.NewPopulationRand(n, low, up)
	it := pswarm.NewIterator(nil, pop,
		pswarm.VmaxBounds(fn.Bounds()),
		pswarm.DB(db),
	)
	return it, m
}

func pswarmsolver(fn bench.Func, db *sql.DB, n int, cache bool) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := mesh.NewBounded(&mesh.Infinite{StepSize: (max - min) / 10}, low, up)

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
