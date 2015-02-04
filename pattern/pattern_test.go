package pattern

import (
	"database/sql"
	"math"
	"math/rand"
	"testing"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pop"
	"github.com/rwcarlsen/optim/pswarm"
)

const maxeval = 50000
const maxiter = 1000

func TestDb(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	fn := bench.AllFuncs[0]
	optimum := fn.Optima()[0].Val
	it := buildIter(fn, db)

	best, _, neval, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, true)
	t.Logf("[INFO] %v evals: optimum is %v, got %v", neval, optimum, best.Val)

	var count int
	err = db.QueryRow("SELECT COUNT(*) FROM " + TblPolls).Scan(&count)
	if err != nil {
		t.Errorf("[ERROR] particles table query failed: %v", err)
	} else if count == 0 {
		t.Errorf("[ERROR] particles table has no rows")
	}

	count = 0
	err = db.QueryRow("SELECT COUNT(*) FROM " + TblInfo).Scan(&count)
	if err != nil {
		t.Errorf("[ERROR] best table query failed: %v", err)
	} else if count == 0 {
		t.Errorf("[ERROR] best table has no rows")
	}
}

func TestCompass(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildIter(fn, nil)

		best, _, n, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, true)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals: optimum is %v, got %v. %v", fn.Name(), n, optimum, best.Val, err)
		} else if n < maxeval {
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

		best, _, n, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, true)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals: optimum is %v, got %v. %v", fn.Name(), n, optimum, best.Val, err)
		} else if n < maxeval {
			t.Logf("[pass:%v] %v evals: optimum is %v, got %v", fn.Name(), n, optimum, best.Val)
		} else {
			t.Errorf("[FAIL:%v] %v evals: optimum is %v, got %v", fn.Name(), n, optimum, best.Val)
		}
	}
}

func TestHybridCache(t *testing.T) {
	//funcs := []bench.Func{bench.Rosenbrock{30}}
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildHybrid(fn, true)

		best, niter, n, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, true)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v. %v", fn.Name(), n, niter, optimum, best.Val, err)
		} else if n < maxeval {
			t.Logf("[pass:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		} else {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		}
	}
}

func buildIter(fn bench.Func, db *sql.DB) optim.Iterator {
	start := initialpoint(fn.Bounds())
	return NewIterator(nil, start, DB(db))
}

func initialpoint(low, up []float64) optim.Point {
	max, min := up[0], low[0]
	pos := make([]float64, len(low))
	for i := range low {
		pos[i] = rand.Float64()*(max-min) + min
	}
	return optim.NewPoint(pos, math.Inf(1))
}

func buildHybrid(fn bench.Func, cache bool) optim.Iterator {
	//optim.Rand = rand.New(rand.NewSource(time.Now().Unix()))

	start := initialpoint(fn.Bounds())

	low, up := fn.Bounds()
	var ev optim.Evaler = optim.SerialEvaler{}
	if cache {
		ev = optim.NewCacheEvaler(optim.SerialEvaler{})
	}

	// generate initial points
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 10
		maxv[i] = minv[i] * 2
	}

	n := 30 + 3*len(low)
	if n > maxeval/150 {
		n = maxeval / 150
	}
	points := pop.New(n, low, up)

	// configure solver
	pop := pswarm.NewPopulation(points, minv, maxv)
	swarm := pswarm.NewIterator(ev, nil, pop,
		pswarm.LinInertia(0.9, 0.4, maxeval/n),
	)
	return NewIterator(ev, start, SearchIter(swarm),
		ContinuousSearch,
	)
}
