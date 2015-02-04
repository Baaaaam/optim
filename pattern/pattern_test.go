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

		best, niter, n, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, true)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		} else {
			t.Logf("[pass:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		}
	}
}

func TestHybridNocache(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildHybrid(fn, false, nil)

		best, niter, n, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, true)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		} else {
			t.Logf("[pass:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		}
	}
}

func TestHybridCache(t *testing.T) {
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildHybrid(fn, true, nil)

		best, niter, n, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, true)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		} else {
			t.Logf("[pass:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), n, niter, optimum, best.Val)
		}
	}
}

func buildIter(fn bench.Func, db *sql.DB) optim.Iterator {
	start := initialpoint(fn)
	return NewIterator(nil, start, DB(db))
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

func initialPop(fn bench.Func) pswarm.Population {
	low, up := fn.Bounds()
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 2
		maxv[i] = minv[i] * 2
	}

	n := 30 + 7*len(low)
	if n > maxeval/150 {
		n = maxeval / 150
	}
	points := pop.New(n, low, up)
	return pswarm.NewPopulation(points, minv, maxv)
}

func buildHybrid(fn bench.Func, cache bool, db *sql.DB) optim.Iterator {
	//optim.Rand = rand.New(rand.NewSource(time.Now().Unix()))

	var ev optim.Evaler = optim.SerialEvaler{}
	if cache {
		ev = optim.NewCacheEvaler(optim.SerialEvaler{})
	}
	start := initialpoint(fn)

	pop := initialPop(fn)
	swarm := pswarm.NewIterator(ev, nil, pop,
		//pswarm.KillDist(10000),
		pswarm.VmaxBounds(fn.Bounds()),
		//pswarm.LinInertia(0.9, 0.4, maxeval/len(pop)),
		pswarm.DB(db),
	)
	return NewIterator(ev, start,
		SearchIter(swarm),
		ContinuousSearch,
		DB(db),
	)
}
