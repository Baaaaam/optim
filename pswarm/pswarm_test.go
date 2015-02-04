package pswarm

import (
	"database/sql"
	"testing"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pop"
)

const maxeval = 10000
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

	best, _, neval, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, false)
	if err != nil {
		t.Errorf("[ERROR] %v", err)
	}

	t.Logf("[INFO] %v evals: optimum is %v, got %v", neval, optimum, best.Val)

	var count int
	err = db.QueryRow("SELECT COUNT(*) FROM " + TblParticles).Scan(&count)
	if err != nil {
		t.Errorf("[ERROR] particles table query failed: %v", err)
	} else if count == 0 {
		t.Errorf("[ERROR] particles table has no rows")
	}

	count = 0
	err = db.QueryRow("SELECT COUNT(*) FROM " + TblBest).Scan(&count)
	if err != nil {
		t.Errorf("[ERROR] best table query failed: %v", err)
	} else if count == 0 {
		t.Errorf("[ERROR] best table has no rows")
	}
}

func TestSimple(t *testing.T) {
	//for _, fn := range []bench.Func{bench.Rosenbrock{2}} {
	for _, fn := range bench.AllFuncs {
		//db, _ := sql.Open("sqlite3", fn.Name()+"new.sqlite")
		optimum := fn.Optima()[0].Val
		it := buildIter(fn, nil)

		best, niter, neval, err := bench.Benchmark(it, fn, .01, maxeval, maxiter, false)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), neval, niter, optimum, best.Val)
		} else if neval < maxeval {
			t.Logf("[pass:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), neval, niter, optimum, best.Val)
		}
	}
}

func buildIter(fn bench.Func, db *sql.DB) *Iterator {
	low, up := fn.Bounds()
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 2
		maxv[i] = minv[i] * 2
	}

	n := 30 + 1*len(low)
	if n > maxeval/150 {
		n = maxeval / 150
	}

	points := pop.New(n, low, up)
	pop := NewPopulation(points, minv, maxv)
	return NewIterator(nil, nil, pop,
		VmaxBounds(low, up),
		DB(db),
	)

}
