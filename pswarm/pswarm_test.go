package pswarm

import (
	"database/sql"
	"testing"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pop"
)

const maxeval = 10000

func TestDb(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	fn := bench.AllFuncs[0]
	optimum := fn.Optima()[0].Val
	it := buildIter(fn, db)

	best, _, neval, err := bench.Benchmark(it, fn, .01, maxeval)
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
	for _, fn := range bench.AllFuncs {
		optimum := fn.Optima()[0].Val
		it := buildIter(fn, nil)

		best, niter, neval, err := bench.Benchmark(it, fn, .01, maxeval)
		if err != nil {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v. %v", fn.Name(), neval, niter, optimum, best.Val, err)
		} else if neval < maxeval {
			t.Logf("[pass:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), neval, niter, optimum, best.Val)
		} else {
			t.Errorf("[FAIL:%v] %v evals (%v iter): optimum is %v, got %v", fn.Name(), neval, niter, optimum, best.Val)
		}
	}
}

func buildIter(fn bench.Func, db *sql.DB) *Iterator {
	low, up := fn.Bounds()
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		minv[i] = (up[i] - low[i]) / 6
		maxv[i] = minv[i] * 1.7
	}

	n := 10 + 7*len(low)
	if n > maxeval/110 {
		n = maxeval / 110
	}

	points := pop.New(n, low, up)
	pop := NewPopulation(points, minv, maxv)
	return NewIterator(nil, nil, pop,
		LinInertia(0.9, 0.4, maxeval/n), DB(db),
	)
}
