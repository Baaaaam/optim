package pswarm

import (
	"database/sql"
	"testing"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim/bench"
)

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
