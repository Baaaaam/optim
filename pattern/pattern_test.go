package pattern

import (
	"database/sql"
	"testing"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
)

func TestDb(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	fn := bench.Basic[0]
	optimum := fn.Optima()[0].Val
	it, m := patternsolver(fn, db)

	solv := &optim.Solver{
		Method:  it,
		Obj:     optim.Func(fn.Eval),
		Mesh:    m,
		MaxIter: 100,
		MinStep: -1,
	}
	solv.Run()

	t.Logf("[INFO] %v evals: want %v, got %v", solv.Neval(), optimum, solv.Best().Val)

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

func patternsolver(fn bench.Func, db *sql.DB) (optim.Method, optim.Mesh) {
	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := optim.NewBounded(&optim.Infinite{StepSize: (max - min) / 10}, low, up)
	p := optim.NewPoint(m.Origin(), 0)
	it := New(p, DB(db))
	return it, m
}
