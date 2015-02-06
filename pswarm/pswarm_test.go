package pswarm

import (
	"database/sql"
	"math"
	"testing"

	_ "github.com/mxk/go-sqlite/sqlite3"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/mesh"
)

type fakeRand struct {
	rands []float64
	i     int
}

func (fr *fakeRand) At(i int) float64 { return fr.rands[i] }
func (fr *fakeRand) Float64() float64 { fr.i++; return fr.rands[fr.i-1] }
func (_ *fakeRand) Intn(n int) int    { return n - 1 }
func (_ *fakeRand) Perm(n int) []int {
	p := make([]int, n)
	for i := 0; i < n; i++ {
		p[i] = i
	}
	return p
}

func TestNewPopulationRand(t *testing.T) {
	l, u := -5.0, 5.0
	ndim := 7
	n := 100000

	vmax := (u - l) / 2
	low := []float64{}
	up := []float64{}
	for i := 0; i < ndim; i++ {
		low = append(low, l)
		up = append(up, u)
	}

	pop := NewPopulationRand(n, low, up)

	vtot0 := 0.0
	for i, p := range pop {
		if p.Val != math.Inf(1) && !t.Failed() {
			t.Errorf("particle's initial value is not infinity")
		}
		if p.Best.Val != math.Inf(1) && !t.Failed() {
			t.Errorf("particle's initial best value is not infinity")
		}

		for j, v := range p.Vel {
			if v > vmax || v < -vmax {
				t.Errorf("particle[%v].Vel[%v] outside bounds: %v !< %v !< %v", i, j, -vmax, v, vmax)
			}
			if j == 0 {
				vtot0 += v
			}
		}
	}

	avg := (vtot0 / float64(n))
	if math.Abs(avg) > 0.01*vmax {
		t.Errorf("bad avg vel for 1st dimension: want 0, got %v", .01*vmax, avg)
	} else {
		t.Logf("avg vel for 1st dimension: %v < %v (aka .01*vmax)", math.Abs(avg), .01*vmax)
	}

}

func TestPopulation_Points(t *testing.T) {
	l, u := -5.0, 5.0
	ndim := 2
	n := 100

	low := []float64{}
	up := []float64{}
	for i := 0; i < ndim; i++ {
		low = append(low, l)
		up = append(up, u)
	}

	pop := NewPopulationRand(n, low, up)

	want := 42.0
	pop[0].Val = want
	points := pop.Points()
	points[0].Val = 7
	points[0] = optim.NewPoint([]float64{0}, 13)
	got := pop[0].Val

	if got != want {
		t.Errorf("a modification to a returned point modified the original particle:")
		t.Errorf("    want %v, got %v", want, got)
	}
}

func TestPopulation_Best(t *testing.T) {
	l, u := -5.0, 5.0
	ndim := 2
	n := 100

	low := []float64{}
	up := []float64{}
	for i := 0; i < ndim; i++ {
		low = append(low, l)
		up = append(up, u)
	}

	pop := NewPopulationRand(n, low, up)

	want := 42.0

	pop[4].Val = want
	best := pop.Best()
	if best.Val == want {
		t.Errorf("Best method uses the particles current value instead of the particles best value.")
	}

	pop[4].Best.Val = want
	best = pop.Best()
	if best.Val != want {
		t.Errorf("Best method is broken somehow.")
	}
}

func TestParticle_Move(t *testing.T) {
	vmax := []float64{40, 40, 40}
	fakerng := &fakeRand{[]float64{.314, .739}, 0}

	foo := optim.Rand
	optim.Rand = fakerng
	defer func() { optim.Rand = foo }()

	// define params
	x0 := []float64{1, 2, 5}
	v0 := []float64{1.2, 3.3, 3.7}
	xbest := []float64{2, 3, 11}
	globest := []float64{-7, 9, 2}

	wantpos := make([]float64, len(x0))
	wantvel := make([]float64, len(x0))
	for i := range wantpos {
		wantvel[i] = v0[i]*DefaultInertia + fakerng.At(1)*DefaultSocial*(globest[i]-x0[i]) + fakerng.At(0)*DefaultCognition*(xbest[i]-x0[i])
		wantpos[i] = x0[i] + wantvel[i]
	}

	// initialize and execute
	p := &Particle{
		Point: optim.NewPoint(x0, 42),
		Vel:   v0,
		Best:  optim.NewPoint(xbest, 41),
	}
	glob := optim.NewPoint(globest, 41)

	p.Move(glob, vmax, DefaultInertia, DefaultSocial, DefaultCognition)

	// test
	vel := p.Vel
	pos := p.Pos()
	for i := range pos {
		if math.Abs(pos[i]-wantpos[i]) > 1e-10 {
			t.Errorf("pos[%v]: want %v, got %v", i, wantpos[i], pos[i])
		}
		if math.Abs(vel[i]-wantvel[i]) > 1e-10 {
			t.Errorf("vel[%v]: want %v, got %v", i, wantvel[i], vel[i])
		}
	}
}

func TestDb(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	fn := bench.AllFuncs[0]
	optimum := fn.Optima()[0].Val

	it, m := swarmsolver(fn, db)
	solv := &optim.Solver{
		Iter:    it,
		Obj:     optim.Func(fn.Eval),
		Mesh:    m,
		MaxIter: 100,
		MinStep: -1,
	}
	solv.Run()

	t.Logf("[INFO] %v evals: want %v, got %v", solv.Neval(), optimum, solv.Best().Val)

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

func swarmsolver(fn bench.Func, db *sql.DB) (optim.Iterator, mesh.Mesh) {
	low, up := fn.Bounds()
	m := mesh.NewBounded(&mesh.Infinite{StepSize: 0}, low, up)

	n := 20

	pop := NewPopulationRand(n, low, up)
	it := NewIterator(nil, pop, DB(db))
	return it, m
}
