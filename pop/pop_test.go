package pop

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestNewConstr(t *testing.T) {
	n := 100
	maxiter := 100000
	lb := []float64{0, 0, 0, 0, 0}
	ub := []float64{100, 100, 100, 100, 100}

	// single linear constraint is: x1+x2 <= 10
	// this results in a
	// (10 / 100) * (10 / 100) * 1/2 chance == 0.005
	// that a random point will be feasible
	low := mat64.NewDense(1, 1, []float64{0})
	up := mat64.NewDense(1, 1, []float64{10})
	A := mat64.NewDense(1, 5, []float64{1, 1, 0, 0, 0})
	prob := .005

	points, nbad, iter := NewConstr(n, maxiter, lb, ub, low, A, up)

	if nbad > 0 {
		t.Errorf("got %v bad points", nbad)
	}
	if iter == n {
		t.Errorf("all initial random points were feasible - what?")
	}

	actual := float64(n) / float64(iter)
	diff := (actual - prob) / prob
	if diff < -.05 || diff > 0.5 {
		t.Errorf("expected %v%% of points to be feasible, got %v%%", prob*100, actual*100)
	}

	t.Logf("took %v iterations, %v%% of points were feasible", iter, 100*actual)
	for i, p := range points {
		t.Logf("    point %v: %v", i, p)
	}
}
