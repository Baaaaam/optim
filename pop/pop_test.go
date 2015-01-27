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
	tot := 0.0
	for i, p := range points {
		v1, v2 := p.At(0), p.At(1)
		tot += v1 + v2
		t.Logf("    point %v: %v", i, p)
	}

	avg := tot / float64(n) / 2
	t.Logf("avg of (x1+x2/2) == %v", avg)
}

// TestNewConstrBad makes sure that in the event that we cannot supply all
// feasible points within maxiter that NewConstr correctly favors points that
// violate constraints less.
func TestNewConstrBad(t *testing.T) {
	n := 100
	maxiter := 10000
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

	// we should expect about half of the points to be bad - because
	// maxiter == n / prob / 2
	expbad := int(float64(n) * float64(maxiter) / (float64(n) / prob))

	points, nbad, iter := NewConstr(n, maxiter, lb, ub, low, A, up)

	if nbad < expbad-10 || nbad > expbad+10 {
		t.Errorf("got %v bad points, expected ~%v", nbad, expbad)
	} else {
		t.Logf("got %v bad points, expected ~%v", nbad, expbad)
	}
	if iter < maxiter {
		t.Errorf("didn't hid maxiter - why?")
	}

	actual := float64(n-nbad) / float64(iter)
	diff := (actual - prob) / prob
	if diff < -.1 || diff > 0.1 {
		t.Errorf("expected %v%% of points to be feasible, got %.3f%%", prob*100, actual*100)
	}

	t.Logf("took %v iterations, %.3f%% of points were feasible", iter, 100*actual)
	tot := 0.0
	for i, p := range points {
		v1, v2 := p.At(0), p.At(1)
		tot += v1 + v2
		t.Logf("    point %v: %v", i, p)
	}

	avg := tot / float64(n) / 2
	if avg > 10 {
		t.Errorf("avg of (x1+x2/2) == %v, expected <= 10", avg)
	} else {
		t.Logf("avg of (x1+x2/2) == %v, expected <= 10", avg)
	}
}
