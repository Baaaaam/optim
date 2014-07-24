package pattern_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pattern"
)

func TestAckley(t *testing.T) {
	// initialize iterator
	ev := optim.SerialEvaler{}
	s := pattern.NullSearcher{}
	p := &pattern.CompassPoller{
		Step:     1.0,
		Expand:   2.0,
		Contract: 0.5,
		Direcs: [][]float64{
			[]float64{1, 0},
			[]float64{-1, 0},
			[]float64{0, 1},
			[]float64{0, -1},
		},
	}

	rand.Seed(time.Now().Unix())
	x := (rand.Float64()*2 - 1.0) * 10
	y := (rand.Float64()*2 - 1.0) * 10
	point := optim.Point{Pos: []float64{x, y}, Val: math.Inf(1)}

	it := pattern.NewIterator(point, ev, p, s)

	// run benchmark
	fn := bench.Ackley{}
	optimum := fn.Optima()[0].Val
	best, n, _ := bench.Benchmark(it, bench.Ackley{}, .01, 10000)
	if math.Abs(best.Val-optimum)/optimum >= 0.1 {
		t.Errorf("Failed to converge: optimum is %v, got %v", optimum, best.Val)
	} else {
		t.Errorf("Success (%v evals): optimum is %v, got %v", n, optimum, best.Val)
	}
}
