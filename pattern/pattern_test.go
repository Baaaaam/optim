package pattern_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/pattern"
)

func TestAckley(t *testing.T) {
	ev := optim.SerialEvaler{}
	s := pattern.NullSearcher{}
	obj := optim.NewObjectivePrinter(optim.SimpleObjectiver(Ackley))
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

	for i := 0; i < 100; i++ {
		point, _, _ = it.Iterate(obj)
	}

	t.Log("BestVal: ", point.Val)
	t.Log("Found at: ", point.Pos)
}

func Ackley(v []float64) float64 {
	x := v[0]
	y := v[1]
	return -20*math.Exp(-0.2*math.Sqrt(0.5*(x*x+y*y))) -
		math.Exp(0.5*(math.Cos(2*math.Pi*x)+math.Cos(2*math.Pi*y))) +
		20 + math.E
}

var (
	sin  = math.Sin
	cos  = math.Cos
	abs  = math.Abs
	exp  = math.Exp
	sqrt = math.Sqrt
)

func CrossTray(v []float64) float64 {
	x := v[0]
	y := v[1]
	return -.0001 * math.Pow(abs(sin(x)*sin(y)*exp(abs(100-sqrt(x*x+y*y)/math.Pi)))+1, 0.1)
}

func Eggholder(v []float64) float64 {
	x := v[0]
	y := v[1]
	return -(y+47)*sin(sqrt(abs(y+x/2+47))) - x*sin(sqrt(abs(x-(y+47))))
}
