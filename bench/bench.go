// Package bench provides tools for testing solvers against benchmark
// optimization functions from
// http://en.wikipedia.org/wiki/Test_functions_for_optimization.
package bench

import (
	"math"

	"github.com/rwcarlsen/optim"
)

var (
	sin  = math.Sin
	cos  = math.Cos
	abs  = math.Abs
	exp  = math.Exp
	sqrt = math.Sqrt
)

var AllFuncs = []Func{
	Ackley{},
	CrossTray{},
	Eggholder{},
	HolderTable{},
}

type Func interface {
	Eval(v []float64) float64
	Bounds() (low, up []float64)
	Optima() []optim.Point
	Name() string
}

type Ackley struct{}

func (fn Ackley) Name() string { return "Ackley" }

func (fn Ackley) Eval(v []float64) float64 {
	if !InsideBounds(v, fn) {
		return math.Inf(1)
	}

	x := v[0]
	y := v[1]
	return -20*math.Exp(-0.2*math.Sqrt(0.5*(x*x+y*y))) -
		math.Exp(0.5*(math.Cos(2*math.Pi*x)+math.Cos(2*math.Pi*y))) +
		20 + math.E
}

func (fn Ackley) Bounds() (low, up []float64) {
	return []float64{-5, -5}, []float64{5, 5}
}

func (fn Ackley) Optima() []optim.Point {
	return []optim.Point{
		optim.Point{Pos: []float64{0, 0}, Val: 0},
	}
}

type CrossTray struct{}

func (fn CrossTray) Name() string { return "CrossTray" }

func (fn CrossTray) Eval(v []float64) float64 {
	if !InsideBounds(v, fn) {
		return math.Inf(1)
	}

	x := v[0]
	y := v[1]
	return -.0001 * math.Pow(abs(sin(x)*sin(y)*exp(abs(100-sqrt(x*x+y*y)/math.Pi)))+1, 0.1)
}

func (fn CrossTray) Bounds() (low, up []float64) {
	return []float64{-10, -10}, []float64{10, 10}
}

func (fn CrossTray) Optima() []optim.Point {
	return []optim.Point{
		optim.Point{Pos: []float64{1.34941, -1.34941}, Val: -2.06261},
		optim.Point{Pos: []float64{1.34941, 1.34941}, Val: -2.06261},
		optim.Point{Pos: []float64{-1.34941, 1.34941}, Val: -2.06261},
		optim.Point{Pos: []float64{-1.34941, -1.34941}, Val: -2.06261},
	}
}

type Eggholder struct{}

func (fn Eggholder) Name() string { return "Eggholder" }

func (fn Eggholder) Eval(v []float64) float64 {
	if !InsideBounds(v, fn) {
		return math.Inf(1)
	}

	x := v[0]
	y := v[1]
	return -(y+47)*sin(sqrt(abs(y+x/2+47))) - x*sin(sqrt(abs(x-(y+47))))
}

func (fn Eggholder) Bounds() (low, up []float64) {
	return []float64{-512, -512}, []float64{512, 512}
}

func (fn Eggholder) Optima() []optim.Point {
	return []optim.Point{
		optim.Point{Pos: []float64{512, 404.2319}, Val: -959.6407},
	}
}

type HolderTable struct{}

func (fn HolderTable) Name() string { return "HolderTable" }

func (fn HolderTable) Eval(v []float64) float64 {
	if !InsideBounds(v, fn) {
		return math.Inf(1)
	}

	x := v[0]
	y := v[1]
	return -abs(sin(x) * cos(y) * exp(abs(1-sqrt(x*x+y*y)/math.Pi)))
}

func (fn HolderTable) Bounds() (low, up []float64) {
	return []float64{-10, -10}, []float64{10, 10}
}

func (fn HolderTable) Optima() []optim.Point {
	return []optim.Point{
		optim.Point{Pos: []float64{8.05502, 9.66459}, Val: -19.2085},
		optim.Point{Pos: []float64{-8.05502, 9.66459}, Val: -19.2085},
		optim.Point{Pos: []float64{8.05502, -9.66459}, Val: -19.2085},
		optim.Point{Pos: []float64{-8.05502, -9.66459}, Val: -19.2085},
	}
}

func Benchmark(it optim.Iterator, fn Func, tol float64, maxeval int) (best optim.Point, neval int, err error) {
	obj := optim.SimpleObjectiver(fn.Eval)
	optimum := fn.Optima()[0].Val
	for neval < maxeval {
		var n int
		best, n, err = it.Iterate(obj)
		neval += n
		if err != nil {
			return optim.Point{}, neval, err
		} else if optimum == 0 && abs(optimum-best.Val) < 1e-10 {
			return best, neval, nil
		} else if optimum != 0 && abs(best.Val-optimum)/abs(optimum) < tol {
			return best, neval, nil
		}
	}
	return best, neval, nil
}

func InsideBounds(p []float64, fn Func) bool {
	low, up := fn.Bounds()
	for i := range p {
		if p[i] < low[i] || p[i] > up[i] {
			return false
		}
	}
	return true
}
