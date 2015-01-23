// Package bench provides tools for testing solvers against benchmark
// optimization functions from
// http://en.wikipedia.org/wiki/Test_functions_for_optimization.
package bench

import (
	"fmt"
	"math"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
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
	Schaffer2{},
	Styblinski{NDim: 1},
	Styblinski{NDim: 10},
	Styblinski{NDim: 100},
	Styblinski{NDim: 500},
	Rosenbrock{NDim: 2},
	Rosenbrock{NDim: 10},
	Rosenbrock{NDim: 100},
	Rosenbrock{NDim: 500},
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
		optim.NewPoint([]float64{0, 0}, 0),
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
		optim.NewPoint([]float64{1.34941, -1.34941}, -2.06261),
		optim.NewPoint([]float64{1.34941, 1.34941}, -2.06261),
		optim.NewPoint([]float64{-1.34941, 1.34941}, -2.06261),
		optim.NewPoint([]float64{-1.34941, -1.34941}, -2.06261),
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
		optim.NewPoint([]float64{512, 404.2319}, -959.6407),
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
		optim.NewPoint([]float64{8.05502, 9.66459}, -19.2085),
		optim.NewPoint([]float64{-8.05502, 9.66459}, -19.2085),
		optim.NewPoint([]float64{8.05502, -9.66459}, -19.2085),
		optim.NewPoint([]float64{-8.05502, -9.66459}, -19.2085),
	}
}

type Schaffer2 struct{}

func (fn Schaffer2) Name() string { return "Schaffer2" }

func (fn Schaffer2) Eval(v []float64) float64 {
	if !InsideBounds(v, fn) {
		return math.Inf(1)
	}

	x := v[0]
	y := v[1]
	return 0.5 + (math.Pow(sin(x*x-y*y), 2)-0.5)/math.Pow(1+.0001*(x*x+y*y), 2)
}

func (fn Schaffer2) Bounds() (low, up []float64) {
	return []float64{-100, -100}, []float64{100, 100}
}

func (fn Schaffer2) Optima() []optim.Point {
	return []optim.Point{
		optim.NewPoint([]float64{0, 0}, 0),
	}
}

type Styblinski struct {
	NDim int
}

func (fn Styblinski) Name() string { return fmt.Sprintf("Styblinski_%vD", fn.NDim) }

func (fn Styblinski) Eval(x []float64) float64 {
	if !InsideBounds(x, fn) {
		return math.Inf(1)
	}

	tot := 0.0
	for _, v := range x {
		tot += math.Pow(v, 4) - 16*math.Pow(v, 2) + 5*v
	}
	return tot / 2
}

func (fn Styblinski) Bounds() (low, up []float64) {
	low = make([]float64, fn.NDim)
	up = make([]float64, fn.NDim)
	for i := range low {
		low[i] = -5
		up[i] = 5
	}
	return low, up
}

func (fn Styblinski) Optima() []optim.Point {
	pos := make([]float64, fn.NDim)
	for i := range pos {
		pos[i] = -2.903534
	}
	return []optim.Point{
		optim.NewPoint(pos, -39.16599*float64(fn.NDim)),
	}
}

type Rosenbrock struct {
	NDim int
}

func (fn Rosenbrock) Name() string { return fmt.Sprintf("Rosenbrock_%vD", fn.NDim) }

func (fn Rosenbrock) Eval(x []float64) float64 {
	if !InsideBounds(x, fn) {
		return math.Inf(1)
	}

	tot := 0.0
	for i := 0; i < fn.NDim-1; i++ {
		tot += 100*math.Pow(x[i+1]-x[i]*x[i], 2) + math.Pow(x[i]-1, 2)
	}
	return tot
}

func (fn Rosenbrock) Bounds() (low, up []float64) {
	low = make([]float64, fn.NDim)
	up = make([]float64, fn.NDim)
	for i := range low {
		low[i] = -1000
		up[i] = 1000
	}
	return low, up
}

func (fn Rosenbrock) Optima() []optim.Point {
	pos := make([]float64, fn.NDim)
	for i := range pos {
		pos[i] = 1
	}
	return []optim.Point{
		optim.NewPoint(pos, 0),
	}
}

func Benchmark(it optim.Iterator, fn Func, tol float64, maxeval int) (best optim.Point, neval int, err error) {
	obj := optim.SimpleObjectiver(fn.Eval)
	optimum := fn.Optima()[0].Val
	thresh := tol * abs(optimum)
	if 0.001 > thresh {
		thresh = 0.001
	}

	low, up := fn.Bounds()
	max, min := up[0], low[0]
	m := mesh.NewBounded(&mesh.Infinite{StepSize: (max - min) / 5}, low, up)

	for neval < maxeval {
		var n int
		best, n, err = it.Iterate(obj, m)
		neval += n
		if err != nil {
			return best, neval, err
		} else if abs(optimum-best.Val) < thresh {
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
