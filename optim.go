package optim

import (
	"fmt"

	"github.com/rwcarlsen/optim/mesh"
)

type Point struct {
	Pos []float64
	Val float64
}

type Iterator interface {
	// Iterate runs a single iteration of a solver and reports the number of
	// function evaluations n and the best point.
	Iterate(obj Objectiver, m mesh.Mesh) (best Point, n int, err error)
}

type Evaler interface {
	// Eval evaluates each point using obj and returns the values and number
	// of function evaluations n.  Unevaluated points should not be returned
	// in the results slice.
	Eval(obj Objectiver, points ...Point) (results []Point, err error)
}

type Objectiver interface {
	// Objective evaluates the variables in v and returns the objective
	// function value.  The objective function must be framed so that lower
	// values are better. If the evaluation fails, positive infinity should be
	// returned along with an error.  Note that it is possible for an error to
	// be returned if the evaulation succeeds.
	Objective(v []float64) (float64, error)
}

//type CacheEvaler struct {
//	ev    Evaler
//	cache map[[64]byte]float64
//}
//
//func NewCacheEvaler(ev Evaler, dims int) *CacheEvaler {
//	return &CacheEvaler{
//		ev:    ev,
//		cache: map[[64]byte]float64{},
//	}
//}
//
//func (ev CacheEvaler) Eval(obj Objectiver, points ...[]float64) (vals []float64, n int, err error) {
//	for
//}

type SerialEvaler struct {
	ContinueOnErr bool
}

func (ev SerialEvaler) Eval(obj Objectiver, points ...Point) (results []Point, err error) {
	results = make([]Point, len(points))
	for i, p := range points {
		results[i].Pos = append([]float64{}, p.Pos...)
		results[i].Val, err = obj.Objective(p.Pos)
		if err != nil && !ev.ContinueOnErr {
			return results, err
		}
	}
	return results, nil
}

type SimpleObjectiver func([]float64) float64

func (so SimpleObjectiver) Objective(v []float64) (float64, error) { return so(v), nil }

type ObjectivePrinter struct {
	Objectiver
	Count int
}

func NewObjectivePrinter(obj Objectiver) *ObjectivePrinter {
	return &ObjectivePrinter{Objectiver: obj}
}

func (op *ObjectivePrinter) Objective(v []float64) (float64, error) {
	val, err := op.Objectiver.Objective(v)

	op.Count++
	fmt.Print(op.Count, " ")
	for _, x := range v {
		fmt.Print(x, " ")
	}
	fmt.Println("    ", val)

	return val, err
}
