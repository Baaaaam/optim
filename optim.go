package optim

import (
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rwcarlsen/optim/mesh"
)

type Point struct {
	Pos []float64
	Val float64
}

func hashPoint(p Point) [sha1.Size]byte {
	data := make([]byte, len(p.Pos)*8)
	for i, x := range p.Pos {
		binary.BigEndian.PutUint64(data[i*8:], math.Float64bits(x))
	}
	return sha1.Sum(data)
}

type Iterator interface {
	// Iterate runs a single iteration of a solver and reports the number of
	// function evaluations n and the best point.
	Iterate(obj Objectiver, m mesh.Mesh) (best Point, n int, err error)

	AddPoint(p Point)
}

type Evaler interface {
	// Eval evaluates each point using obj and returns the values and number
	// of function evaluations n.  Unevaluated points should not be returned
	// in the results slice.
	Eval(obj Objectiver, points ...Point) (results []Point, n int, err error)
}

type Objectiver interface {
	// Objective evaluates the variables in v and returns the objective
	// function value.  The objective function must be framed so that lower
	// values are better. If the evaluation fails, positive infinity should be
	// returned along with an error.  Note that it is possible for an error to
	// be returned if the evaulation succeeds.
	Objective(v []float64) (float64, error)
}

type CacheEvaler struct {
	ev    Evaler
	cache map[[sha1.Size]byte]float64
}

func NewCacheEvaler(ev Evaler) *CacheEvaler {
	return &CacheEvaler{
		ev:    ev,
		cache: map[[sha1.Size]byte]float64{},
	}
}

func (ev *CacheEvaler) Eval(obj Objectiver, points ...Point) (results []Point, n int, err error) {
	newp := make([]Point, 0, len(points))
	for _, p := range points {
		if val, ok := ev.cache[hashPoint(p)]; ok {
			results = append(results, Point{Pos: p.Pos, Val: val})
		} else {
			newp = append(newp, p)
		}
	}

	newresults, n, err := ev.ev.Eval(obj, newp...)
	for _, p := range newresults {
		ev.cache[hashPoint(p)] = p.Val
	}
	return append(results, newresults...), n, err
}

type SerialEvaler struct {
	ContinueOnErr bool
}

func (ev SerialEvaler) Eval(obj Objectiver, points ...Point) (results []Point, n int, err error) {
	results = make([]Point, 0, len(points))
	for _, p := range points {
		val, err := obj.Objective(p.Pos)
		results = append(results, Point{Pos: append([]float64{}, p.Pos...), Val: val})
		if err != nil && !ev.ContinueOnErr {
			return results, len(results), err
		}
	}
	return results, len(results), nil
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
