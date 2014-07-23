package pattern

import (
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

type Searcher interface {
	Search() (improved bool, bestpos []float64)
}

type Point struct {
	Pos []float64
	Val float64
}

type Iterator interface {
	Iterate(p Point, ob optim.Objectiver, ev optim.Evaler, poll Poller) (Point, error)
}

type SimpleIter struct{}

func (it *SimpleIter) Iterate(p Point, ob optim.Objectiver, ev optim.Evaler, poll Poller) (Point, error) {
	panic("not implemented")
}

type Poller interface {
	Poll(obj optim.Objectiver, ev optim.Evaler, from Point) (success bool, bestval, bestpos []float64)
}

type SimplePoller struct {
	Mesh  mesh.Mesh
	Basis [][]float64
}
