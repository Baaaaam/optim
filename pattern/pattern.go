package pattern

import (
	"errors"

	"github.com/rwcarlsen/optim"
)

var FoundBetterErr = errors.New("better position discovered")

type Point struct {
	Pos []float64
	Val float64
}

type Iterator struct {
	obj *ObjStopper
	ev  optim.Evaler
	p   Poller
	s   Searcher
}

func NewIterator(o optim.Objectiver, e optim.Evaler, p Poller, s Searcher) *Iterator {
	return &Iterator{
		obj: &ObjStopper{Objectiver: o},
		ev:  e,
		p:   p,
		s:   s,
	}
}

func (it *Iterator) Iterate(p Point) (Point, error) {
	success, best, err := it.s.Search(p)
	if err != nil {
		return Point{}, err
	} else if success {
		return best, nil
	}

	it.obj.Best = p.Val
	success, best, err = it.p.Poll(it.obj, it.ev, p)
	if err != nil {
		return Point{}, err
	} else if success {
		return best, nil
	} else {
		return p, nil
	}
}

type Poller interface {
	Poll(obj optim.Objectiver, ev optim.Evaler, from Point) (success bool, best Point, err error)
}

type CompassPoller struct {
	Step     float64
	Direcs   [][]float64
	Expand   float64
	Contract float64
}

func (cp *CompassPoller) Poll(obj optim.Objectiver, ev optim.Evaler, from Point) (success bool, best Point, err error) {
	points := make([][]float64, len(cp.Direcs))
	for i, dir := range cp.Direcs {
		points[i] = make([]float64, len(from.Pos))
		for j, v := range dir {
			points[i][j] = from.Pos[j] + cp.Step*v
		}
	}

	results, err := ev.Eval(obj, points...)
	if err == nil || err == FoundBetterErr {
		for i := range results {
			if results[i] < from.Val {
				cp.Step *= cp.Expand
				return true, Point{Pos: points[i], Val: results[i]}, nil
			}
		}
	} else if err != nil {
		return false, Point{}, err
	}

	cp.Step *= cp.Contract
	return false, from, nil
}

type Searcher interface {
	Search(curr Point) (success bool, best Point, err error)
}

type NullSearcher struct{}

func (_ NullSearcher) Search(curr Point) (success bool, best Point, err error) {
	return false, Point{}, nil
}

type ObjStopper struct {
	Best float64
	optim.Objectiver
}

func (s *ObjStopper) Objective(v []float64) (float64, error) {
	obj, err := s.Objectiver.Objective(v)
	if err != nil {
		return obj, err
	} else if obj < s.Best {
		return obj, FoundBetterErr
	}
	return obj, nil
}
