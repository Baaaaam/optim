package pattern

import (
	"errors"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

var FoundBetterErr = errors.New("better position discovered")

type Iterator struct {
	ev   optim.Evaler
	p    Poller
	s    Searcher
	curr optim.Point
}

func NewIterator(start optim.Point, e optim.Evaler, p Poller, s Searcher) *Iterator {
	return &Iterator{
		curr: start,
		ev:   e,
		p:    p,
		s:    s,
	}
}

func (it *Iterator) AddPoint(p optim.Point) {
	if p.Val < it.curr.Val {
		it.curr = p
	}
}

func (it *Iterator) Iterate(o optim.Objectiver, m mesh.Mesh) (best optim.Point, n int, err error) {
	obj := &ObjStopper{Objectiver: o}
	success, best, ns, err := it.s.Search(o, it.p.Mesh(), it.curr)
	n += ns
	if err != nil {
		return optim.Point{}, n, err
	} else if success {
		it.curr = best
		return best, n, nil
	}

	obj.Best = it.curr.Val
	success, best, np, err := it.p.Poll(obj, it.ev, it.curr)
	n += np
	if err != nil {
		return optim.Point{}, n, err
	} else if success {
		it.curr = best
		return best, n, nil
	} else {
		return it.curr, n, nil
	}
}

type Poller interface {
	Poll(obj optim.Objectiver, ev optim.Evaler, from optim.Point) (success bool, best optim.Point, neval int, err error)
	StepSize() float64
	Mesh() mesh.Mesh
}

type CompassPoller struct {
	Step     float64
	Expand   float64
	Contract float64
	direcs   [][]float64
	curr     optim.Point
}

func generateDirecs(ndim int) [][]float64 {
	dirs := make([][]float64, 2*ndim)
	for i := 0; i < ndim; i++ {
		dirs[i] = make([]float64, ndim)
		dirs[i][i] = 1
		dirs[ndim+i] = make([]float64, ndim)
		dirs[ndim+i][i] = -1
	}
	return dirs
}

func (cp *CompassPoller) StepSize() float64 { return cp.Step }

func (cp *CompassPoller) Mesh() mesh.Mesh {
	return &mesh.Infinite{
		Origin: cp.curr.Pos,
		Step:   cp.Step,
	}
}

func (cp *CompassPoller) Poll(obj optim.Objectiver, ev optim.Evaler, from optim.Point) (success bool, best optim.Point, neval int, err error) {
	if cp.direcs == nil {
		cp.direcs = generateDirecs(len(from.Pos))
	}
	cp.curr = from

	points := make([]optim.Point, len(cp.direcs))
	for i, dir := range cp.direcs {
		points[i].Pos = make([]float64, len(from.Pos))
		for j, v := range dir {
			points[i].Pos[j] = from.Pos[j] + cp.Step*v
		}
	}

	results, n, err := ev.Eval(obj, points...)

	if err == nil || err == FoundBetterErr {
		for i := range results {
			if results[i].Val < cp.curr.Val {
				cp.curr = results[i]
			}
		}
		if cp.curr.Val < from.Val {
			cp.Step *= cp.Expand
			return true, cp.curr, n, nil
		}
	} else if err != nil {
		return false, optim.Point{}, n, err
	}
	cp.Step *= cp.Contract
	return false, cp.curr, n, nil
}

type Searcher interface {
	Search(o optim.Objectiver, m mesh.Mesh, curr optim.Point) (success bool, best optim.Point, n int, err error)
}

type NullSearcher struct{}

func (_ NullSearcher) Search(o optim.Objectiver, m mesh.Mesh, curr optim.Point) (success bool, best optim.Point, n int, err error) {
	return false, optim.Point{}, 0, nil
}

type WrapSearcher struct {
	Iter optim.Iterator
}

func (s *WrapSearcher) Search(o optim.Objectiver, m mesh.Mesh, curr optim.Point) (success bool, best optim.Point, n int, err error) {
	s.Iter.AddPoint(curr)
	best, n, err = s.Iter.Iterate(o, m)
	if err != nil {
		return false, optim.Point{}, n, err
	}
	if best.Val < curr.Val {
		return true, best, n, nil
	}
	return false, curr, n, nil
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
