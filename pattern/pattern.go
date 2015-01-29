package pattern

import (
	"errors"
	"math"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

var FoundBetterErr = errors.New("better position discovered")
var ZeroStepErr = errors.New("poll step size contracted to zero")

type Iterator struct {
	ev               optim.Evaler
	Poller           Poller
	Searcher         Searcher
	curr             optim.Point
	ContinuousSearch bool // true to not project search points onto poll step size mesh
	NfailGrow        int  // number of successive successful polls before growing mesh
	NfailShrink      int  // number of successive failed polls before shrinking mesh
	nsuccess         int  // (internal) number of successive successful polls
	nfail            int  // (internal) number of successive failed polls
}

type Option func(*Iterator)

func NfailGrow(n int) Option {
	return func(it *Iterator) {
		it.NfailGrow = n
	}
}

func NfailShrink(n int) Option {
	return func(it *Iterator) {
		it.NfailShrink = n
	}
}

func SearchIter(it optim.Iterator) Option {
	return func(iter *Iterator) {
		iter.Searcher = &WrapSearcher{Iter: it}
	}
}

func ContinuousSearch(it *Iterator) {
	it.ContinuousSearch = true
}

func NewIterator(e optim.Evaler, start optim.Point, opts ...Option) *Iterator {
	if e == nil {
		e = optim.SerialEvaler{}
	}
	it := &Iterator{
		curr:        start,
		ev:          e,
		Poller:      &CompassPoller{},
		Searcher:    NullSearcher{},
		NfailShrink: 1,
		NfailGrow:   2,
	}

	for _, opt := range opts {
		opt(it)
	}
	return it
}

func (it *Iterator) AddPoint(p optim.Point) {
	if p.Val < it.curr.Val {
		it.curr = p
	}
}

// Iterate mutates m and so for each iteration, the same, mutated m should be
// passed in.
func (it *Iterator) Iterate(o optim.Objectiver, m mesh.Mesh) (best optim.Point, n int, err error) {
	prevstep := m.Step()
	if it.ContinuousSearch {
		m.SetStep(0)
	}
	success, best, ns, err := it.Searcher.Search(o, m, it.curr)
	m.SetStep(prevstep)

	n += ns
	if err != nil {
		return best, n, err
	} else if success {
		it.curr = best
		m.SetOrigin(best.Pos()) // important to recenter mesh on new best point
		return best, n, nil
	}

	obj := &ObjStopper{Objectiver: o, Best: it.curr.Val}
	success, best, np, err := it.Poller.Poll(obj, it.ev, m, it.curr)
	n += np
	if err != nil {
		return it.curr, n, err
	} else if success {
		it.nsuccess++
		it.nfail = 0
		if it.nsuccess >= it.NfailGrow {
			m.SetStep(m.Step() * 2.0)
			it.nsuccess = 0 // reset after resize
		}
		m.SetOrigin(best.Pos()) // important to recenter mesh on new best point
		it.curr = best
		return best, n, nil
	} else {
		it.nsuccess = 0
		it.nfail++
		var err error
		if it.nfail >= it.NfailShrink {
			m.SetStep(m.Step() * 0.5)
			it.nfail = 0 // reset after resize
			if m.Step() == 0 {
				err = ZeroStepErr
			}
		}
		return it.curr, n, err
	}
}

type Poller interface {
	Poll(obj optim.Objectiver, ev optim.Evaler, m mesh.Mesh, from optim.Point) (success bool, best optim.Point, neval int, err error)
}

type CompassPoller struct {
	direcs [][]float64
	curr   optim.Point
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

func (cp *CompassPoller) Poll(obj optim.Objectiver, ev optim.Evaler, m mesh.Mesh, from optim.Point) (success bool, best optim.Point, neval int, err error) {
	if cp.direcs == nil {
		cp.direcs = generateDirecs(from.Len())
	}
	cp.curr = from

	points := make([]optim.Point, 0, len(cp.direcs))
	for _, dir := range cp.direcs {
		pos := make([]float64, len(dir))
		for j, v := range dir {
			pos[j] = from.At(j) + m.Step()*v
		}
		p := optim.NewPoint(pos, math.Inf(1))

		// It is possible that due to the mesh gridding, the poll point is
		// outside of constraints or bounds and will be rounded back to the
		// current point. Check for this and skip the poll point if this is
		// the case.
		nearestp := optim.Nearest(p, m)
		dist := optim.L2Dist(from, nearestp)
		eps := 1e-5
		if dist > eps {
			points = append(points, nearestp)
		}
	}

	results, n, err := ev.Eval(obj, points...)

	if err == nil || err == FoundBetterErr {
		err = nil
		for i := range results {
			if results[i].Val < cp.curr.Val {
				cp.curr = results[i]
			}
		}
		if cp.curr.Val < from.Val {
			return true, cp.curr, n, nil
		}
		return false, cp.curr, n, nil
	} else {
		return false, cp.curr, n, err
	}
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

// ObjStopper is wraps an Objectiver and returns the objective value along
// with FoundBetterErr as soon as calculates a value better than Best.  This
// is useful for things like terminating early with opportunistic polling.
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
