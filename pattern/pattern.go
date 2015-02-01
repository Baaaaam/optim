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
		Poller:      &CompassPoller{Nrandom: start.Len() * 2, Nkeep: start.Len()},
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
		it.nfail = 0
		it.curr = best
		m.SetOrigin(best.Pos()) // important to recenter mesh on new best point
		return best, n, nil
	}

	success, best, np, err := it.Poller.Poll(o, it.ev, m, it.curr)
	n += np
	if err != nil {
		return it.curr, n, err
	} else if success {
		it.nsuccess++
		it.nfail = 0
		if it.nsuccess == it.NfailGrow { // == allows -1 to mean never grow
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
		if it.nfail == it.NfailShrink { // == allows -1 to mean never shrink
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
	// Nrandom specifies the number of random-direction chosen points to
	// include in addition to the compass direction points on each poll.
	Nrandom int
	// Nkeep specifies the number of previous successful poll directions to
	// reuse on the next poll. The number of reused directions is min(Nkeep,
	// nsuccessful).
	Nkeep int
	keepdirecs [][]int
}

func direcbetween(from, to optim.Point, m mesh.Mesh) []int {
	d := make([]int, from.Len())
	step := m.Step()
	for i := 0; i < from.Len(); i++ {
		d[i] = int((to.At(i) - from.At(i)) / step)
	}
	return d
}

func (cp *CompassPoller) Poll(obj optim.Objectiver, ev optim.Evaler, m mesh.Mesh, from optim.Point) (success bool, best optim.Point, neval int, err error) {
	best = from

	pollpoints := genPollPoints(from, m)
	pollpoints = append(pollpoints, genRandPollPoints(from, m, cp.Nrandom)...)
	for _, dir := range cp.keepdirecs {
		pollpoints = append(pollpoints, pointFromDirec(from, dir, m))
	}
	cp.keepdirecs = nil

	points := make([]optim.Point, 0, len(pollpoints))
	for _, p := range pollpoints {
		// It is possible that due to the mesh gridding, the poll point is
		// outside of constraints or bounds and will be rounded back to the
		// current point. Check for this and skip the poll point if this is
		// the case.
		dist := optim.L2Dist(from, p)
		eps := 1e-10
		if dist > eps {
			points = append(points, p)
		}
	}

	objstop := &ObjStopper{Objectiver: obj, Best: from.Val}
	results, n, err := ev.Eval(objstop, points...)
	if err != nil && err != FoundBetterErr {
		return false, best, n, err
	}

	for _, p := range results {
		if p.Val < best.Val {
			cp.keepdirecs = append(cp.keepdirecs, direcbetween(from, p, m))
			best = p
		}
	}
	if len(cp.keepdirecs) > cp.Nkeep {
		cp.keepdirecs = cp.keepdirecs[:cp.Nkeep]
	}

	if best.Val < from.Val {
		return true, best, n, nil
	} else {
		return false, from, n, nil
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

func genPollPoints(from optim.Point, m mesh.Mesh) []optim.Point {
	ndim := from.Len()
	polls := make([]optim.Point, 0, 2*ndim)
	for i := 0; i < ndim; i++ {
		d := make([]int, ndim)
		d[i] = 1
		polls = append(polls, pointFromDirec(from, d, m))

		d = make([]int, ndim)
		d[i] = -1
		polls = append(polls, pointFromDirec(from, d, m))
	}
	return polls
}

func pointFromDirec(from optim.Point, direc []int, m mesh.Mesh) optim.Point {
	pos := make([]float64, from.Len())
	for i := range pos {
		pos[i] = from.At(i) + float64(direc[i]) * m.Step()

	}
	p := optim.NewPoint(pos, math.Inf(1))
	return optim.Nearest(p, m)
}

func genRandPollPoints(from optim.Point, m mesh.Mesh, n int) []optim.Point {
	ndim := from.Len()
	polls := make([]optim.Point, 0, n)
	for len(polls) < n {
		d1 := make([]int, ndim)
		d2 := make([]int, ndim)

		hasnonzero := false
		for i := 0; i < ndim; i++ {
			r := optim.Rand.Intn(3) - 1 // r in {-1,0,1}
			d1[i] += r
			d2[i] += -r
			hasnonzero = hasnonzero || (r != 0)
		}
		if hasnonzero {
			polls = append(polls, pointFromDirec(from, d1, m))
			polls = append(polls, pointFromDirec(from, d2, m))
		}
	}
	return polls
}
