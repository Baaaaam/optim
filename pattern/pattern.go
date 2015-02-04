package pattern

import (
	"crypto/sha1"
	"database/sql"
	"errors"
	"fmt"
	"math"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

var FoundBetterErr = errors.New("better position discovered")
var ZeroStepErr = errors.New("poll step size contracted to zero")

const (
	TblPolls = "patternpolls"
	TblInfo  = "patterninfo"
)

type Iterator struct {
	ev               optim.Evaler
	Poller           Poller
	Searcher         Searcher
	curr             optim.Point
	ContinuousSearch bool // true to not project search points onto poll step size mesh
	NsuccessGrow     int  // number of successive successful polls before growing mesh
	NfailShrink      int  // number of successive failed polls before shrinking mesh
	nsuccess         int  // (internal) number of successive successful polls
	nfail            int  // (internal) number of successive failed polls
	Db               *sql.DB
	count            int
}

type Option func(*Iterator)

func NsuccessGrow(n int) Option {
	return func(it *Iterator) {
		it.NsuccessGrow = n
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

func DB(db *sql.DB) Option {
	return func(it *Iterator) {
		it.Db = db
	}
}

func NewIterator(e optim.Evaler, start optim.Point, opts ...Option) *Iterator {
	if e == nil {
		e = optim.SerialEvaler{}
	}
	it := &Iterator{
		curr:         start,
		ev:           e,
		Poller:       &CompassPoller{Nrandom: start.Len() * 2, Nkeep: start.Len()},
		Searcher:     NullSearcher{},
		NfailShrink:  1,
		NsuccessGrow: 2,
	}

	for _, opt := range opts {
		opt(it)
	}
	it.initdb()
	return it
}

func (it *Iterator) initdb() {
	if it.Db == nil {
		return
	}

	s := "CREATE TABLE IF NOT EXISTS " + TblPolls + " (iter INTEGER,val REAL"
	s += it.xdbsql("define")
	s += ");"

	_, err := it.Db.Exec(s)
	panicif(err)

	s = "CREATE TABLE IF NOT EXISTS " + TblInfo + " (iter INTEGER,step INTEGER,nsearch INTEGER,npoll INTEGER,val REAL"
	s += it.xdbsql("define")
	s += ");"
	_, err = it.Db.Exec(s)
	panicif(err)
}

func (it Iterator) xdbsql(op string) string {
	s := ""
	for i := range it.curr.Pos() {
		if op == "?" {
			s += ",?"
		} else if op == "define" {
			s += fmt.Sprintf(",x%v REAL", i)
		} else if op == "x" {
			s += fmt.Sprintf(",x%v", i)
		} else {
			panic("invalid db op " + op)
		}
	}
	return s
}

func (it Iterator) updateDb(nsearch, npoll *int, step float64) {
	if it.Db == nil {
		return
	}

	tx, err := it.Db.Begin()
	if err != nil {
		panic(err.Error())
	}
	defer tx.Commit()

	s1 := "INSERT INTO " + TblPolls + " (iter,val" + it.xdbsql("x") + ") VALUES (?,?" + it.xdbsql("?") + ");"
	for _, p := range it.Poller.Points() {
		args := []interface{}{it.count, p.Val}
		args = append(args, pos2iface(p.Pos())...)
		_, err := tx.Exec(s1, args...)
		panicif(err)
	}

	s2 := "INSERT INTO " + TblInfo + " (iter,step,nsearch, npoll,val" + it.xdbsql("x") + ") VALUES (?,?,?,?,?" + it.xdbsql("?") + ");"
	glob := it.curr
	args := []interface{}{it.count, step, *nsearch, *npoll, glob.Val}
	args = append(args, pos2iface(glob.Pos())...)
	_, err = tx.Exec(s2, args...)
	panicif(err)
}

func (it *Iterator) AddPoint(p optim.Point) {
	if p.Val < it.curr.Val {
		it.curr = p
	}
}

// Iterate mutates m and so for each iteration, the same, mutated m should be
// passed in.
func (it *Iterator) Iterate(o optim.Objectiver, m mesh.Mesh) (best optim.Point, n int, err error) {
	var nevalsearch, nevalpoll int
	var success bool
	defer it.updateDb(&nevalsearch, &nevalpoll, m.Step())
	it.count++

	prevstep := m.Step()
	if it.ContinuousSearch {
		m.SetStep(0)
	}
	success, best, nevalsearch, err = it.Searcher.Search(o, m, it.curr)
	m.SetStep(prevstep)

	n += nevalsearch
	if err != nil {
		return best, n, err
	} else if success {
		it.nfail = 0
		it.curr = best
		m.SetOrigin(best.Pos()) // important to recenter mesh on new best point
		return best, n, nil
	}

	success, best, nevalpoll, err = it.Poller.Poll(o, it.ev, m, it.curr)
	n += nevalpoll
	if err != nil {
		return it.curr, n, err
	} else if success {
		it.nsuccess++
		it.nfail = 0
		if it.nsuccess == it.NsuccessGrow { // == allows -1 to mean never grow
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
	// Poll polls on mesh m centered on point from.  It is responsible for
	// selecting points and evaluating them with ev using obj.  If a better
	// point was found, it returns success == true, the point, and number of
	// evaluations.  If a better point was not found, it returns false, the
	// from point, and the number of evaluations.  If err is non-nil, success
	// must be false and best must be from - neval may be non-zero.
	Poll(obj optim.Objectiver, ev optim.Evaler, m mesh.Mesh, from optim.Point) (success bool, best optim.Point, neval int, err error)
	// Points returns the points that were checked on the most recent poll
	Points() []optim.Point
}

type CompassPoller struct {
	// Nrandom specifies the number of random-direction chosen points to
	// include in addition to the compass direction points on each poll.
	Nrandom int
	// Nkeep specifies the number of previous successful poll directions to
	// reuse on the next poll. The number of reused directions is min(Nkeep,
	// nsuccessful).
	Nkeep      int
	keepdirecs [][]int
	points     []optim.Point
	prevhash   [sha1.Size]byte
	prevstep   float64
}

func (cp *CompassPoller) Points() []optim.Point { return cp.points }

func (cp *CompassPoller) Poll(obj optim.Objectiver, ev optim.Evaler, m mesh.Mesh, from optim.Point) (success bool, best optim.Point, neval int, err error) {
	best = from
	pollpoints := []optim.Point{}

	// Only poll compass directions if we haven't polled from this point
	// before.
	h := from.Hash()
	if h != cp.prevhash || cp.prevstep != m.Step() {
		// TODO: write test that checks we poll compass dirs again if only mesh
		// step changed (and not from point)
		pollpoints = append(pollpoints, genPollPoints(from, m)...)
		cp.prevhash = h
	} else {
		// Use random directions instead.
		pollpoints = append(pollpoints, genRandPollPoints(from, m, 2*from.Len())...)
	}
	cp.prevstep = m.Step()

	pollpoints = append(pollpoints, genRandPollPoints(from, m, cp.Nrandom)...)

	// Add successful directions from last poll.
	for _, dir := range cp.keepdirecs {
		pollpoints = append(pollpoints, pointFromDirec(from, dir, m))
	}
	cp.keepdirecs = nil

	cp.points = make([]optim.Point, 0, len(pollpoints))
	for _, p := range pollpoints {
		// It is possible that due to the mesh gridding, the poll point is
		// outside of constraints or bounds and will be rounded back to the
		// current point. Check for this and skip the poll point if this is
		// the case.
		dist := optim.L2Dist(from, p)
		eps := 1e-10
		if dist > eps {
			cp.points = append(cp.points, p)
		}
	}
	cp.points = pollpoints

	objstop := &ObjStopper{Objectiver: obj, Best: from.Val}
	results, n, err := ev.Eval(objstop, cp.points...)
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
		pos[i] = from.At(i) + float64(direc[i])*m.Step()

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

		nNonzero := 1
		if ndim == 1 { // compass directions cover everything
			return polls
		} else if ndim == 2 { // this check prevents calling Intn(0) - which is invalid
			nNonzero = 2 // exclude compass directions
		} else {
			// Intn(-2)+2 is to exclude vector of all zeros and compass directions.
			nNonzero = optim.Rand.Intn(ndim-2) + 2
		}
		perms := optim.Rand.Perm(ndim)
		for i := 0; i < nNonzero; i++ {
			r := optim.Rand.Intn(2)
			if r == 0 {
				d1[perms[i]] = 1
				d2[perms[i]] = -1
			} else {
				d1[perms[i]] = -1
				d2[perms[i]] = 1
			}
		}
		polls = append(polls, pointFromDirec(from, d1, m))
		polls = append(polls, pointFromDirec(from, d2, m))
	}
	return polls
}

func pos2iface(pos []float64) []interface{} {
	iface := []interface{}{}
	for _, v := range pos {
		iface = append(iface, v)
	}
	return iface
}

func direcbetween(from, to optim.Point, m mesh.Mesh) []int {
	d := make([]int, from.Len())
	step := m.Step()
	for i := 0; i < from.Len(); i++ {
		d[i] = int((to.At(i) - from.At(i)) / step)
	}
	return d
}

// TODO: remove all uses of this
func panicif(err error) {
	if err != nil {
		panic(err.Error())
	}
}
