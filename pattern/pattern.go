package pattern

import (
	"crypto/sha1"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"

	"github.com/rwcarlsen/optim"
)

var FoundBetterErr = errors.New("better position discovered")
var ZeroStepErr = errors.New("poll step size contracted to zero")

const (
	TblPolls = "patternpolls"
	TblInfo  = "patterninfo"
)

type Option func(*Method)

func Evaler(e optim.Evaler) Option { return func(m *Method) { m.ev = e } }

func NsuccessGrow(n int) Option {
	return func(m *Method) {
		m.NsuccessGrow = n
	}
}

const (
	Share   = true
	NoShare = false
)

func SearchMethod(m optim.Method, share bool) Option {
	return func(m2 *Method) {
		m2.Searcher = &WrapSearcher{Method: m, Share: share}
	}
}

func DiscreteSearch(m *Method) {
	m.DiscreteSearch = true
}

// Poll2N sets the method to poll in both forward and backward in every
// compass direction.
func Poll2N(m *Method) { m.Poller.SpanFn = Compass2N }

// PollNp1 sets the method to poll in n compass directions with random
// polarity plus one direction with the opposite of all other directions in
// every dimension.
func PollNp1(m *Method) { m.Poller.SpanFn = CompassNp1 }

// PollRandN sets the method to poll in n random directions setting the
// direction for a randomly chosen number of dimensions to +/- step size.
func PollRandN(n int) Option { return func(m *Method) { m.Poller.SpanFn = RandomN(n) } }

func DB(db *sql.DB) Option {
	return func(m *Method) {
		m.Db = db
	}
}

func SkipEps(eps float64) Option { return func(m *Method) { m.Poller.SkipEps = eps } }

func Nkeep(n int) Option { return func(m *Method) { m.Poller.Nkeep = n } }

type Method struct {
	ev             optim.Evaler
	Poller         *Poller
	Searcher       Searcher
	Curr           *optim.Point
	DiscreteSearch bool // true to project search points onto poll step size mesh
	NsuccessGrow   int  // number of successive successful polls before growing mesh
	nsuccess       int  // (internal) number of successive successful polls
	Db             *sql.DB
	count          int
}

func New(start *optim.Point, opts ...Option) *Method {
	m := &Method{
		Curr:         start,
		ev:           optim.SerialEvaler{},
		Poller:       &Poller{Nkeep: start.Len(), SkipEps: 1e-10},
		Searcher:     NullSearcher{},
		NsuccessGrow: -1,
	}

	for _, opt := range opts {
		opt(m)
	}
	m.initdb()
	return m
}

func (m *Method) AddPoint(p *optim.Point) {
	if p.Val < m.Curr.Val {
		m.Curr = p
	}
}

// Iterate mutates m and so for each iteration, the same, mutated m should be
// passed in.
func (m *Method) Iterate(o optim.Objectiver, mesh optim.Mesh) (best *optim.Point, n int, err error) {
	var nevalsearch, nevalpoll int
	var success bool
	defer m.updateDb(&nevalsearch, &nevalpoll, mesh.Step())
	m.count++

	if !m.DiscreteSearch {
		success, best, nevalsearch, err = m.Searcher.Search(o, nil, m.Curr)
	} else {
		success, best, nevalsearch, err = m.Searcher.Search(o, mesh, m.Curr)
	}

	n += nevalsearch
	if err != nil {
		return best, n, err
	} else if success {
		m.Curr = best
		return best, n, nil
	}

	// It is important to recenter mesh on new best point before polling.
	// This is necessary because the search may not be operating on the
	// current mesh grid.  This doesn't need to happen if search succeeds
	// because search either always operates on the same grid, or always
	// operates in continuous space.
	mesh.SetOrigin(m.Curr.Pos) // TODO: test that this doesn't get set to Zero pos [0 0 0...] on first iteration.

	success, best, nevalpoll, err = m.Poller.Poll(o, m.ev, mesh, m.Curr)
	n += nevalpoll
	if err != nil {
		return m.Curr, n, err
	} else if success {
		m.Curr = best
		m.nsuccess++
		if m.nsuccess == m.NsuccessGrow { // == allows -1 to mean never grow
			mesh.SetStep(mesh.Step() * 2.0)
			m.nsuccess = 0 // reset after resize
		}

		// Important to recenter mesh on new best point.  More particularly,
		// the mesh may have been resized and the new best may not lie on the
		// previous mesh grid.
		mesh.SetOrigin(best.Pos)

		return best, n, nil
	} else {
		m.nsuccess = 0
		var err error
		mesh.SetStep(mesh.Step() * 0.5)
		if mesh.Step() == 0 {
			err = ZeroStepErr
		}
		return m.Curr, n, err
	}
}

func (m *Method) initdb() {
	if m.Db == nil {
		return
	}

	s := "CREATE TABLE IF NOT EXISTS " + TblPolls + " (iter INTEGER,val REAL"
	s += m.xdbsql("define")
	s += ");"

	_, err := m.Db.Exec(s)
	if checkdberr(err) {
		return
	}

	s = "CREATE TABLE IF NOT EXISTS " + TblInfo + " (iter INTEGER,step INTEGER,nsearch INTEGER,npoll INTEGER,val REAL"
	s += m.xdbsql("define")
	s += ");"
	_, err = m.Db.Exec(s)
	if checkdberr(err) {
		return
	}
}

func (m Method) xdbsql(op string) string {
	s := ""
	for i := range m.Curr.Pos {
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

func (m Method) updateDb(nsearch, npoll *int, step float64) {
	if m.Db == nil {
		return
	}

	tx, err := m.Db.Begin()
	if err != nil {
		panic(err.Error())
	}
	defer tx.Commit()

	s1 := "INSERT INTO " + TblPolls + " (iter,val" + m.xdbsql("x") + ") VALUES (?,?" + m.xdbsql("?") + ");"
	for _, p := range m.Poller.Points() {
		args := []interface{}{m.count, p.Val}
		args = append(args, pos2iface(p.Pos)...)
		_, err := tx.Exec(s1, args...)
		if checkdberr(err) {
			return
		}
	}

	s2 := "INSERT INTO " + TblInfo + " (iter,step,nsearch, npoll,val" + m.xdbsql("x") + ") VALUES (?,?,?,?,?" + m.xdbsql("?") + ");"
	glob := m.Curr
	args := []interface{}{m.count, step, *nsearch, *npoll, glob.Val}
	args = append(args, pos2iface(glob.Pos)...)
	_, err = tx.Exec(s2, args...)
	if checkdberr(err) {
		return
	}
}

type Poller struct {
	// Nkeep specifies the number of previous successful poll directions to
	// reuse on the next poll. The number of reused directions is min(Nkeep,
	// nsuccessful).
	Nkeep int
	// SkipEps is the distance from the center point within which a poll point
	// is excluded from evaluation.  This can occur if a mesh projection
	// results in a point being projected back near the poll origin point.
	SkipEps    float64
	SpanFn     SpanFunc
	keepdirecs []direc
	points     []*optim.Point
	prevhash   [sha1.Size]byte
	prevstep   float64
}

func (cp *Poller) Points() []*optim.Point { return cp.points }

type direc struct {
	dir []int
	val float64
}

type byval []direc

func (b byval) At(i int) []int     { return b[i].dir }
func (b byval) Less(i, j int) bool { return b[i].val < b[j].val }
func (b byval) Len() int           { return len(b) }
func (b byval) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }

// Poll polls on mesh m centered on point from.  It is responsible for
// selecting points and evaluating them with ev using obj.  If a better
// point was found, it returns success == true, the point, and number of
// evaluations.  If a better point was not found, it returns false, the
// from point, and the number of evaluations.  If err is non-nil, success
// must be false and best must be from - neval may be non-zero.
func (cp *Poller) Poll(obj optim.Objectiver, ev optim.Evaler, m optim.Mesh, from *optim.Point) (success bool, best *optim.Point, neval int, err error) {
	best = from

	pollpoints := []*optim.Point{}

	// Only poll compass directions if we haven't polled from this point
	// before.  DONT DELETE - this can fire sometimes if the mesh isn't
	// allowed to contract below a certain step (i.e. integer meshes).
	h := from.Hash()
	if h != cp.prevhash || cp.prevstep != m.Step() {
		// TODO: write test that checks we poll compass dirs again if only mesh
		// step changed (and not from point)
		if cp.SpanFn == nil {
			pollpoints = append(pollpoints, genPollPoints(from, Compass2N, m)...)
		} else {
			pollpoints = append(pollpoints, genPollPoints(from, cp.SpanFn, m)...)
		}
		pollpoints = append(pollpoints, genPollPoints(from, RandomN(from.Len()), m)...)
		cp.prevhash = h
	} else {
		// Use random directions instead.
		pollpoints = append(pollpoints, genPollPoints(from, RandomN(2*from.Len()), m)...)
	}
	cp.prevstep = m.Step()

	// Add successful directions from last poll.  We want to add these points
	// in front of the other points so we can potentially stop earlier if
	// polling opportunistically.
	prevgood := make([]*optim.Point, len(cp.keepdirecs))
	for i, dir := range cp.keepdirecs {
		prevgood[i] = pointFromDirec(from, dir.dir, m)
	}
	pollpoints = append(prevgood, pollpoints...)
	//pollpoints = append(pollpoints, prevgood...)

	cp.points = make([]*optim.Point, 0, len(pollpoints))
	if cp.SkipEps == 0 {
		cp.points = pollpoints
	} else {
		for _, p := range pollpoints {
			// It is possible that due to the mesh gridding, the poll point is
			// outside of constraints or bounds and will be rounded back to the
			// current point. Check for this and skip the poll point if this is
			// the case.
			dist := optim.L2Dist(from, p)
			if dist > cp.SkipEps {
				cp.points = append(cp.points, p)
			}
		}
	}

	objstop := &objStopper{Objectiver: obj, Best: from.Val}
	results, n, err := ev.Eval(objstop, cp.points...)
	if err != nil && err != FoundBetterErr {
		return false, best, n, err
	}

	// this is separate from best to allow all points better than from to be
	// added to keepdirecs before we update the best point.
	nextbest := from

	// Sort results and keep the best Nkeep as poll directions.
	for _, p := range results {
		if p.Val < best.Val {
			cp.keepdirecs = append(cp.keepdirecs, direc{direcbetween(from, p, m), p.Val})
		}
		if p.Val < nextbest.Val {
			nextbest = p
		}
	}
	best = nextbest

	sort.Sort(byval(cp.keepdirecs))
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
	Search(o optim.Objectiver, m optim.Mesh, curr *optim.Point) (success bool, best *optim.Point, n int, err error)
}

type NullSearcher struct{}

func (_ NullSearcher) Search(o optim.Objectiver, m optim.Mesh, curr *optim.Point) (success bool, best *optim.Point, n int, err error) {
	return false, curr, 0, nil // TODO: test that this returns curr instead of something else
}

type WrapSearcher struct {
	Method optim.Method
	// Share specifies whether to add the current best point to the
	// searcher's underlying method before performing the search.
	Share bool
}

func (s *WrapSearcher) Search(o optim.Objectiver, m optim.Mesh, curr *optim.Point) (success bool, best *optim.Point, n int, err error) {
	if s.Share {
		s.Method.AddPoint(curr)
	}
	best, n, err = s.Method.Iterate(o, m)
	if err != nil {
		return false, &optim.Point{Val: math.Inf(1)}, n, err
	}
	if best.Val < curr.Val {
		return true, best, n, nil
	}
	// TODO: write test that checks we return curr instead of best for search
	// fail.
	return false, curr, n, nil
}

// objStopper is wraps an Objectiver and returns the objective value along
// with FoundBetterErr as soon as calculates a value better than Best.  This
// is useful for things like terminating early with opportunistic polling.
type objStopper struct {
	Best float64
	optim.Objectiver
}

func (s *objStopper) Objective(v []float64) (float64, error) {
	obj, err := s.Objectiver.Objective(v)
	if err != nil {
		return obj, err
	} else if obj < s.Best {
		return obj, FoundBetterErr
	}
	return obj, nil
}

func genPollPoints(from *optim.Point, span SpanFunc, m optim.Mesh) []*optim.Point {
	ndim := from.Len()
	dirs := span(ndim)
	polls := make([]*optim.Point, 0, len(dirs))
	for _, d := range dirs {
		polls = append(polls, pointFromDirec(from, d, m))
	}
	return polls
}

func pointFromDirec(from *optim.Point, direc []int, m optim.Mesh) *optim.Point {
	pos := make([]float64, from.Len())
	step := m.Step()
	for i, x0 := range from.Pos {
		pos[i] = x0 + float64(direc[i])*step

	}
	return &optim.Point{m.Nearest(pos), math.Inf(1)}
}

// SpanFunc is returns a set of poll directions (maybe positive spanning set?)
type SpanFunc func(ndim int) [][]int

// Compass2N returns a compass positive basis set of polling directions in a
// randomized order.
func Compass2N(ndim int) [][]int {
	dirs := make([][]int, 2*ndim)
	perms := optim.Rand.Perm(ndim)
	for i := 0; i < ndim; i++ {
		d := make([]int, ndim)
		d[i] = 1
		dirs[perms[i]] = d

		d = make([]int, ndim)
		d[i] = -1
		dirs[ndim+perms[i]] = d
	}
	return dirs
}

func CompassNp1(ndim int) [][]int {
	dirs := make([][]int, 0, ndim+1)
	final := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		d := make([]int, ndim)

		r := optim.Rand.Intn(2)
		d[i] = 1
		final[i] = -1
		if r == 0 {
			d[i] = -1
			final[i] = 1
		}

		dirs = append(dirs, d)
	}
	dirs = append(dirs, final)
	end := len(dirs) - 1
	// poll the diagonal direction first
	dirs[0], dirs[end] = dirs[end], dirs[0]
	return dirs
}

// RandomN returns ndim random polling directions that exclude the
// compass directions.
func RandomN(n int) SpanFunc {
	return func(ndim int) [][]int {
		dirs := make([][]int, 0, n)
		for len(dirs) < n {
			d1 := make([]int, ndim)
			d2 := make([]int, ndim)

			nNonzero := 1
			if ndim == 1 { // compass directions cover everything
				return dirs
			} else if ndim == 2 { // this check prevents calling Intn(0) - which is invalid
				nNonzero = 2 // exclude compass directions
			} else {
				// Intn(-2)+2 is to exclude vector of all zeros and compass directions.
				//nNonzero = optim.Rand.Intn(ndim-2) + 2
				nNonzero = optim.Rand.Intn(2) + 2
				// TODO: skew this nNonzero distribution to have more lower numbers
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
			dirs = append(dirs, d1)
			dirs = append(dirs, d2)
		}
		return dirs
	}
}

func pos2iface(pos []float64) []interface{} {
	iface := []interface{}{}
	for _, v := range pos {
		iface = append(iface, v)
	}
	return iface
}

func direcbetween(from, to *optim.Point, m optim.Mesh) []int {
	d := make([]int, from.Len())
	step := m.Step()
	for i, x0 := range from.Pos {
		d[i] = int((to.Pos[i] - x0) / step)
	}
	return d
}

func checkdberr(err error) bool {
	if err != nil {
		log.Print("swarm: db write failed -", err)
		return true
	}
	return false
}
