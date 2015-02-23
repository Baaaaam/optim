package optim

import (
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync"

	"github.com/gonum/matrix/mat64"
)

var Rand Rng = rand.New(rand.NewSource(1))

type Rng interface {
	Float64() float64
	Intn(n int) int
	Perm(n int) []int
}

func RandFloat() float64 { return Rand.Float64() }

type Solver struct {
	Method       Method
	Obj          Objectiver
	Mesh         Mesh
	MaxIter      int
	MaxEval      int
	MaxNoImprove int
	MinStep      float64

	neval, niter int
	noimprove    int
	best         Point
	err          error
}

func (s *Solver) Best() Point { return s.best }
func (s *Solver) Niter() int  { return s.niter }
func (s *Solver) Neval() int  { return s.neval }
func (s *Solver) Err() error  { return s.err }

func (s *Solver) Run() error {
	for s.Next() {
	}
	return s.Err()
}

func (s *Solver) Next() (more bool) {
	if s.Mesh == nil {
		s.Mesh = &InfMesh{}
	}
	if s.niter == 0 {
		s.best.Val = math.Inf(1)
	}

	var n int
	var best Point
	best, n, s.err = s.Method.Iterate(s.Obj, s.Mesh)
	s.neval += n
	s.niter++

	if best.Val < s.best.Val {
		s.best = best
		s.noimprove = 0
	} else {
		s.noimprove++
	}

	if s.err != nil {
		return false
	}

	more = true && (s.MaxNoImprove == 0 || s.noimprove < s.MaxNoImprove)
	more = more && (s.MaxIter == 0 || s.niter < s.MaxIter)
	more = more && (s.MaxEval == 0 || s.neval < s.MaxEval)
	more = more && (s.MinStep == 0 || s.Mesh.Step() > s.MinStep)
	return more
}

type Point struct {
	Pos []float64
	Val float64
}

func (p *Point) Len() int             { return len(p.Pos) }
func (p *Point) Matrix() *mat64.Dense { return mat64.NewDense(p.Len(), 1, p.Pos) }
func (p *Point) String() string       { return fmt.Sprintf("f%v = %v", p.Pos, p.Val) }

func (p *Point) Clone() *Point {
	pos := make([]float64, len(p.Pos))
	copy(pos, p.Pos)
	return &Point{Pos: pos, Val: p.Val}
}

func (p *Point) Hash() [sha1.Size]byte {
	data := make([]byte, p.Len()*8)
	for i := 0; i < p.Len(); i++ {
		binary.BigEndian.PutUint64(data[i*8:], math.Float64bits(p.Pos[i]))
	}
	return sha1.Sum(data)
}

type Method interface {
	// Iterate runs a single iteration of a solver and reports the number of
	// function evaluations n and the best point.
	Iterate(obj Objectiver, m Mesh) (best Point, n int, err error)

	// AddPoint enables limited hybriding of different optimization iterators
	// by allowing iterators/solvers to add share information by suggesting
	// points to each other.
	AddPoint(p Point)
}

type Evaler interface {
	// Eval evaluates each point using obj and returns the values and number
	// of times obj was called n.  If unevaluated points are returned
	// in the results slice, they should have objective value set to
	// +Infinity. The order of points in results must be the same as the order
	// of the passed in points.  len(results) may be less than len(points).
	Eval(obj Objectiver, points ...Point) (results []*Point, n int, err error)
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
	// UseCount reports the number of times a cached objective evaluation was
	// successfully used to avoid recalculation.
	UseCount int
}

func NewCacheEvaler(ev Evaler) *CacheEvaler {
	return &CacheEvaler{
		ev:    ev,
		cache: map[[sha1.Size]byte]float64{},
	}
}

func (ev *CacheEvaler) Eval(obj Objectiver, points ...Point) (results []*Point, n int, err error) {
	results = points
	fromnew := make([]int, 0, len(points))
	newp := make([]*Point, 0, len(points))
	for i, p := range points {
		h := p.Hash()
		if val, ok := ev.cache[h]; ok {
			p.Val = val
			ev.UseCount++
		} else {
			fromnew = append(fromnew, i)
			newp = append(newp, p)
			results[i].Val = math.Inf(1) // to be safe
		}
	}

	newresults, n, err := ev.ev.Eval(obj, newp...)
	for _, p := range newresults {
		ev.cache[p.Hash()] = p.Val
	}

	for i, p := range newresults {
		results[fromnew[i]].Val = p.Val
	}

	// shrink if error resulted in fewer new results being returned
	if len(fromnew) > 0 {
		i := fromnew[len(newresults)-1]
		results = results[:i+1]
	}

	return results, n, err
}

type SerialEvaler struct {
	ContinueOnErr bool
}

func (ev SerialEvaler) Eval(obj Objectiver, points ...Point) (results []*Point, n int, err error) {
	results = make([]*Point, len(points))

	indexes := uniqof(points)
	defer fillfromuniq(indexes, results)

	for i, p := range points {
		if indexes[i] != i {
			// skip duplicates
			continue
		}

		p.Val, err = obj.Objective(p.Pos)
		n++
		results[i] = p
		if err != nil && !ev.ContinueOnErr {
			return results[:i+1], n, err
		}
	}

	return results, n, nil
}

type errpoint struct {
	Point
	Index int
	Err   error
}

// uniqof returns a set of indexes that map a point index into another index
// in the same point slice that can be used go get objective values of
// duplicate positions.  If a point has a unique position, indexes[i] = i.
func uniqof(points []*Point) (indexes []int) {
	indexes = make([]int, len(points))
	alreadyhave := map[[sha1.Size]byte]int{}
	for i, p := range points {
		h := p.Hash()
		if v, ok := alreadyhave[h]; !ok {
			alreadyhave[h] = i
			indexes[i] = i
		} else {
			indexes[i] = v
		}
	}
	return indexes
}

func fillfromuniq(indexes []int, points []*Point) {
	for i := range points {
		if v := indexes[i]; v != i {
			points[i] = points[v].Clone()
		}
	}
}

type ParallelEvaler struct{}

func (ev ParallelEvaler) Eval(obj Objectiver, points ...Point) (results []*Point, n int, err error) {
	ch := make(chan errpoint, len(points))
	wg := sync.WaitGroup{}
	indexes := uniqof(points)
	for i, p := range points {
		if indexes[i] != i {
			// skip duplicates
			continue
		}

		wg.Add(1)
		go func(i int, p Point) {
			defer wg.Done()
			perr := errpoint{Point: p, Index: i}
			perr.Val, perr.Err = obj.Objective(p.Pos())
			ch <- perr
		}(i, p)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	results = make([]*Point, len(points))
	for p := range ch {
		n++
		results[p.Index] = p.Point
		if p.Err != nil {
			err = p.Err
		}
	}
	fillfromuniq(indexes, results)
	return results, n, err
}

type Func func([]float64) float64

func (so Func) Objective(v []float64) (float64, error) { return so(v), nil }

type ObjectiveLogger struct {
	Obj Objectiver
	W   io.Writer
}

func (l *ObjectiveLogger) Objective(v []float64) (float64, error) {
	val, err := l.Obj.Objective(v)

	fmt.Fprintf(l.W, "f%v = %v\n", v, val)
	return val, err
}

// ObjectivePenalty wraps an objective function and adds a penalty factor for
// any violated linear constraints. If Weight is zero the underlying
// objective value will be returned unaltered.
type ObjectivePenalty struct {
	Obj     Objectiver
	A       *mat64.Dense
	Low, Up *mat64.Dense
	Weight  float64
	a       *mat64.Dense // stacked version of A
	b       *mat64.Dense // Low and Up stacked
	ranges  []float64    // ranges[i] = u[i] - l[i]
}

func (o *ObjectivePenalty) init() {
	if o.a != nil {
		// already initialized
		return
	}
	o.a, o.b, o.ranges = StackConstr(o.Low, o.A, o.Up)
}

func (o *ObjectivePenalty) Objective(v []float64) (float64, error) {
	o.init()
	val, err := o.Obj.Objective(v)

	if o.Weight == 0 {
		return val, err
	}

	ax := &mat64.Dense{}
	x := mat64.NewDense(len(v), 1, v)
	ax.Mul(o.a, x)

	m, _ := ax.Dims()

	penalty := 0.0
	for i := 0; i < m; i++ {
		if diff := ax.At(i, 0) - o.b.At(i, 0); diff > 0 {
			// maybe use "*=" for compounding penalty buildup
			penalty += diff / o.ranges[i] * o.Weight
		}
	}

	return val * (1 + penalty), err
}

func L2Dist(p1, p2 Point) float64 {
	tot := 0.0
	for i := 0; i < p1.Len(); i++ {
		diff := p1.At(i) - p2.At(i)
		tot += diff * diff
	}
	return math.Sqrt(tot)
}

// StackConstrBoxed converts the equations:
//
//     lb <= Ix <= ub
//     and
//     low <= Ax <= up
//
// into a single equation of the form:
//
//     Ax <= b
func StackConstrBoxed(lb, ub []float64, low, A, up *mat64.Dense) (stackA, b *mat64.Dense, ranges []float64) {
	lbm := mat64.NewDense(len(lb), 1, lb)
	ubm := mat64.NewDense(len(ub), 1, ub)

	stacklow := &mat64.Dense{}
	stacklow.Stack(low, lbm)

	stackup := &mat64.Dense{}
	stackup.Stack(up, ubm)

	boxA := mat64.NewDense(len(lb), len(lb), nil)
	for i := 0; i < len(lb); i++ {
		boxA.Set(i, i, 1)
	}

	stacked := &mat64.Dense{}
	stacked.Stack(A, boxA)
	return StackConstr(stacklow, stacked, stackup)
}

func StackConstr(low, A, up *mat64.Dense) (stackA, b *mat64.Dense, ranges []float64) {
	neglow := &mat64.Dense{}
	neglow.Scale(-1, low)
	b = &mat64.Dense{}
	b.Stack(up, neglow)

	negA := &mat64.Dense{}
	negA.Scale(-1, A)
	stackA = &mat64.Dense{}
	stackA.Stack(A, negA)

	// capture the range of each constraint from A because this information is
	// lost when converting from "low <= Ax <= up" via stacking to "Ax <= up".
	m, _ := A.Dims()
	ranges = make([]float64, m, 2*m)
	for i := 0; i < m; i++ {
		ranges[i] = up.At(i, 0) - low.At(i, 0)
		if ranges[i] == 0 {
			if up.At(i, 0) == 0 {
				ranges[i] = 1
			} else {
				ranges[i] = up.At(i, 0)
			}
		}
	}
	ranges = append(ranges, ranges...)

	return stackA, b, ranges
}
