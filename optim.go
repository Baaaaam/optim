package optim

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim/mesh"
)

var Rand Rng = rand.New(rand.NewSource(1))

type Rng interface {
	Float64() float64
}

func RandFloat() float64 { return Rand.Float64() }

func Solve(it Iterator, obj Objectiver, m mesh.Mesh, maxiter, maxeval int) (best Point, niter, neval int, err error) {
	for neval < maxeval && niter < maxiter {
		var n int
		best, n, err = it.Iterate(obj, m)
		neval += n
		niter++
		if err != nil {
			return best, niter, neval, err
		}
	}
	return best, niter, neval, nil
}

type Point struct {
	pos []float64
	Val float64
}

func NewPoint(pos []float64, val float64) Point {
	cpos := make([]float64, len(pos))
	copy(cpos, pos)
	return Point{pos: cpos, Val: val}
}

func (p Point) At(i int) float64 { return p.pos[i] }

func (p Point) Len() int { return len(p.pos) }

func (p Point) Matrix() *mat64.Dense {
	return mat64.NewDense(p.Len(), 1, p.Pos())
}

func (p Point) Pos() []float64 {
	pos := make([]float64, len(p.pos))
	copy(pos, p.pos)
	return pos
}

func (p Point) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "f[%.5f", p.pos[0])
	for _, v := range p.pos[1:] {
		fmt.Fprintf(&b, " %.5f", v)
	}
	fmt.Fprintf(&b, "] = %.5f", p.Val)
	return b.String()
}

func hashPoint(p Point) [sha1.Size]byte {
	data := make([]byte, p.Len()*8)
	for i := 0; i < p.Len(); i++ {
		binary.BigEndian.PutUint64(data[i*8:], math.Float64bits(p.At(i)))
	}
	return sha1.Sum(data)
}

type Iterator interface {
	// Iterate runs a single iteration of a solver and reports the number of
	// function evaluations n and the best point.
	Iterate(obj Objectiver, m mesh.Mesh) (best Point, n int, err error)

	// AddPoint enables limited hybriding of different optimization iterators
	// by allowing iterators/solvers to add share information by suggesting
	// points to each other.
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
	fromnew := make([]int, 0, len(points))
	newp := make([]Point, 0, len(points))
	for i, p := range points {
		if val, ok := ev.cache[hashPoint(p)]; ok {
			p.Val = val
		} else {
			fromnew = append(fromnew, i)
			newp = append(newp, p)
		}
	}

	newresults, n, err := ev.ev.Eval(obj, newp...)
	for _, p := range newresults {
		ev.cache[hashPoint(p)] = p.Val
	}

	for i, p := range newresults {
		points[fromnew[i]].Val = p.Val
	}

	// shrink if error resulted in fewer new results being returned
	if len(fromnew) > 0 {
		points = points[:fromnew[len(newresults)-1]+1]
	}

	return points, n, err
}

type SerialEvaler struct {
	ContinueOnErr bool
}

func (ev SerialEvaler) Eval(obj Objectiver, points ...Point) (results []Point, n int, err error) {
	results = make([]Point, 0, len(points))
	for _, p := range points {
		p.Val, err = obj.Objective(p.Pos())
		results = append(results, p)
		if err != nil && !ev.ContinueOnErr {
			return results, len(results), err
		}
	}
	return results, len(results), nil
}

type errpoint struct {
	Point
	Err error
}

type ParallelEvaler struct{}

func (ev ParallelEvaler) Eval(obj Objectiver, points ...Point) (results []Point, n int, err error) {
	results = make([]Point, 0, len(points))
	wg := sync.WaitGroup{}
	wg.Add(len(points))
	ch := make(chan errpoint, len(points))
	for _, p := range points {
		go func() {
			perr := errpoint{Point: p}
			perr.Val, perr.Err = obj.Objective(p.Pos())
			ch <- perr
			wg.Done()
		}()
	}

	for p := range ch {
		if p.Err != nil {
			err = p.Err
		}
		results = append(results, p.Point)
	}
	wg.Wait()
	return results, len(results), err
}

type Func func([]float64) float64

func (so Func) Objective(v []float64) (float64, error) { return so(v), nil }

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

func Nearest(p Point, m mesh.Mesh) Point {
	return NewPoint(m.Nearest(p.pos), p.Val)
}

func L2Dist(p1, p2 Point) float64 {
	tot := 0.0
	for i := 0; i < p1.Len(); i++ {
		tot += math.Pow(p1.At(i)-p2.At(i), 2)
	}
	return math.Sqrt(tot)
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
	}
	ranges = append(ranges, ranges...)

	return stackA, b, ranges
}
