package pswarm

import (
	"database/sql"
	"fmt"
	"math"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
	"github.com/rwcarlsen/optim/pop"
)

// These params originate from work done by Clerc:
//
//     Clerc and M.  “The swarm and the queen: towards a deterministic and
//     adaptive particle swarm optimization” Proc. 1999 Congress on
//     Evolutionary Computation, pp. 1951-1957
//
// The cognition and social parameters correspond to c1 and c2 values of 2.05
// that have been multiplied by their constriction coeffient - i.e.
// DefaultSocial = Constriction(2.05, 2.05)*2.05.  DefaultInertia is set equal
// to the corresponding constriction coefficient.
//
// I have found that inertia = Constriction(2.098, 2.098) and  seems to work
// better when using the swarm solver (without pattern search) for 30D
// Rosenbrock.
const (
	DefaultCognition = 1.496179765663133
	DefaultSocial    = 1.496179765663133
	DefaultInertia   = 0.7298437881283576
)

const (
	TblParticles = "swarmparticles"
	TblBest      = "swarmbest"
)

// Constriction calculates the constriction coefficient for the given c1 and
// c2 for the particle velocity equation:
//
//    v_next = k(v_curr + c1*rand*(p_glob-x) + c2*rand*(p_personal-x))
//
//    or
//
//    v_next = w*v_curr + b1*rand*(p_glob-x) + b2*rand*(p_personal-x)
//
//    (with constriction coefficient multiplied through.
//
// c1+c2 should usually be greater than (but close to) 4.  'w = k' is often
// referred to as the inertia in the traditional swarm equation
func Constriction(c1, c2 float64) float64 {
	phi := c1 + c2
	return 2 / math.Abs(2-phi-math.Sqrt(phi*phi-4*phi))
}

type Particle struct {
	Id int
	optim.Point
	Vel  []float64
	Best optim.Point
}

func (p *Particle) Move(best optim.Point, vmax []float64, inertia, social, cognition float64) {
	// update velocity
	r1 := optim.RandFloat()
	r2 := optim.RandFloat()
	for i, currv := range p.Vel {
		p.Vel[i] = inertia*currv +
			cognition*r1*(p.Best.At(i)-p.At(i)) +
			social*r2*(best.At(i)-p.At(i))
		p.Vel[i] = math.Min(p.Vel[i], vmax[i])
	}

	// update position
	pos := make([]float64, p.Len())
	for i := range pos {
		pos[i] = p.At(i) + p.Vel[i]
	}
	p.Point = optim.NewPoint(pos, math.Inf(1))
}

func (p *Particle) Update(newp optim.Point) {
	// DO NOT update p's position with newp's position - it may have been
	// projected onto a mesh and be different.
	p.Val = newp.Val
	if p.Val < p.Best.Val {
		p.Best = newp
	}
}

type Population []*Particle

// NewPopulation initializes a population of particles using the given points
// and generates velocities for each dimension i initialized to uniform random
// values between minv[i] and maxv[i].  github.com/rwcarlsen/optim.Rand is
// used for random numbers.
func NewPopulation(points []optim.Point, minv, maxv []float64) Population {
	pop := make(Population, len(points))
	for i, p := range points {
		pop[i] = &Particle{
			Id:    i,
			Point: p,
			Best:  optim.NewPoint(p.Pos(), math.Inf(1)),
			Vel:   make([]float64, len(minv)),
		}
		for j := range minv {
			pop[i].Vel[j] = minv[j] + (maxv[j]-minv[j])*optim.RandFloat()
		}
	}
	return pop
}

func NewPopulationRand(n int, low, up []float64) Population {
	minv := make([]float64, len(up))
	maxv := make([]float64, len(up))
	for i := range up {
		maxv[i] = (up[i] - low[i]) * 1
		minv[i] = 0 * maxv[i]
	}

	points := pop.New(n, low, up)
	return NewPopulation(points, minv, maxv)
}

func (pop Population) Points() []optim.Point {
	points := make([]optim.Point, 0, len(pop))
	for _, p := range pop {
		points = append(points, p.Point)
	}
	return points
}

func (pop Population) Best() *Particle {
	if len(pop) == 0 {
		return nil
	}

	best := pop[0]
	for _, p := range pop[1:] {
		// TODO: write test to make sure this checks p.Best.Val < best.Best.Val
		// and NOT p.Val or best.Val.
		if p.Best.Val < best.Best.Val {
			best = p
		}
	}
	return best
}

type Option func(*Iterator)

func Vmax(vmaxes []float64) Option {
	return func(it *Iterator) {
		it.Vmax = vmaxes
	}
}

// VmaxBounds sets the maximum particle velocity for each dimension equal to
// the bounded range for the problem - i.e. up[i]-low[i] for each dimension.
// This is a good rule of thumb given in:
//
//     Eberhart, R.C.; Yuhui Shi, "Particle swarm optimization: developments,
//     applications and resources," Evolutionary Computation, 2001. Proceedings of
//     the 2001 Congress on , vol.1, no., pp.81,86 vol. 1, 2001 doi:
//     10.1109/CEC.2001.934374
func VmaxBounds(low, up []float64) Option {
	return func(it *Iterator) {
		it.Vmax = make([]float64, len(low))
		for i := range it.Vmax {
			it.Vmax[i] = up[i] - low[i]
		}
	}
}

func DB(db *sql.DB) Option {
	return func(it *Iterator) {
		it.Db = db
	}
}

func KillDist(dist float64) Option {
	return func(it *Iterator) {
		it.KillDist = dist
	}
}

func VelUpdParams(cognition, social float64) Option {
	return func(it *Iterator) {
		it.Cognition = cognition
		it.Social = social
	}
}

// LinInertia sets particle inertia for velocity updates to varry linearly
// from the start (high) to end (low) values from 0 to maxiter.  Common values
// are start = 0.9 and end = 0.4 - for details see:
//
// Eberhart, R.C.; Yuhui Shi, "Particle swarm optimization: developments,
// applications and resources," Evolutionary Computation, 2001. Proceedings of
// the 2001 Congress on , vol.1, no., pp.81,86 vol. 1, 2001 doi:
// 10.1109/CEC.2001.934374
func LinInertia(start, end float64, maxiter int) Option {
	return func(it *Iterator) {
		it.InertiaFn = func(iter int) float64 {
			return start - (start-end)*float64(iter)/float64(maxiter)
		}
	}
}

func FixedInertia(v float64) Option {
	return func(it *Iterator) {
		it.InertiaFn = func(iter int) float64 { return v }
	}
}

type Iterator struct {
	// KillDist is the distance from the global optimum below which particles
	// are killed.  Zero for never killing particles.  Large values result in
	// a particle being killed whenever it becomes the global optimum.
	KillDist float64
	Pop      Population
	optim.Evaler
	Cognition float64
	Social    float64
	InertiaFn func(iter int) float64
	// Vmax is the speed limit in each dimension for particles.  If nil,
	// infinity is used.
	Vmax  []float64
	Db    *sql.DB
	count int
	best  optim.Point
}

func NewIterator(e optim.Evaler, pop Population, opts ...Option) *Iterator {
	if e == nil {
		e = optim.SerialEvaler{}
	}

	vmax := make([]float64, pop[0].Len())
	for i := range vmax {
		vmax[i] = math.Inf(1)
	}

	it := &Iterator{
		Pop:       pop,
		Evaler:    e,
		Cognition: DefaultCognition,
		Social:    DefaultSocial,
		InertiaFn: func(iter int) float64 { return DefaultInertia },
		Vmax:      vmax,
		best:      pop.Best().Point,
	}

	for _, opt := range opts {
		opt(it)
	}

	it.initdb()
	return it
}

func (it *Iterator) AddPoint(p optim.Point) {
	if p.Val < it.best.Val {
		it.best = p
	}
}

func (it *Iterator) Iterate(obj optim.Objectiver, m mesh.Mesh) (best optim.Point, neval int, err error) {
	it.count++

	// project positions onto mesh
	points := it.Pop.Points()
	if m != nil {
		for i, p := range points {
			points[i] = optim.Nearest(p, m)
		}
	}

	// evaluate current positions
	results, n, err := it.Evaler.Eval(obj, points...)
	if err != nil {
		return optim.Point{Val: math.Inf(1)}, n, err
	}
	for i := range results {
		it.Pop[i].Update(results[i])
	}
	it.updateDb()

	// move particles and update current best
	for _, p := range it.Pop {
		p.Move(it.best, it.Vmax, it.InertiaFn(it.count), it.Social, it.Cognition)
	}

	pbest := it.Pop.Best()
	// TODO: write test to make sure this checks pbest.Best.Val instead of p.Val.
	if pbest != nil && pbest.Best.Val < it.best.Val {
		it.best = pbest.Best
		// only kill if moving particles found a new best
		if it.KillDist > 0 && optim.L2Dist(pbest.Point, it.best) < it.KillDist {
			for i, p := range it.Pop {
				if p.Id == pbest.Id {
					it.Pop = append(it.Pop[:i], it.Pop[i+1:]...)
					break
				}
			}
		}
	}

	return it.best, n, nil
}

func (it *Iterator) initdb() {
	if it.Db == nil {
		return
	}

	s := "CREATE TABLE IF NOT EXISTS " + TblParticles + " (particle INTEGER, iter INTEGER, val REAL"
	s += it.xdbsql("define")
	s += ");"

	_, err := it.Db.Exec(s)
	panicif(err)

	s = "CREATE TABLE IF NOT EXISTS " + TblBest + " (iter INTEGER, val REAL"
	s += it.xdbsql("define")
	s += ");"
	_, err = it.Db.Exec(s)
	panicif(err)
}

func (it *Iterator) xdbsql(op string) string {
	s := ""
	for i := range it.Pop[0].Pos() {
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

func pos2iface(pos []float64) []interface{} {
	iface := []interface{}{}
	for _, v := range pos {
		iface = append(iface, v)
	}
	return iface
}

func (it *Iterator) updateDb() {
	if it.Db == nil {
		return
	}

	tx, err := it.Db.Begin()
	if err != nil {
		panic(err.Error())
	}
	defer tx.Commit()

	s1 := "INSERT INTO " + TblParticles + " (particle,iter,val" + it.xdbsql("x") + ") VALUES (?,?,?" + it.xdbsql("?") + ");"
	s2 := "INSERT INTO " + TblBest + " (iter,val" + it.xdbsql("x") + ") VALUES (?,?" + it.xdbsql("?") + ");"
	for _, p := range it.Pop {
		args := []interface{}{p.Id, it.count, p.Val}
		args = append(args, pos2iface(p.Pos())...)
		_, err := tx.Exec(s1, args...)
		panicif(err)
	}

	glob := it.best
	args := []interface{}{it.count, glob.Val}
	args = append(args, pos2iface(glob.Pos())...)
	_, err = tx.Exec(s2, args...)
	panicif(err)
}

// TODO: remove all uses of this
func panicif(err error) {
	if err != nil {
		panic(err.Error())
	}
}
