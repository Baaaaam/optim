package swarm

import (
	"database/sql"
	"fmt"
	"math"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

// These params are calculated using a constriction factor originally
// described in:
//
//     Clerc and M.  “The swarm and the queen: towards a deterministic and
//     adaptive particle swarm optimization” Proc. 1999 Congress on
//     Evolutionary Computation, pp. 1951-1957
//
// The cognition and social parameters correspond to c1 and c2 values of 2.05
// that have been multiplied by their constriction coeffient - i.e.
// DefaultSocial = Constriction(2.05, 2.05)*2.05.  DefaultInertia is set equal
// to the constriction coefficient.
const (
	DefaultCognition = 1.496179765663133
	DefaultSocial    = 1.496179765663133
	DefaultInertia   = 0.7298437881283576
)

const (
	// TblParticles is the name of the sql database table that contains
	// positions and values for particles for each iteration.
	TblParticles = "swarmparticles"
	// TblParticlesMeshed is the name of the sql database table that contains
	// mesh-projected positions (where objective evaluations actually
	// occurred)  and values for particles for each iteration.
	TblParticlesMeshed = "swarmparticlesmesh"
	// TblParticlesBest is the name of the sql database table that contains
	// each particle's personal best position at each iteration.
	TblParticlesBest = "swarmparticlesbest"
	// TblBest is the name of the sql database table that contains
	// the best position for the entire swarm at each iteration.
	TblBest = "swarmbest"
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

func (p *Particle) Move(gbest optim.Point, vmax []float64, inertia, social, cognition float64) {
	// update velocity
	for i, currv := range p.Vel {
		// random numbers r1 and r2 MUST go inside this loop and be generated
		// uniquely for each dimension of p's velocity.
		r1 := optim.RandFloat()
		r2 := optim.RandFloat()
		p.Vel[i] = inertia*currv +
			cognition*r1*(p.Best.At(i)-p.At(i)) +
			social*r2*(gbest.At(i)-p.At(i))
		if math.Abs(p.Vel[i]) > vmax[i] {
			p.Vel[i] = math.Copysign(vmax[i], p.Vel[i])
		}
	}

	// update position
	pos := make([]float64, p.Len())
	for i := range pos {
		pos[i] = p.At(i) + p.Vel[i]
	}
	p.Point = optim.NewPoint(pos, math.Inf(1))
}

func (p *Particle) Kill(gbest optim.Point, xtol, vtol float64) bool {
	if xtol == 0 || vtol == 0 {
		return false
	}

	totv := 0.0
	diffx := 0.0
	for i, v := range p.Vel {
		totv += v * v
		diffx += math.Pow(p.At(i)-gbest.At(i), 2)
	}
	return (totv < vtol*vtol) && (diffx < xtol*xtol)
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
func NewPopulation(points []optim.Point, vmax []float64) Population {
	pop := make(Population, len(points))
	for i, p := range points {
		pop[i] = &Particle{
			Id:    i,
			Point: p,
			Best:  p,
			Vel:   make([]float64, len(vmax)),
		}
		for j, v := range vmax {
			pop[i].Vel[j] = v * (1 - 2*optim.RandFloat())
		}
	}
	return pop
}

// NewPopulationRand creates a population of randomly positioned particles
// uniformly distributed in the box-bounds described by low and up.
func NewPopulationRand(n int, low, up []float64) Population {
	points := optim.RandPop(n, low, up)
	return NewPopulation(points, vmaxfrombounds(low, up))
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

func VmaxAll(vmax float64) Option {
	return func(it *Iterator) {
		for i := range it.Vmax {
			it.Vmax[i] = vmax
		}
	}
}

// VmaxBounds sets the maximum particle speed for each dimension equal to
// the bounded range for the problem - i.e. up[i]-low[i]/2 for each dimension.
// This is a good rule of thumb given in:
//
//     Eberhart, R.C.; Yuhui Shi, "Particle swarm optimization: developments,
//     applications and resources," Evolutionary Computation, 2001. Proceedings of
//     the 2001 Congress on , vol.1, no., pp.81,86 vol. 1, 2001 doi:
//     10.1109/CEC.2001.934374
func VmaxBounds(low, up []float64) Option {
	return func(it *Iterator) {
		it.Vmax = vmaxfrombounds(low, up)
	}
}

func DB(db *sql.DB) Option {
	return func(it *Iterator) {
		it.Db = db
	}
}

func KillTol(xtol, vtol float64) Option {
	return func(it *Iterator) {
		it.Xtol = xtol
		it.Vtol = vtol
	}
}

func LearnFactors(cognition, social float64) Option {
	return func(it *Iterator) {
		it.Cognition = cognition
		it.Social = social
	}
}

// LinInertia sets particle inertia for velocity updates to varry linearly
// from the start (high) to end (low) values from 0 to maxiter.  Common values
// are start = 0.9 and end = 0.4 - for details see:
//
//     Eberhart, R.C.; Yuhui Shi, "Particle swarm optimization: developments,
//     applications and resources," Evolutionary Computation, 2001. Proceedings of
//     the 2001 Congress on , vol.1, no., pp.81,86 vol. 1, 2001 doi:
//     10.1109/CEC.2001.934374
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
	// Xtol is the distance from the global best under which particles are
	// considered to removal.  This must occur simultaneously with the Vtol
	// condition.
	Xtol float64
	// Vtol is the velocity under which particles are considered to removal.
	// This must occur simultaneously with the Xtol condition.
	Vtol float64
	Pop  Population
	optim.Evaler
	Cognition float64
	Social    float64
	InertiaFn func(iter int) float64
	// Vmax is the speed limit per dimension for particles.  If nil,
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
	it.updateDb(m)

	// move particles and update current best
	for _, p := range it.Pop {
		p.Move(it.best, it.Vmax, it.InertiaFn(it.count), it.Social, it.Cognition)
	}

	// TODO: write test to make sure this checks pbest.Best.Val instead of p.Val.
	pbest := it.Pop.Best()
	if pbest != nil && pbest.Best.Val < it.best.Val {
		it.best = pbest.Best
	}

	// Kill slow particles near global optimum.
	// This MUST go after the updating of the iterator's best position.
	for i, p := range it.Pop {
		if p.Kill(it.best, it.Xtol, it.Vtol) {
			it.Pop = append(it.Pop[:i], it.Pop[i+1:]...)
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

	s = "CREATE TABLE IF NOT EXISTS " + TblParticlesMeshed + " (particle INTEGER, iter INTEGER, val REAL"
	s += it.xdbsql("define")
	s += ");"

	_, err = it.Db.Exec(s)
	panicif(err)

	s = "CREATE TABLE IF NOT EXISTS " + TblParticlesBest + " (particle INTEGER, iter INTEGER, best REAL"
	s += it.xdbsql("define")
	s += ");"

	_, err = it.Db.Exec(s)
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

func (it *Iterator) updateDb(m mesh.Mesh) {
	if it.Db == nil {
		return
	}

	tx, err := it.Db.Begin()
	if err != nil {
		panic(err.Error())
	}
	defer tx.Commit()

	s0 := "INSERT INTO " + TblParticles + " (particle,iter,val" + it.xdbsql("x") + ") VALUES (?,?,?" + it.xdbsql("?") + ");"
	s0b := "INSERT INTO " + TblParticlesMeshed + " (particle,iter,best" + it.xdbsql("x") + ") VALUES (?,?,?" + it.xdbsql("?") + ");"
	s1 := "INSERT INTO " + TblParticlesBest + " (particle,iter,best" + it.xdbsql("x") + ") VALUES (?,?,?" + it.xdbsql("?") + ");"
	for _, p := range it.Pop {
		args := []interface{}{p.Id, it.count, p.Val}
		args = append(args, pos2iface(p.Pos())...)
		_, err := tx.Exec(s0, args...)
		panicif(err)

		args = []interface{}{p.Id, it.count, p.Best.Val}
		args = append(args, pos2iface(p.Best.Pos())...)
		_, err = tx.Exec(s1, args...)
		panicif(err)

		args = []interface{}{p.Id, it.count, p.Val}
		args = append(args, pos2iface(m.Nearest(p.Pos()))...)
		_, err = tx.Exec(s0b, args...)
		panicif(err)
	}

	s2 := "INSERT INTO " + TblBest + " (iter,val" + it.xdbsql("x") + ") VALUES (?,?" + it.xdbsql("?") + ");"
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

func vmaxfrombounds(low, up []float64) []float64 {
	vmax := make([]float64, len(low))
	for i := range vmax {
		// Eberhart et al. suggest this: (up-low)/2 - removing divide by two
		// seems to help swarm avoid premature convergence in difficult
		// problems.
		vmax[i] = (up[i] - low[i])
	}
	return vmax
}
