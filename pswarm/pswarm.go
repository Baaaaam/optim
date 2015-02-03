package pswarm

import (
	"database/sql"
	"fmt"
	"math"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

const (
	DefaultCognition = 0.5
	DefaultSocial    = 0.5
	DefaultInertia   = 0.9
)

const (
	TblParticles = "swarmparticles"
	TblBest      = "swarmbest"
)

type Particle struct {
	Id int
	optim.Point
	Vel  []float64
	Best optim.Point
}

func (p *Particle) Update(newp optim.Point) {
	p.Val = newp.Val
	if p.Val < p.Best.Val || p.Best.Len() == 0 {
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
		if p.Val < best.Val {
			best = p
		}
	}
	return best
}

type Option func(*Iterator)

func Vmax(vel float64) Option {
	return func(it *Iterator) {
		it.Mover.Vmax = vel
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
		it.Mover.Cognition = cognition
		it.Mover.Social = social
	}
}

func LinInertia(start, end float64, maxiter int) Option {
	return func(it *Iterator) {
		it.Mover.InertiaFn = func(iter int) float64 {
			return start - (start-end)*float64(iter)/float64(maxiter)
		}
	}
}

type Iterator struct {
	// KillDist is the distance from the global optimum below which particles
	// are killed.  Zero for never killing particles.  Large values result in
	// a particle being killed whenever it becomes the global optimum.
	KillDist float64
	Pop      Population
	optim.Evaler
	*Mover
	Db    *sql.DB
	count int
	best  optim.Point
}

func NewIterator(e optim.Evaler, m *Mover, pop Population, opts ...Option) *Iterator {
	if e == nil {
		e = optim.SerialEvaler{}
	}
	if m == nil {
		m = &Mover{Cognition: DefaultCognition, Social: DefaultSocial}
	}
	it := &Iterator{
		Pop:      pop,
		Evaler:   e,
		Mover:    m,
		best:     pop.Best().Point,
		KillDist: math.Inf(1),
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
	points := it.Pop.Points()
	if m != nil {
		for i, p := range points {
			points[i] = optim.Nearest(p, m)
		}
	}
	results, n, err := it.Evaler.Eval(obj, points...)
	if err != nil {
		return optim.Point{}, n, err
	}

	for i := range results {
		it.Pop[i].Update(results[i])
	}

	it.updateDb()
	it.Mover.Move(it.best, it.Pop)

	pbest := it.Pop.Best()
	if pbest != nil && pbest.Val < it.best.Val {
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

type Mover struct {
	Cognition float64
	Social    float64
	// Vmax is the speed limit for particles.  If not specified,
	// Vmax=1.5*currVel.
	Vmax      float64
	InertiaFn func(int) float64
	iter      int
}

func (mv *Mover) Move(best optim.Point, pop Population) {
	mv.iter++
	if mv.InertiaFn == nil {
		mv.InertiaFn = func(iter int) float64 {
			return DefaultInertia
		}
	}

	for _, p := range pop {
		vmax := mv.Vmax
		if mv.Vmax == 0 {
			// if no vmax is given, use 1.5 * current speed
			vmax = 1.5 * Speed(p.Vel)
		}

		w1 := optim.RandFloat()
		w2 := optim.RandFloat()
		// update velocity
		for i, currv := range p.Vel {
			p.Vel[i] = mv.InertiaFn(mv.iter)*currv +
				mv.Cognition*w1*(p.Best.At(i)-p.At(i)) +
				mv.Social*w2*(best.At(i)-p.At(i))
			if s := Speed(p.Vel); mv.Vmax > 0 && Speed(p.Vel) > mv.Vmax {
				for i := range p.Vel {
					p.Vel[i] *= vmax / s
				}
			}

		}

		// update position
		pos := make([]float64, p.Len())
		for i := range pos {
			pos[i] = p.At(i) + p.Vel[i]
		}
		p.Point = optim.NewPoint(pos, p.Val)
	}
}

func Speed(vel []float64) float64 {
	tot := 0.0
	for _, v := range vel {
		tot += v * v
	}
	return math.Sqrt(tot)
}

// TODO: remove all uses of this
func panicif(err error) {
	if err != nil {
		panic(err.Error())
	}
}
