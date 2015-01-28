package pswarm

import (
	"math"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

const (
	DefaultCognition = 0.5
	DefaultSocial    = 0.5
	DefaultInertia   = 0.9
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

func (pop Population) Best() optim.Point {
	best := pop[0].Best
	for _, p := range pop[1:] {
		if p.Best.Val < best.Val {
			best = p.Best
		}
	}
	return best
}

type Iterator struct {
	Pop Population
	optim.Evaler
	*Mover
}

type Option func(*Iterator)

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

func NewIterator(e optim.Evaler, m *Mover, pop Population, opts ...Option) *Iterator {
	if e == nil {
		e = optim.SerialEvaler{}
	}
	if m == nil {
		m = &Mover{Cognition: DefaultCognition, Social: DefaultSocial}
	}
	it := &Iterator{
		Pop:    pop,
		Evaler: optim.SerialEvaler{},
		Mover:  m,
	}

	for _, opt := range opts {
		opt(it)
	}
	return it
}

func (it Iterator) AddPoint(p optim.Point) {
	if p.Val < it.Pop.Best().Val {
		it.Pop[0].Best = p
	}
}

func (it Iterator) Iterate(obj optim.Objectiver, m mesh.Mesh) (best optim.Point, neval int, err error) {
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

	it.Mover.Move(it.Pop)
	return it.Pop.Best(), n, nil
}

type Mover struct {
	Cognition float64
	Social    float64
	Vmax      float64
	InertiaFn func(int) float64
	iter      int
}

func (mv *Mover) Move(pop Population) {
	mv.iter++
	if mv.InertiaFn == nil {
		mv.InertiaFn = func(iter int) float64 {
			return DefaultInertia
		}
	}

	best := pop.Best()

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
				mv.Cognition*w1*(best.At(i)-p.At(i)) +
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
