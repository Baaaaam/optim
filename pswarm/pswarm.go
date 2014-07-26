package pswarm

import (
	"math/rand"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

type Particle struct {
	Id   int
	Pos  []float64
	Vel  []float64
	Val  float64
	Best optim.Point
}

func (p *Particle) Update(newval float64) {
	p.Val = newval
	if p.Val < p.Best.Val || p.Best.Pos == nil {
		p.Best.Val = p.Val
		p.Best.Pos = append([]float64{}, p.Pos...)
	}
}

type Population []*Particle

func (pop Population) Points() []optim.Point {
	points := make([]optim.Point, len(pop))
	for i := range pop {
		points[i].Pos = append([]float64{}, pop[i].Pos...)
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

type Mover interface {
	Move(p Population)
}

type SimpleIter struct {
	Pop Population
	optim.Evaler
	Mover
}

func (it SimpleIter) AddPoint(p optim.Point) {
	if p.Val < it.Pop.Best().Val {
		it.Pop[0].Best = p
	}
}

func (it SimpleIter) Iterate(obj optim.Objectiver, m mesh.Mesh) (best optim.Point, neval int, err error) {
	points := it.Pop.Points()
	if m != nil {
		for i := range points {
			points[i].Pos = m.Nearest(points[i].Pos)
		}
	}
	results, n, err := it.Evaler.Eval(obj, points...)
	if err != nil {
		return optim.Point{}, n, err
	}
	for i := range results {
		it.Pop[i].Update(results[i].Val)
	}

	it.Mover.Move(it.Pop)
	return it.Pop.Best(), n, nil
}

const (
	DefaultCognition = 0.5
	DefaultSocial    = 0.5
	DefaultInertia   = 0.9
)

type SimpleMover struct {
	Cognition float64
	Social    float64
	InertiaFn func() float64
	Rng       *rand.Rand
}

func (mv *SimpleMover) Move(pop Population) {
	if mv.Rng == nil {
		src := rand.NewSource(0)
		mv.Rng = rand.New(src)
	}
	if mv.InertiaFn == nil {
		mv.InertiaFn = func() float64 {
			return DefaultInertia
		}
	}

	bestPos := pop.Best().Pos

	for _, p := range pop {
		w1 := mv.Rng.Float64()
		w2 := mv.Rng.Float64()
		// update velocity
		for i, currv := range p.Vel {
			p.Vel[i] = mv.InertiaFn()*currv +
				mv.Cognition*w1*(p.Best.Pos[i]-p.Pos[i]) +
				mv.Social*w2*(bestPos[i]-p.Pos[i])
		}

		// update position
		for i := range p.Pos {
			p.Pos[i] += p.Vel[i]
		}
	}
}
