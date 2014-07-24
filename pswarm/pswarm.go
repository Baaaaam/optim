package pswarm

import (
	"math/rand"

	"github.com/rwcarlsen/optim"
)

type Particle struct {
	Id      int
	Pos     []float64
	Vel     []float64
	Val     float64
	BestVal float64
	BestPos []float64
}

func (p *Particle) Update(newval float64) {
	p.Val = newval
	if p.Val < p.BestVal || p.BestPos == nil {
		p.BestVal = p.Val
		p.BestPos = append([]float64{}, p.Pos...)
	}
}

type Population []*Particle

func (pop Population) Points() [][]float64 {
	points := make([][]float64, len(pop))
	for i := range pop {
		points[i] = append([]float64{}, pop[i].Pos...)
	}
	return points
}

func (pop Population) Best() (val float64, pos []float64) {
	val = pop[0].BestVal
	pos = pop[0].BestPos
	for _, p := range pop[1:] {
		if p.BestVal < val {
			val = p.BestVal
			pos = p.BestPos
		}
	}
	return val, pos
}

// Run a single iteration of the algorithm. combine/adjust particle numbers if
// necessary.
type Iterator interface {
	Iterate(pop Population, ob optim.Objectiver, ev optim.Evaler, mv Mover) (Population, error)
}

// mover tracks constraints:
//    ConstrA   *mat64.Dense
//    Constrb   *mat64.Dense
// Must update the particles' position, velocity,
type Mover interface {
	Move(p Population)
}

type SimpleIter struct {
	Pop Population
	optim.Evaler
	Mover
}

func (it SimpleIter) Iterate(obj optim.Objectiver) (best optim.Point, neval int, err error) {
	vals, n, err := it.Evaler.Eval(obj, it.Pop.Points()...)
	if err != nil {
		return optim.Point{}, n, err
	}
	for i := range vals {
		it.Pop[i].Update(vals[i])
	}

	it.Mover.Move(it.Pop)
	val, pos := it.Pop.Best()
	return optim.Point{Pos: pos, Val: val}, n, nil
}

const (
	DefaultCognition = 0.5
	DefaultSocial    = 0.5
	DefaultInertia   = 0.8
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

	_, bestPos := pop.Best()

	for _, p := range pop {
		w1 := mv.Rng.Float64()
		w2 := mv.Rng.Float64()
		// update velocity
		for i, currv := range p.Vel {
			p.Vel[i] = mv.InertiaFn()*currv +
				mv.Cognition*w1*(p.BestPos[i]-p.Pos[i]) +
				mv.Social*w2*(bestPos[i]-p.Pos[i])
		}

		// update position
		for i := range p.Pos {
			p.Pos[i] += p.Vel[i]
		}
	}
}
