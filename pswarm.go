package pswarm

type Particle struct {
	Pos       []float64
	Vel       []float64
	Objective float64
	Best      float64
}

type Population interface {
	Particles() []Particle
	Best() (x []float64, f float64)
}

// Run a single iteration of the algorithm. combine/adjust particle numbers if
// necessary.
type Iterator interface {
	Iterate(pop Population, ob Objectiver, ev Evaler, mv Mover)
}

func Iterate(pop Population, ob Objectiver, ev Evaler, mv Mover) {
	ev.Eval(ob, pop.Particles()...)
	mv.Move(pop, pop.Particles()...)
}

// evaler will need to project particles onto mesh. and will decide whether to
// parallellize or not. Must update the particles' position, velocity,
// objective, and best objective.
type Evaler interface {
	Eval(obj Objectiver, ps ...Particle) float64
}

type Objectiver interface {
	Objective(v []float64) (float64, error)
}

// mover tracks constraints:
//    ConstrA   *mat64.Dense
//    Constrb   *mat64.Dense
type Mover interface {
	Move(pop Population, ps ...Particle)
}
