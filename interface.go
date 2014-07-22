package pswarm

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
		p.BestPos = p.Pos
	}
}

type Population []*Particle

func (pop Population) Best() (val float64, pos []float64) {
	val = pop[0].BestVal
	pos = pop[0].BestPos
	for _, p := range pop[1:] {
		if p.BestVal < val || pos == nil {
			val = p.BestVal
			pos = p.BestPos
		}
	}
	return val, pos
}

// Run a single iteration of the algorithm. combine/adjust particle numbers if
// necessary.
type Iterator interface {
	Iterate(pop Population, ob Objectiver, ev Evaler, mv Mover) (Population, error)
}

// evaler will need to project particles onto mesh. and will decide whether to
// parallellize or not.
// Must update particles' objective, and best objective, and best pos.
type Evaler interface {
	Eval(obj Objectiver, pop Population) error
}

type Objectiver interface {
	Objective(v []float64) (float64, error)
}

// mover tracks constraints:
//    ConstrA   *mat64.Dense
//    Constrb   *mat64.Dense
// Must update the particles' position, velocity,
type Mover interface {
	Move(p Population)
}
