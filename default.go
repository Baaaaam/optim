package pswarm

import (
	"fmt"
	"math/rand"
)

type SimpleObjectiver func([]float64) float64

func (so SimpleObjectiver) Objective(v []float64) (float64, error) { return so(v), nil }

type SimpleIter struct{}

func (it SimpleIter) Iterate(p Population, obj Objectiver, ev Evaler, mv Mover) (Population, error) {
	err := ev.Eval(obj, p)
	if err != nil {
		return nil, err
	}
	mv.Move(p)
	return p, nil
}

type SerialEvaler struct {
	StopOnErr bool
}

func (ev SerialEvaler) Eval(obj Objectiver, pop Population) error {
	for _, p := range pop {
		val, err := obj.Objective(p.Pos)
		if err != nil && ev.StopOnErr {
			return err
		}
		p.Update(val)
	}
	return nil
}

const (
	DefaultCognition = 0.5
	DefaultSocial    = 0.5
	DefaultInertia   = 0.6
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
