package optim

import "fmt"

// evaler will need to project particles onto mesh. and will decide whether to
// parallellize or not.
// Must update particles' objective, and best objective, and best pos.
type Evaler interface {
	Eval(obj Objectiver, points [][]float64) (vals []float64, err error)
}

type Objectiver interface {
	Objective(v []float64) (float64, error)
}

type SerialEvaler struct {
	StopOnErr bool
}

func (ev SerialEvaler) Eval(obj Objectiver, points [][]float64) (vals []float64, err error) {
	vals = make([]float64, len(points))
	for i, p := range points {
		vals[i], err = obj.Objective(p)
		if err != nil && ev.StopOnErr {
			return nil, err
		}
	}
	return vals, nil
}

type SimpleObjectiver func([]float64) float64

func (so SimpleObjectiver) Objective(v []float64) (float64, error) { return so(v), nil }

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
