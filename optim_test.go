package optim

import (
	"errors"
	"math"
	"testing"
)

const errcount = 3

type ErrObj struct {
	count int
}

func (o *ErrObj) Objective(x []float64) (float64, error) {
	o.count++
	if o.count >= errcount {
		return math.Inf(1), errors.New("fake error")
	}
	return 0, nil
}

func TestSerialEvalerErr(t *testing.T) {
	obj := &ErrObj{}
	ev := SerialEvaler{}

	results, n, err := ev.Eval(obj, Point{}, Point{}, Point{}, Point{}, Point{})
	if len(results) != errcount {
		t.Errorf("returned wrong number of results: expected %v, got %v", errcount, len(results))
	}
	if n != errcount {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", errcount, n)
	}
	if err == nil {
		t.Errorf("did not propogate error through return")
	}

}
