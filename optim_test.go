package optim

import (
	"errors"
	"math"
	"testing"
)

const errcount = 3

var tpoint = Point{
	Pos: []float64{1, 2, 3},
	Val: 0,
}

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

	results, n, err := ev.Eval(obj, tpoint, tpoint, tpoint, tpoint, tpoint)
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

func TestCacheEvalerErr(t *testing.T) {
	obj := &ErrObj{}
	ev := NewCacheEvaler(SerialEvaler{})
	r1, n1, _ := ev.Eval(obj, tpoint)
	r2, n2, err := ev.Eval(obj, tpoint, tpoint, tpoint, tpoint)

	if v := len(r1) + len(r2); v != errcount {
		t.Errorf("returned wrong number of results: expected %v, got %v", errcount, v)
	}
	if n1+n2 != 1 {
		t.Errorf("returned wrong evaluation count: expected 1, got %v", n1+n2)
	}
	if err != nil {
		t.Errorf("failed to prevent extra evaluations")
	}
}
