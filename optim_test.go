package optim

import (
	"errors"
	"math"
	"sync"
	"testing"
)

var tpoints = []Point{
	NewPoint([]float64{1, 2, 3}, 0),
	NewPoint([]float64{1, 2, 3}, 0), // duplicate point on purpose
	NewPoint([]float64{1, 2, 4}, 0),
	NewPoint([]float64{1, 2, 5}, 0),
	NewPoint([]float64{1, 2, 6}, 0),
	NewPoint([]float64{1, 2, 7}, 0),
}

type ObjTest struct {
	count int
	max   int
	sync.Mutex
}

func (o *ObjTest) Objective(x []float64) (float64, error) {
	o.Lock()
	defer o.Unlock()

	o.count++
	if o.count >= o.max {
		return math.Inf(1), errors.New("fake error")
	}
	tot := 0.0
	for _, v := range x {
		tot += v
	}
	return tot, nil
}

func TestSerialEvalerErr(t *testing.T) {
	errcount := 3
	exprlen := errcount + 1 // we get an extra obj call due to duplicate avoidance
	expn := exprlen - 1
	obj := &ObjTest{max: errcount}
	ev := SerialEvaler{}

	r, n, err := ev.Eval(obj, tpoints...)

	if len(r) != exprlen {
		// if this fires, duplicate point avoidance may be broken
		t.Errorf("returned wrong number of results: expected %v, got %v", exprlen, len(r))
	}
	if n != expn {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", expn, n)
	}
	if err == nil {
		t.Errorf("did not propagate error through return")
	}

	// exclude last entry in r because it was the error'd obj evaluation
	for i, p := range r[:len(r)-1] {
		expobj := 0.0
		for _, v := range tpoints[i].Pos() {
			expobj += v
		}
		if p.Val != expobj {
			t.Errorf("point %v (%v) objective value: expected %v, got %v", i, tpoints[i].Pos(), expobj, p.Val)
		}
	}
}

func TestParallelEvalerErr(t *testing.T) {
	errcount := 4
	exprlen := len(tpoints)
	expn := exprlen - 1
	obj := &ObjTest{max: errcount}
	ev := ParallelEvaler{}

	r, n, err := ev.Eval(obj, tpoints...)

	// parallel always evaluates all points
	if len(r) != exprlen {
		t.Errorf("returned wrong number of results: expected %v, got %v", exprlen, len(r))
	}
	if n == len(tpoints) {
		t.Errorf("failed to avoid evaluation of duplicate points", errcount, n)
	}
	if n != expn {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", errcount, n)
	}
	if err == nil {
		t.Errorf("did not propagate error through return")
	}

	for i, p := range r[:errcount-1] {
		expobj := 0.0
		for _, v := range tpoints[i].Pos() {
			expobj += v
		}
		if p.Val != expobj {
			t.Errorf("point %v (%v) objective value: expected %v, got %v", i, tpoints[i].Pos(), expobj, p.Val)
		}
	}
}

func TestCacheEvalerErr(t *testing.T) {
	errcount := 3
	exprlen := errcount + 1 // we get an extra obj call due to duplicate avoidance
	expn := exprlen - 1
	obj := &ObjTest{max: errcount}
	ev := NewCacheEvaler(SerialEvaler{})

	r, n, err := ev.Eval(obj, tpoints...)

	if len(r) != exprlen {
		t.Errorf("returned wrong number of r: expected %v, got %v", exprlen, len(r))
	}
	if n != expn {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", expn, n)
	}
	if err == nil {
		t.Errorf("did not propogate error through return")
	}
}

func TestCacheEvaler(t *testing.T) {
	obj := &ObjTest{max: 100000}
	ev := NewCacheEvaler(SerialEvaler{})
	expn := len(tpoints) - 1

	r1, n1, err1 := ev.Eval(obj, tpoints...)
	r2, n2, err2 := ev.Eval(obj, tpoints...)

	if v := len(r1) + len(r2); v != 2*len(tpoints) {
		t.Errorf("returned wrong number of results: expected %v, got %v", 2*len(tpoints), v)
	}
	if n1+n2 != expn {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", expn, n1+n2)
	}
	if err1 != nil || err2 != nil {
		t.Errorf("got unexpected err (err1 and err2): %v and %v", err1, err2)
	}

	for i := range r1 {
		for j := range tpoints[i].Pos() {
			if exp, got := tpoints[i].At(j), r1[i].At(j); exp != got {
				t.Errorf("bad pos: expected %+v, got %+v", exp, got)
			}
			if exp, got := tpoints[i].At(j), r2[i].At(j); exp != got {
				t.Errorf("bad cached pos: expected %+v, got %+v", exp, got)
			}
		}
	}
}
