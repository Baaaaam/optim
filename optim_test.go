package optim

import (
	"errors"
	"math"
	"testing"
)

var tpoints = []Point{
	Point{Pos: []float64{1, 2, 3}, Val: 0},
	Point{Pos: []float64{1, 2, 4}, Val: 0},
	Point{Pos: []float64{1, 2, 5}, Val: 0},
	Point{Pos: []float64{1, 2, 6}, Val: 0},
	Point{Pos: []float64{1, 2, 7}, Val: 0},
}

type ObjTest struct {
	count int
	max   int
}

func (o *ObjTest) Objective(x []float64) (float64, error) {
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
	obj := &ObjTest{max: errcount}
	ev := SerialEvaler{}

	r, n, err := ev.Eval(obj, tpoints...)

	if len(r) != errcount {
		t.Errorf("returned wrong number of r: expected %v, got %v", errcount, len(r))
	}
	if n != errcount {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", errcount, n)
	}
	if err == nil {
		t.Errorf("did not propogate error through return")
	}
}

func TestCacheEvalerErr(t *testing.T) {
	errcount := 3
	obj := &ObjTest{max: errcount}
	ev := NewCacheEvaler(SerialEvaler{})

	r, n, err := ev.Eval(obj, tpoints...)

	if len(r) != errcount {
		t.Errorf("returned wrong number of r: expected %v, got %v", errcount, len(r))
	}
	if n != errcount {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", errcount, n)
	}
	if err == nil {
		t.Errorf("did not propogate error through return")
	}
}

func TestCacheEvaler(t *testing.T) {
	obj := &ObjTest{max: 100000}
	ev := NewCacheEvaler(SerialEvaler{})

	r1, n1, err1 := ev.Eval(obj, tpoints...)
	r2, n2, err2 := ev.Eval(obj, tpoints...)

	if v := len(r1) + len(r2); v != 2*len(tpoints) {
		t.Errorf("returned wrong number of results: expected %v, got %v", 2*len(tpoints), v)
	}
	if n1+n2 != len(tpoints) {
		t.Errorf("returned wrong evaluation count: expected %v, got %v", len(tpoints), n1+n2)
	}
	if err1 != nil || err2 != nil {
		t.Errorf("got unexpected (err2 and err2): %v and %v", err1, err2)
	}

	for i := range r1 {
		for j := range tpoints[i].Pos {
			if exp, got := tpoints[i].Pos[j], r1[i].Pos[j]; exp != got {
				t.Errorf("bad pos: expected %+v, got %+v", exp, got)
			}
			if exp, got := tpoints[i].Pos[j], r2[i].Pos[j]; exp != got {
				t.Errorf("bad cached pos: expected %+v, got %+v", exp, got)
			}
		}
	}
}
