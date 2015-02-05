package optim

import (
	"errors"
	"math"
	"sync"
	"testing"
)

func testpoints() []Point {
	return []Point{
		NewPoint([]float64{1, 2, 3}, 0),
		NewPoint([]float64{1, 2, 3}, 0), // duplicate point on purpose
		NewPoint([]float64{1, 2, 4}, 0),
		NewPoint([]float64{1, 2, 5}, 0),
		NewPoint([]float64{1, 2, 6}, 0),
		NewPoint([]float64{1, 2, 7}, 0),
	}
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

func TestUniqOfPoints(t *testing.T) {
	wantindexes := []int{0, 0, 2, 3, 4, 5}

	// check uniqof func
	points := testpoints()
	indexes := uniqof(points)
	for i, got := range indexes {
		if got != wantindexes[i] {
			t.Errorf("indexes[%v] WRONG: want %v, got %v", i, wantindexes[i], got)
		} else {
			t.Logf("indexes[%v] right: got %v", i, got)
		}
	}

	// check fillfromuniq func
	wantvals := []float64{0, 0, 2, 3, 4, 5}
	for i := range points {
		points[i].Val = float64(i)
	}
	fillfromuniq(indexes, points)
	for i, got := range points {
		if got.Val != wantvals[i] {
			t.Errorf("filled points[%v].Val WRONG: want %v, got %v", i, wantvals[i], got)
		} else {
			t.Logf("filled points[%v].Val right: got %v", i, got)
		}
	}
}

func TestSerialEvaler_DupPoints(t *testing.T) {
	obj := &ObjTest{max: 10000}
	ev := SerialEvaler{}

	tpoints := testpoints()
	r, _, _ := ev.Eval(obj, tpoints...)

	dups := testpoints()
	for i, p := range r {
		orig := dups[i]
		for k := 0; k < p.Len(); k++ {
			if p.At(k) != orig.At(k) {
				t.Errorf("result[%v] wrong point: want %v, got %v", orig, p)
				break
			}
		}
	}
}

func TestSerialEvalerErr(t *testing.T) {
	errcount := 3
	exprlen := errcount + 1 // we get an extra obj call due to duplicate avoidance
	expn := exprlen - 1
	obj := &ObjTest{max: errcount}
	ev := SerialEvaler{}

	tpoints := testpoints()
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
	tpoints := testpoints()
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
	tpoints := testpoints()
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
	tpoints := testpoints()
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
