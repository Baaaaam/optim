package pattern

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/mesh"
)

type ConstrPoller struct {
	Constr *mat64.Dense
	Limit  *mat64.Dense
	Step   float64
	curr   optim.Point
	m      *mesh.Infinite
}

func (cp *ConstrPoller) StepSize() float64 { return cp.Step }

func (cp *ConstrPoller) Mesh() mesh.Mesh {
	if cp.m == nil {
		cp.m = &mesh.Infinite{Origin: cp.curr.Pos(), Step: cp.Step}
	}
	return cp.m
}

func (cp *ConstrPoller) Poll(obj optim.Objectiver, ev optim.Evaler, from optim.Point) (success bool, best optim.Point, neval int, err error) {
	// calculate polling directions
	// multiply constr matrix againx point and collect violated constraints
	var direcs [][]float64
	cp.curr = from

	direcs = generateDirecs(from.Len())
	points := make([]optim.Point, 0, len(direcs))
	for i, dir := range direcs {
		pos := make([]float64, len(dir))
		for j, v := range dir {
			pos[j] = from.At(j) + cp.Step*v
		}
		p := optim.NewPoint(pos, math.Inf(1))

		// Loop until we get a point that doesn't violate any constraints.
		// First find the most violated constraint. Then project the offending
		// poll point onto the constraint plane. Repeat until no constraints
		// are violated.  Note that for discrete variable problems, this gives
		// you float poll points.
		for {
			var maxconstr []float64
			maxviol := 0.0
			maxlimit := 0.0
			x := mat64.NewDense(from.Len(), 1, p.Pos())
			x.Mul(cp.Constr, x)
			nr, _ := cp.Constr.Dims()
			for r := 0; r < nr; r++ {
				if diff := cp.Limit[r] - x.At(r, 0); diff < maxviol {
					maxviol = diff
					maxlimit = cp.Limit[r]
					maxconstr = cp.Constr.Row(nil, r)
				}
			}

			if maxconstr == nil {
				break
			} else {
				p = project(p, maxconstr, maxlimit)
			}
		}
		points = append(points, p)
	}

	// poll neighboring points
	results, n, err := ev.Eval(obj, points...)

	if err == nil || err == FoundBetterErr {
		err = nil
		for i := range results {
			if results[i].Val < cp.curr.Val {
				cp.curr = results[i]
			}
		}
		if cp.curr.Val < from.Val {
			return true, cp.curr, n, nil
		}
	} else if err != nil {
		return false, cp.curr, n, err
	}

	return false, cp.curr, n, nil
}

func project(p optim.Point, norm []float64, bound float64) optim.Point {
	pt := p.Pos()
	return optim.NewPoint(sub(pt, scale(norm, (bound+dot(norm, pt))/dot(norm, norm))), math.Inf(1))
}

func dot(a, b []float64) float64 {
	tot := 0.0
	for i := range a {
		tot += a[i] * b[i]
	}
	return tot
}

func add(a, b []float64) []float64 {
	c := make([]float64, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}
func sub(a, b []float64) []float64 {
	c := make([]float64, len(a))
	for i := range a {
		c[i] = a[i] - b[i]
	}
	return c
}

func scale(a []float64, mult float64) []float64 {
	c := make([]float64, len(a))
	for i := range a {
		c[i] = a[i] * mult
	}
	return c
}
