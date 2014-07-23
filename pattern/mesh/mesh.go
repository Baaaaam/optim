package mesh

import (
	"math"

	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
)

func init() {
	mat64.Register(goblas.Blasser)
}

type Mesh interface {
	Nearest(p []float64) []float64
}

type SimpleMesh struct {
	Origin   []float64
	Basis    *mat64.Dense
	Step     float64
	inverter *mat64.Dense
}

func (sm *SimpleMesh) Nearest(p []float64) []float64 {
	if sm.Step == 0 {
		panic("SimpleMesh has step size 0")
	} else if sm.Origin != nil && len(sm.Origin) != len(p) {
		panic("point passed to Nearest is has wrong length")
	}

	// set up origin, basis, and inverter if necessary
	if sm.Origin == nil {
		sm.Origin = make([]float64, len(p))
	}
	if sm.Basis == nil {
		sm.Basis = mat64.NewDense(len(sm.Origin), len(sm.Origin), nil)
		for i := 0; i < len(sm.Origin); i++ {
			sm.Basis.Set(i, i, 1)
		}
	}
	if sm.inverter == nil {
		sm.inverter = mat64.Inverse(sm.Basis)
	}

	// translate p based on origin and transform to new vector space
	newp := make([]float64, len(p))
	for i := range newp {
		newp[i] = p[i] - sm.Origin[i]
	}
	v := mat64.NewDense(len(sm.Origin), 1, newp)
	rotv := mat64.NewDense(len(sm.Origin), 1, nil)
	rotv.Mul(sm.inverter, v)

	// calculate nearest point
	nearest := mat64.NewDense(len(p), 1, nil)
	for i := range sm.Origin {
		n, rem := math.Modf(rotv.At(i, 0) / sm.Step)
		if rem/sm.Step > 0.5 {
			n++
		}
		nearest.Set(i, 0, float64(n)*sm.Step)
	}

	// transform back to standard space
	rotv.Mul(sm.Basis, nearest)
	return rotv.Col(nil, 0)
}
