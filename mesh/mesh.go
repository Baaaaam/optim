package mesh

import (
	"fmt"
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

type Infinite struct {
	Origin   []float64
	Basis    *mat64.Dense
	Step     float64
	inverter *mat64.Dense
}

func (sm *Infinite) Nearest(p []float64) []float64 {
	if sm.Step == 0 {
		return append([]float64{}, p...)
	} else if l := len(sm.Origin); l != 0 && l != len(p) {
		panic(fmt.Sprintf("origin len %v incompatible with point len %v", l, len(p)))
	}

	// set up origin, basis, and inverter if necessary
	if len(sm.Origin) == 0 {
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

type Bounded struct {
	Lower []float64
	Upper []float64
	core  Mesh
}

func NewBounded(m Mesh, lower, upper []float64) *Bounded {
	if len(lower) != len(upper) {
		panic("mesh lower and upper bound vectors have difference lengths")
	} else { // force panic if bounds lengths don't math mesh m's # dims
		m.Nearest(lower)
	}
	return &Bounded{
		Lower: lower,
		Upper: upper,
		core:  m,
	}
}

func (m *Bounded) Nearest(p []float64) []float64 {
	pdup := make([]float64, len(p))
	copy(pdup, p)
	for i := range pdup {
		pdup[i] = math.Max(m.Lower[i], pdup[i])
		pdup[i] = math.Min(m.Upper[i], pdup[i])
	}
	return m.core.Nearest(pdup)
}
