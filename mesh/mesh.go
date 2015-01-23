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

// Mesh is an interface for projecting arbitrary dimensional points onto some
// kind of (potentially discrete) mesh.
type Mesh interface {
	// Nearest returns the nearest
	Nearest(p []float64) []float64
}

// Inifinite is a grid-based, linear-axis mesh that extends in all dimensions
// without bounds.  The length of Origin defines the dimensionality of the
// mesh. If Origin == nil, the dimensionality is set by the first call to
// Nearest.  If Basis == nil, a unit basis (the identify matrix) is used.  If
// Step == 0, then the mesh represents continuous space and the Nearest method
// just returns the point passed to it.
type Infinite struct {
	Origin []float64
	// Basis contains a set of row vectors defining the directions of each
	// mesh axis for the car.
	Basis *mat64.Dense
	// Step represents the discretization or grid size of the mesh.
	Step     float64
	inverter *mat64.Dense
}

// Nearest returns the nearest grid point to p by rounding each dimensional
// position to the nearest grid point.  If the mesh basis is not the identity
// matrix, then p is transformed to the mesh basis before rounding and then
// retransformed back.
func (sm *Infinite) Nearest(p []float64) []float64 {
	if sm.Step == 0 {
		return append([]float64{}, p...)
	} else if l := len(sm.Origin); l != 0 && l != len(p) {
		panic(fmt.Sprintf("origin len %v incompatible with point len %v", l, len(p)))
	}

	// set up origin and inverter matrix if necessary
	if len(sm.Origin) == 0 {
		sm.Origin = make([]float64, len(p))
	}
	if sm.Basis != nil && sm.inverter == nil {
		sm.inverter = mat64.Inverse(sm.Basis)
	}

	// translate p based on origin and transform to new vector space
	newp := make([]float64, len(p))
	for i := range newp {
		newp[i] = p[i] - sm.Origin[i]
	}
	v := mat64.NewDense(len(sm.Origin), 1, newp)
	rotv := v
	if sm.inverter != nil {
		rotv.Mul(sm.inverter, v)
	}

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
	if sm.Basis != nil {
		rotv.Mul(sm.Basis, nearest)
	} else {
		rotv = nearest
	}
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

// Nearest returns the nearest bounded grid point to p by sliding each
// dimensional position to the nearest value inside bounds and then rounding
// to the nearest grid point.  If the mesh basis is not the identity matrix,
// then p is transformed to the mesh basis before rounding and then
// retransformed back.
func (m *Bounded) Nearest(p []float64) []float64 {
	pdup := make([]float64, len(p))
	copy(pdup, p)
	for i := range pdup {
		pdup[i] = math.Max(m.Lower[i], pdup[i])
		pdup[i] = math.Min(m.Upper[i], pdup[i])
	}
	return m.core.Nearest(pdup)
}
