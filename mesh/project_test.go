package mesh

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
)

type projtest struct {
	A    [][]float64
	b    []float64
	x0   []float64
	want []float64
}

func TestOrthoProj(t *testing.T) {
	eps := 1e-10
	var tests []projtest = []projtest{
		{
			A: [][]float64{
				{2, 1},
			},
			b:    []float64{2},
			x0:   []float64{1, 2},
			want: []float64{0.20, 1.60},
		},
	}

	n := 1000
	xmax := 10 * float64(n)

	A := [][]float64{make([]float64, n)}
	b := []float64{xmax}
	x0 := make([]float64, n)
	want := make([]float64, n)
	for i := range A[0] {
		A[0][i] = 1
		x0[i] = xmax
		want[i] = 10
	}
	bigtest := projtest{A: A, b: b, x0: x0, want: want}
	tests = append(tests, bigtest)

	for n, test := range tests {
		adata := []float64{}
		for _, vals := range test.A {
			adata = append(adata, vals...)
		}
		A := mat64.NewDense(len(test.A), len(test.A[0]), adata)
		b := mat64.NewDense(len(test.b), 1, test.b)
		got := OrthoProj(test.x0, A, b)

		for i := range got {
			if diff := math.Abs(got[i] - test.want[i]); diff > eps {
				t.Errorf("test %v proj[%v]: want %v, got %v", n, i, test.want[i], got[i])
			}
		}
	}
}
