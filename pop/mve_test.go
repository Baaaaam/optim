package pop

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

type test struct {
	Mat [][]float64
	Exp [][]float64
}

func TestSqrt(t *testing.T) {
	tests := []test{
		test{
			Mat: [][]float64{
				[]float64{4, 0},
				[]float64{0, 9},
			},
			Exp: [][]float64{
				[]float64{2, 0},
				[]float64{0, 3},
			},
		},
		test{
			Mat: [][]float64{
				[]float64{4, 1},
				[]float64{2, 9},
			},
			Exp: [][]float64{
				[]float64{1.9796219992668418, 0.20136625836861471},
				[]float64{0.40273251673722943, 2.9864532911099158},
			},
		},
	}

	for _, test := range tests {
		m := makemat(test.Mat)
		s := Sqrt(m)
		r, c := s.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				exp := test.Exp[i][j]
				got := s.At(i, j)
				if exp != got {
					t.Errorf("(%v,%v): expect %v, got %v", i, j, exp, got)
				}
			}
		}
	}
}

func makemat(vals [][]float64) *mat64.Dense {
	r := len(vals)
	c := len(vals[0])
	m := mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, vals[i][j])
		}
	}
	return m
}
