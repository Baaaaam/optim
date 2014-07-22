package population

import (
	"math/rand"

	"github.com/rwcarlsen/gopswarm"
)

func NewRandom(n int, lb, ub, minv, maxv []float64) pswarm.Population {
	if len(lb) != len(ub) || len(minv) != len(lb) || len(maxv) != len(lb) {
		panic("lb, ub, minv, and maxv vectors must all be same length")
	}

	pop := make(pswarm.Population, n)
	for i := 0; i < n; i++ {
		p := &pswarm.Particle{
			Id:  i,
			Pos: make([]float64, len(lb)),
			Vel: make([]float64, len(lb)),
		}
		pop[i] = p
		for j := 0; j < len(lb); j++ {
			p.Pos[j] = lb[j] + (ub[j]-lb[j])*rand.Float64()
			p.Vel[j] = minv[j] + (maxv[j]-minv[j])*rand.Float64()
		}
	}
	return pop
}
