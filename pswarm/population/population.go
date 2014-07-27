package population

import (
	"math"
	"math/rand"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/pswarm"
)

func NewRandom(n int, lb, ub, minv, maxv []float64) pswarm.Population {
	if len(lb) != len(ub) || len(minv) != len(lb) || len(maxv) != len(lb) {
		panic("lb, ub, minv, and maxv vectors must all be same length")
	}

	pop := make(pswarm.Population, n)
	for i := 0; i < n; i++ {
		p := &pswarm.Particle{
			Id:  i,
			Vel: make([]float64, len(lb)),
		}
		pop[i] = p

		pos := make([]float64, len(lb))
		for j := 0; j < len(lb); j++ {
			pos[j] = lb[j] + (ub[j]-lb[j])*rand.Float64()
			p.Vel[j] = minv[j] + (maxv[j]-minv[j])*rand.Float64()
		}
		p.Point = optim.NewPoint(pos, math.Inf(1))
		p.Best = optim.NewPoint(pos, math.Inf(1))
	}
	return pop
}
