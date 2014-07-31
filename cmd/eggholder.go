package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/rwcarlsen/optim"
	"github.com/rwcarlsen/optim/bench"
	"github.com/rwcarlsen/optim/pattern"
)

const maxiter = 500000

func main() {
	rand.Seed(time.Now().Unix())
	fn := bench.Eggholder{}
	optimum := fn.Optima()[0].Val

	nsuccess := 0
	ntrials := 100
	for n := 0; n < ntrials; n++ {
		best := optim.Point{Val: math.Inf(1)}
		tot := 0
		success := false
		for tot < maxiter {
			it := buildIter(fn)
			subbest, n, _ := bench.Benchmark(it, fn, .01, maxiter)
			tot += n
			if subbest.Val < best.Val {
				best = subbest
			}
			if math.Abs(best.Val-optimum) < math.Abs(optimum*.01) {
				success = true
				nsuccess++
				break
			}
		}
		if success {
			fmt.Printf("Succeeded (%v evals):\n", tot)
		} else {
			fmt.Printf("Failed (%v evals):\n", tot)
		}
		fmt.Printf("    optimum: %+v\n", fn.Optima()[0])
		fmt.Printf("    best: %+v\n", best)
	}
	fmt.Printf("%v%% succeeded\n", float64(nsuccess)/float64(ntrials)*100)
}

func initialpoint(low, up []float64) optim.Point {
	max, min := up[0], low[0]
	pos := make([]float64, len(low))
	for i := range low {
		pos[i] = rand.Float64()*(max-min) + min
	}
	return optim.NewPoint(pos, math.Inf(1))
}

func buildIter(fn bench.Func) *pattern.Iterator {
	low, up := fn.Bounds()
	max, min := up[0], low[0]

	ev := optim.SerialEvaler{}
	s := pattern.NullSearcher{}
	p := &pattern.CompassPoller{
		Step: (max - min) / 5,
	}

	start := initialpoint(fn.Bounds())
	return pattern.NewIterator(start, ev, p, s)
}
