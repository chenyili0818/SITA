import Optlib.Function.Proximal
import Mathlib.Analysis.Convex.Continuous
import Optlib.Autoformalization.Lemmas

section method

open Set

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
variable {xm x₀ : E} {s : Set E} {f : E → ℝ} {h : E → ℝ}
variable {t : ℝ} {x : ℕ → E} {L : NNReal}

class composite_problem (f : E → ℝ) (h : E → ℝ)

def composite_problem.target (_ : composite_problem f h) := f + h

class proximal_gradient (pro : composite_problem f h) (x₀ : E) :=
  t : ℝ
  x : ℕ → E
  update : ∀ k : ℕ, prox_prop (t • h) (x k - t • (gradient f) (x k)) (x (k + 1))
  initial : x 0 = x₀

variable {pro : composite_problem f h} {alg : proximal_gradient pro x₀}

theorem proximal_gradient_converge [FiniteDimensional ℝ E]
  (xm : E) (L : NNReal)
  (fconv : ConvexOn ℝ univ f) (hconv : ConvexOn ℝ univ h)
  (h₁ : Differentiable ℝ f) (h₂ : LipschitzWith L (gradient f))
  (tpos : 0 < alg.t) (step : alg.t ≤ 1 / L) (hL : L > (0 : ℝ)) :
  ∀ (k : ℕ+), (pro.target (alg.x k) - pro.target xm)
    ≤ 1 / (2 * k * alg.t) * ‖x₀ - xm‖ ^ 2 := by
  sorry

end method
