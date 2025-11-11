import Optlib.Function.Proximal
import Mathlib.Analysis.Convex.Continuous
import Optlib.Autoformalization.Lemmas

section method

open Set

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
variable {xm x₀ : E} {s : Set E} {f : E → ℝ} {h : E → ℝ}

class composite_problem (f : E → ℝ) (h : E → ℝ)

def composite_problem.target (_ : composite_problem f h) := f + h

class Nesterov_first (pro : composite_problem f h) (x₀ : E) :=
  x : ℕ → E
  y : ℕ → E
  t : ℕ → ℝ
  γ : ℕ → ℝ
  oriy : y 0 = x 0
  initial : x 0 = x₀
  update1 : ∀ k : ℕ+, y k = x k + (γ k * (1 - γ (k - 1)) / γ (k - 1)) • (x k - x (k - 1))
  update2 : ∀ k : ℕ, prox_prop (t k • h) (y k - t k • gradient f (y k)) (x (k + 1))

class Nesterov_first_fix_stepsize (pro : composite_problem f h) (x₀ : E) :=
  L : NNReal
  hl : L > (0 : ℝ)
  x : ℕ → E
  y : ℕ → E
  t : ℕ → ℝ
  γ : ℕ → ℝ
  oriy : y 0 = x 0
  initial : x 0 = x₀
  teq : ∀ n : ℕ, t n = 1 / L
  γeq : ∀ n : ℕ, γ n = 2 / (2 + n)
  update1 : ∀ (k : ℕ+), y k = x k + (γ k * (1 - γ (k - 1)) / (γ (k - 1))) • (x k - x (k - 1))
  update2 : ∀ k : ℕ, prox_prop (t k • h) (y k - t k • gradient f (y k)) (x (k + 1))

instance Nesterov_first_fix_stepsize.Nesterov_first {pro : composite_problem f h}
   [self : Nesterov_first_fix_stepsize pro x₀] : Nesterov_first pro x₀ where
  x := self.x
  y := self.y
  t := self.t
  γ := self.γ
  oriy := self.oriy
  initial := self.initial
  update1 := self.update1
  update2 := self.update2

variable {pro : composite_problem f h} {alg : Nesterov_first pro x₀}

theorem Nesterov_first_converge (L : NNReal) (hl : L > (0 : ℝ))
  (h₁ : Differentiable ℝ f) (convf : ConvexOn ℝ univ f)
  (h₂ : LipschitzWith L (gradient f)) (convh : ConvexOn ℝ univ h)
  (oriγ : alg.γ 0 = 1)
  (cond : ∀ n : ℕ+, (1 - alg.γ n) * alg.t n / alg.γ n ^ 2 ≤ alg.t (n - 1) / alg.γ (n - 1) ^ 2)
  (tbound : ∀ k : ℕ, 0 < alg.t k ∧ alg.t k ≤ 1 / L) (γbound : ∀ n : ℕ, 0 < alg.γ n ∧ alg.γ n ≤ 1)
  (minφ : IsMinOn pro.target univ xm) :
    ∀ k, f (alg.x (k + 1)) + h (alg.x (k + 1)) -
    f xm - h xm ≤ (alg.γ k) ^ 2 / (2 * alg.t k) * ‖x₀ - xm‖ ^ 2 := by
  sorry

variable {alg : Nesterov_first_fix_stepsize pro x₀}

theorem Nesterov_first_fix_stepsize_converge (L : NNReal) (hl : L > (0 : ℝ))
  (h₁ : Differentiable ℝ f) (convf: ConvexOn ℝ univ f)
  (h₂ : LipschitzWith L (gradient f)) (convh : ConvexOn ℝ univ h)
  (minφ : IsMinOn pro.target univ xm):
    ∀ (k : ℕ), f (alg.x (k + 1)) + h (alg.x (k + 1)) - f xm - h xm ≤
    2 * L / (k + 2) ^ 2 * ‖x₀ - xm‖ ^ 2 := by
  sorry

end method
