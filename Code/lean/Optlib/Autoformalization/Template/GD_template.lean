import Optlib.Function.Lsmooth

noncomputable section gradient_descent

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

class unconstrained_problem (f : E → ℝ)

variable {f : E → ℝ} {pro : unconstrained_problem f}

class Gradient_Descent (pro : unconstrained_problem f) (x0 : E) :=
  (x : ℕ → E) (t : ℝ) (l : NNReal)
  (update : ∀ k : ℕ, x (k + 1) = x k - t • (gradient f) (x k))
  (hl : l > 0) (step₁ : t > 0) (initial : x 0 = x0)

open InnerProductSpace Set

variable {xm x₀ : E}

variable {alg : Gradient_Descent pro x₀}

-- using the point version for the certain iteration of the gradient method
lemma point_descent_for_convex
    (diff : Differentiable ℝ f) (smooth : LipschitzWith alg.l (gradient f))
    (hfun : ConvexOn ℝ Set.univ f) (step₂ : alg.t ≤ 1 / alg.l) (xm : E) :
    ∀ k : ℕ, f (alg.x (k + 1)) ≤ f xm + 1 / ((2 : ℝ) * alg.t)
      * (‖alg.x k - xm‖ ^ 2 - ‖alg.x (k + 1) - xm‖ ^ 2) := by
  sorry

-- the O(1/t) descent property of the gradient method
lemma gradient_method_convergence
    (diff : Differentiable ℝ f) (smooth : LipschitzWith alg.l (gradient f))
    (hfun : ConvexOn ℝ Set.univ f) (step₂ : alg.t ≤ 1 / alg.l) (xm : E) :
    ∀ k : ℕ, f (alg.x (k + 1)) - f xm ≤ 1 / (2 * (k + 1) * alg.t) * ‖x₀ - xm‖ ^ 2 := by
  sorry

end gradient_descent
