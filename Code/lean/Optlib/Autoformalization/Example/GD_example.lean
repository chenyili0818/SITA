import Optlib.Autoformalization.Template.GD_template
import Optlib.Autoformalization.Lemmas

open Set Real Matrix Finset Filter Bornology BigOperators Topology

noncomputable section LeastSquares

local notation "‖" x "‖₂" => @Norm.norm _ (PiLp.instNorm 2 fun _ ↦ ℝ) x
local notation "‖" x "‖₁" => @Norm.norm _ (PiLp.instNorm 1 fun _ ↦ ℝ) x
local notation "|‖" A "|‖" => ‖(Matrix.toEuclideanLin ≪≫ₗ LinearMap.toContinuousLinearMap) A‖₊

class LeastSquares_problem {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) where
  hA : A ≠ 0

variable {A : Matrix (Fin m) (Fin n) ℝ} {b : Fin m → ℝ}

def LeastSquares_problem.f (_ : LeastSquares_problem A b) : EuclideanSpace ℝ (Fin n) → ℝ :=
  fun x ↦ 1 / 2 * ‖A *ᵥ x - b‖₂ ^ 2

def LeastSquares_problem.l (_ : LeastSquares_problem A b) : NNReal := |‖Aᵀ * A|‖

instance LeastSquares_problem.unconstrained_problem (self : LeastSquares_problem A b) :
    unconstrained_problem self.f where

variable {pro : LeastSquares_problem A b}

class LeastSquares_GD (pro : LeastSquares_problem A b) (x₀ : EuclideanSpace ℝ (Fin n)) where
  t : ℝ
  x : ℕ → EuclideanSpace ℝ (Fin n)
  ht : t > 0
  update : ∀ k : ℕ,
    let gra : EuclideanSpace ℝ (Fin n) := Aᵀ *ᵥ (A *ᵥ x k - b)
    x (k + 1) = x k - t • gra
  initial : x 0 = x₀

lemma LeastSquares_problem.hasGradient (self : LeastSquares_problem A b) :
    ∀ x, HasGradientAt self.f (Aᵀ *ᵥ (A *ᵥ x - b)) x := by
  unfold LeastSquares_problem.f
  apply affine_sq_gradient

lemma LeastSquares_problem.gradient_f (self : LeastSquares_problem A b) :
    ∀ x, gradient self.f x = Aᵀ *ᵥ (A *ᵥ x - b) := by
  exact fun x ↦ HasGradientAt.gradient (self.hasGradient _)

lemma LeastSquares_problem.pos_l (self : LeastSquares_problem A b) :
    self.l > 0 := by
  unfold LeastSquares_problem.l; simp
  rw [Transpose_mul_self_eq_zero]
  exact self.hA

lemma LeastSquares_GD.update_cor (self : LeastSquares_GD pro x₀) :
    ∀ (k : ℕ), self.x (k + 1) = self.x k - self.t • gradient pro.f (self.x k) := by
  intro k
  rw [self.update k]
  rw [pro.gradient_f]

instance LeastSquares_GD.Gradient_Descent (self : LeastSquares_GD pro x₀) :
    Gradient_Descent (LeastSquares_problem.unconstrained_problem pro) x₀ where
  x := self.x
  t := self.t
  l := pro.l
  update := self.update_cor
  hl := pro.pos_l
  step₁ := self.ht
  initial := self.initial

variable {alg : LeastSquares_GD pro x₀}

lemma LeastSquares_GD.diff (_ : LeastSquares_GD pro x₀) :
    Differentiable ℝ pro.f := by
  exact (fun x ↦ HasGradientAt.differentiableAt (affine_sq_gradient A b x))

lemma LeastSquares_GD.smooth (_ : LeastSquares_GD pro x₀) :
    LipschitzWith pro.l (gradient pro.f) := by
  rw [lipschitzWith_iff_norm_sub_le]; intro x y
  rw [pro.gradient_f, pro.gradient_f]
  rw [← Matrix.mulVec_sub, ← sub_add, sub_add_eq_add_sub, sub_add_cancel]
  rw [← Matrix.mulVec_sub]
  unfold LeastSquares_problem.l
  simp
  apply Matrix.l2_opNorm_mulVec (Aᵀ * A)

lemma LeastSquares_GD.convex (_ : LeastSquares_GD pro x₀) :
    ConvexOn ℝ Set.univ pro.f := by
  unfold LeastSquares_problem.f
  exact affine_sq_convex A b

lemma LeastSquares_GD.point_descent
    (step₂ : alg.t ≤ 1 / pro.l) (xm : EuclideanSpace ℝ (Fin n)) :
    ∀ k : ℕ, pro.f (alg.x (k + 1)) ≤ pro.f xm + 1 / ((2 : ℝ) * alg.t)
      * (‖alg.x k - xm‖ ^ 2 - ‖alg.x (k + 1) - xm‖ ^ 2) := by
  apply point_descent_for_convex (alg := alg.Gradient_Descent) alg.diff alg.smooth alg.convex step₂ xm

-- the O(1/t) descent property of the gradient method
lemma LeastSquares_GD.gradient_method
    (step₂ : alg.t ≤ 1 / pro.l) (xm : EuclideanSpace ℝ (Fin n)) :
    ∀ k : ℕ, pro.f (alg.x (k + 1)) - pro.f xm ≤ 1 / (2 * (k + 1) * alg.t) * ‖x₀ - xm‖ ^ 2 := by
  apply gradient_method_convergence (alg := alg.Gradient_Descent) alg.diff alg.smooth alg.convex step₂ xm

end LeastSquares
