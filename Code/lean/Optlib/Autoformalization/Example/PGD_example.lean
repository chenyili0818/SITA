import Optlib.Autoformalization.Template.PGD_template

noncomputable section LASSO

open Set Real Matrix Finset

local notation "‖" x "‖₂" => @Norm.norm _ (PiLp.instNorm 2 fun _ ↦ ℝ) x
local notation "‖" x "‖₁" => @Norm.norm _ (PiLp.instNorm 1 fun _ ↦ ℝ) x
local notation "|‖" A "|‖" => ‖(Matrix.toEuclideanLin ≪≫ₗ LinearMap.toContinuousLinearMap) A‖₊

class Lasso_problem {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (b : (Fin m) → ℝ) (mu : ℝ) where
  (hA : A ≠ 0)
  (hmu : mu > 0)

variable {A : Matrix (Fin m) (Fin n) ℝ} {b : Fin m → ℝ} {mu : ℝ}

def Lasso_problem.f (_ : Lasso_problem A b mu) : EuclideanSpace ℝ (Fin n) → ℝ :=
  fun x ↦ 1 / 2 * ‖A *ᵥ x - b‖₂ ^ 2

def Lasso_problem.g (_ : Lasso_problem A b mu) : EuclideanSpace ℝ (Fin n) → ℝ :=
  fun x ↦ mu * ‖x‖₁

def Lasso_problem.target (self : Lasso_problem A b mu) : EuclideanSpace ℝ (Fin n) → ℝ :=
  fun x ↦ self.f x + self.g x

def Lasso_problem.l (_ : Lasso_problem A b mu) : NNReal := |‖(Aᵀ * A)|‖

instance Lasso_problem.composite_problem (self : Lasso_problem A b mu) :
    composite_problem self.f self.g where

class proximal_gradient_method_Lasso (pro : Lasso_problem A b mu) (x₀ : EuclideanSpace ℝ (Fin n)) where
  t : ℝ
  x : ℕ → EuclideanSpace ℝ (Fin n)
  y : ℕ → EuclideanSpace ℝ (Fin n)
  ht : t > 0
  update1 : ∀ k : ℕ,
    let grad : EuclideanSpace ℝ (Fin n) := Aᵀ *ᵥ (A *ᵥ x k - b)
    y k = x k - t • grad
  update2 : ∀ (k : ℕ), ∀ i, x (k + 1) i = (Real.sign (y k i) * (max (abs (y k i) - t * mu) 0))
  initial : x 0 = x₀

variable {pro : Lasso_problem A b mu} {x₀ : EuclideanSpace ℝ (Fin n)}

lemma Lasso_problem.hasGradient (self : Lasso_problem A b mu) :
    ∀ x, HasGradientAt self.f (Aᵀ *ᵥ (A *ᵥ x - b)) x := by
  unfold Lasso_problem.f
  apply affine_sq_gradient

lemma Lasso_problem.gradient_f (self : Lasso_problem A b mu) :
    ∀ x, gradient self.f x = Aᵀ *ᵥ (A *ᵥ x - b) := by
  exact fun x ↦ HasGradientAt.gradient (self.hasGradient _)

lemma proximal_gradient_method_Lasso.update_cor (self : proximal_gradient_method_Lasso pro x₀) :
    ∀ (k : ℕ), prox_prop (self.t • pro.g) (self.x k - self.t • gradient pro.f (self.x k)) (self.x (k + 1)) := by
  intro k
  unfold Lasso_problem.g
  apply norm_one_proximal
  · exact rfl
  · exact self.ht
  · exact pro.hmu
  · intro i
    rw [self.update2 k i]
    rw [self.update1 k]
    rw [pro.gradient_f]

instance proximal_gradient_method_Lasso.proximal_gradient_method (self : proximal_gradient_method_Lasso pro x₀) :
    proximal_gradient (Lasso_problem.composite_problem pro) x₀ where
  t := self.t
  x := self.x
  initial := self.initial
  update := self.update_cor

lemma Lasso_problem.ConvexOn_f (self : Lasso_problem A b mu) :
    ConvexOn ℝ Set.univ self.f  := by
  unfold Lasso_problem.f
  exact affine_sq_convex A b

lemma Lasso_problem.ConvexOn_g (self : Lasso_problem A b mu) :
    ConvexOn ℝ Set.univ self.g  := by
  unfold Lasso_problem.g
  apply ConvexOn.smul; linarith [self.hmu]; apply norm_one_convex

lemma Lasso_problem.diff_f (self : Lasso_problem A b mu) :
    Differentiable ℝ self.f := by
  exact fun x ↦ HasGradientAt.differentiableAt (self.hasGradient x)

lemma Lasso_problem.lip_f (self : Lasso_problem A b mu) :
    LipschitzWith self.l (gradient self.f) := by
  rw [lipschitzWith_iff_norm_sub_le]; intro x y
  rw [self.gradient_f, self.gradient_f]
  rw [← Matrix.mulVec_sub, ← sub_add, sub_add_eq_add_sub, sub_add_cancel]
  rw [← Matrix.mulVec_sub]
  simp
  apply Matrix.l2_opNorm_mulVec (Aᵀ * A)

lemma Lasso_problem.lpos (self : Lasso_problem A b mu) :
    self.l > 0 := by
  unfold Lasso_problem.l
  simp
  rw [Transpose_mul_self_eq_zero]
  exact self.hA

theorem Lasso_convergence (alg : proximal_gradient_method_Lasso pro x₀)
    (xm : EuclideanSpace ℝ (Fin n))
    (ht2 : alg.t ≤ 1 / pro.l):
    ∀ (k : ℕ+), (pro.target (alg.x k) - pro.target xm)
      ≤ 1 / (2 * k * alg.t) * ‖x₀ - xm‖ ^ 2 := by
  intro k
  apply proximal_gradient_converge (alg := alg.proximal_gradient_method)
    xm pro.l pro.ConvexOn_f pro.ConvexOn_g pro.diff_f pro.lip_f alg.ht ht2 pro.lpos k

end LASSO
