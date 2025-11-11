import Optlib.Autoformalization.Template.NesterovFirst_template

noncomputable section LASSO

open Set Real Matrix Finset

local notation "‖" x "‖₂" => @Norm.norm _ (PiLp.instNorm 2 fun _ ↦ ℝ) x
local notation "‖" x "‖₁" => @Norm.norm _ (PiLp.instNorm 1 fun _ ↦ ℝ) x
local notation "⟪" x ", " y "⟫" => @inner ℝ _ _ x y
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

class Nesterov_first_fix_stepsize_Lasso (pro : Lasso_problem A b mu) (x₀ : EuclideanSpace ℝ (Fin n)) where
  hl : pro.l > (0 : ℝ)
  x : ℕ → EuclideanSpace ℝ (Fin n)
  y : ℕ → EuclideanSpace ℝ (Fin n)
  w : ℕ → EuclideanSpace ℝ (Fin n)
  t : ℕ → ℝ
  γ : ℕ → ℝ
  oriy : y 0 = x 0
  initial : x 0 = x₀
  teq : ∀ n : ℕ, t n = 1 / pro.l
  γeq : ∀ n : ℕ, γ n = 2 / (2 + n)
  update1 : ∀ k : ℕ+, y k = x k + (γ k * (1 - γ (k - 1)) / (γ (k - 1))) • (x k - x (k - 1))
  update2 : ∀ k : ℕ,
    let grad : EuclideanSpace ℝ (Fin n) := Aᵀ *ᵥ (A *ᵥ y k - b)
    w k = y k - t k • grad
  update3 : ∀ k : ℕ, x (k + 1) =
    fun i => Real.sign (w k i) * (max (abs (w k i) - t k * mu) 0)

variable {pro : Lasso_problem A b mu} {x₀ : EuclideanSpace ℝ (Fin n)}

lemma Lasso_problem.hasGradient (self : Lasso_problem A b mu) :
    ∀ x, HasGradientAt self.f (Aᵀ *ᵥ (A *ᵥ x - b)) x := by
  unfold Lasso_problem.f
  apply affine_sq_gradient

lemma Lasso_problem.gradient_f (self : Lasso_problem A b mu) :
    ∀ x, gradient self.f x = Aᵀ *ᵥ (A *ᵥ x - b) := by
  exact fun x ↦ HasGradientAt.gradient (self.hasGradient _)

lemma Nesterov_first_fix_stepsize_Lasso.update_cor (self : Nesterov_first_fix_stepsize_Lasso pro x₀) :
    ∀ (k : ℕ), prox_prop (self.t k • pro.g) (self.y k - self.t k • gradient pro.f (self.y k)) (self.x (k + 1)) := by
  intro k
  unfold Lasso_problem.g
  apply norm_one_proximal
  · exact rfl
  · rw [self.teq k];simp [self.hl]
  · exact pro.hmu
  · intro i
    rw [pro.gradient_f]
    rw [← self.update2 k]
    rw [self.update3 k]

instance Nesterov_first_fix_stepsize_Lasso.Nesterov_first_fix_stepsize (self : Nesterov_first_fix_stepsize_Lasso pro x₀) :
    Nesterov_first_fix_stepsize (Lasso_problem.composite_problem pro) x₀ where
  hl := self.hl
  x := self.x
  y := self.y
  t := self.t
  γ := self.γ
  oriy := self.oriy
  initial := self.initial
  teq := self.teq
  γeq := self.γeq
  update1 := self.update1
  update2 := self.update_cor

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

theorem Lasso_convergence (alg : Nesterov_first_fix_stepsize_Lasso pro x₀)
    (xm : EuclideanSpace ℝ (Fin n)) (minφ : IsMinOn pro.target univ xm):
    ∀ (k : ℕ), pro.f (alg.x (k + 1)) + pro.g (alg.x (k + 1)) - pro.f xm - pro.g xm ≤
    2 * pro.l / (k + 2) ^ 2 * ‖x₀ - xm‖ ^ 2 := by
  intro k
  apply Nesterov_first_fix_stepsize_converge (alg := alg.Nesterov_first_fix_stepsize)
    pro.l alg.hl pro.diff_f pro.ConvexOn_f pro.lip_f pro.ConvexOn_g minφ

end LASSO
