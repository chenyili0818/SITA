import Optlib.Autoformalization.Template.ADMM_template
import Optlib.Autoformalization.Lemmas

noncomputable section Lasso

open Set Real Matrix Finset ContinuousLinearMap Continuous Filter

local notation "‖" x "‖₂" => @Norm.norm _ (PiLp.instNorm 2 fun _ ↦ ℝ) x
local notation "‖" x "‖₁" => @Norm.norm _ (PiLp.instNorm 1 fun _ ↦ ℝ) x

variable {m n : ℕ} {μ : ℝ} {A : Matrix (Fin m) (Fin n) ℝ} {b : EuclideanSpace ℝ (Fin m)}

class Lasso (μ : ℝ) (A : Matrix (Fin m) (Fin n) ℝ) (b : EuclideanSpace ℝ (Fin m)) where
  hmu : μ > 0
  hA : A ≠ 0

def Lasso.f₁ (_ : Lasso μ A b) : (EuclideanSpace ℝ (Fin n)) → ℝ :=
    fun x ↦
      let Ax : EuclideanSpace ℝ (Fin m) := A *ᵥ x
      (1 / 2) * ‖Ax - b‖₂ ^ 2

def Lasso.f₂ (_ : Lasso μ A b) : (EuclideanSpace ℝ (Fin n)) → ℝ := fun x ↦ μ * ‖x‖₁

def Lasso.A₁ (_ : Lasso μ A b) : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n) := 1

def Lasso.A₂ (_ : Lasso μ A b) : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n) := -1

def Lasso.bv (_ : Lasso μ A b) : EuclideanSpace ℝ (Fin n) := 0

instance Lasso.isProblem (self : Lasso μ A b) :
    Problem self.f₁ self.f₂ self.A₁ self.A₂ self.bv where

lemma Lasso.injA₁ (self : Lasso μ A b) : Function.Injective self.A₁ :=
    fun _ _ a ↦ a

lemma Lasso.injA₂ (self : Lasso μ A b) : Function.Injective self.A₂ := by
    intro x y h; simp [Lasso.A₂] at h; assumption

lemma Lasso.f₁_lsc (self : Lasso μ A b) : LowerSemicontinuous self.f₁ := by
  unfold Lasso.f₁
  apply lowerSemicontinuous
  rw [continuous_iff_continuousAt]
  intro x
  apply (affine_sq_gradient A b x).continuousAt

lemma Lasso.f₁_conv (self : Lasso μ A b) : ConvexOn ℝ univ self.f₁ := by
  apply affine_sq_convex

lemma Lasso.f₂_lsc (self : Lasso μ A b) : LowerSemicontinuous self.f₂ :=
  lowerSemicontinuous (Continuous.mul continuous_const (norm continuous_id'))

lemma Lasso.f₂_conv (self : Lasso μ A b) : ConvexOn ℝ univ self.f₂ := by
  unfold Lasso.f₂
  apply ConvexOn.smul (by linarith [self.hmu])
  exact norm_one_convex

lemma Lasso.uniq_subX (self : Lasso μ A b) (x₂ : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin n)) :
    Problem.UniqSubX self.isProblem ρ x₂ y := by
  sorry

lemma Lasso.uniq_subY (self : Lasso μ A b) (x₁ : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin n)) :
    Problem.UniqSubY self.isProblem ρ x₁ y := by
  sorry

lemma Lasso.feasible (self : Lasso μ A b) : (self.isProblem).HasOptimum := by
  sorry

lemma Lasso.slater (self : Lasso μ A b) : (self.isProblem).Slater := by
  sorry

class Lasso_ADMM (pro : Lasso μ A b) (x₀ y₀ u₀ : EuclideanSpace ℝ (Fin n)) where
  ρ : ℝ
  τ : ℝ
  x : ℕ → EuclideanSpace ℝ (Fin n)
  y : ℕ → EuclideanSpace ℝ (Fin n)
  u : ℕ → EuclideanSpace ℝ (Fin n)
  iter₁ : ∀ k,
    let Atb : EuclideanSpace ℝ (Fin n) := Aᵀ *ᵥ b
    x (k + 1) = (Aᵀ * A + ρ • 1)⁻¹ *ᵥ (Atb + ρ • (y k - u k))
  iter₂ : ∀ k,
    let aux : EuclideanSpace ℝ (Fin n) := x (k + 1) + (1 / ρ) • (y k)
    ∀ i, y (k + 1) i = (Real.sign (aux i) * (max (abs (aux i) - (μ / ρ)) 0))
  iter₃ : ∀ k, u (k + 1) = u k + x (k + 1) - y (k + 1)
  initial₁ : x 0 = x0
  initial₂ : y 0 = y0
  initial₃ : u 0 = u0

variable {pro : Lasso μ A b} {x₀ y₀ u₀ : EuclideanSpace ℝ (Fin n)} {alg : Lasso_ADMM pro x₀ y₀ u₀}

lemma Lasso_ADMM.iter_cor1 (self : Lasso_ADMM pro x₀ y₀ u₀) :
    ∀ (k : ℕ), IsMinOn (fun q ↦ pro.isProblem.augLag self.ρ q (self.y k) (self.u k)) univ (self.x (k + 1)) := by
  sorry

lemma Lasso_ADMM.iter_cor2 (self : Lasso_ADMM pro x₀ y₀ u₀) :
    ∀ (k : ℕ), IsMinOn (fun q ↦ pro.isProblem.augLag self.ρ (self.x (k + 1)) q (self.u k)) univ (self.y (k + 1)) := by
  sorry

lemma Lasso_ADMM.iter_cor3 (self : Lasso_ADMM pro x₀ y₀ u₀) :
    ∀ (k : ℕ), self.u (k + 1) = self.u k + pro.A₁ (self.x (k + 1)) + pro.A₂ (self.y (k + 1)) - pro.bv := by
  sorry

instance Lasso_ADMM.toADMM (alg : Lasso_ADMM pro x₀ y₀ u₀) : ADMM pro.isProblem x₀ y₀ u₀ where
  ρ := alg.ρ
  τ := alg.τ
  x := alg.x
  y := alg.y
  u := alg.u
  hX := fun x₂ y ↦ Lasso.uniq_subX pro x₂ y
  hY := fun x₁ y ↦ Lasso.uniq_subY pro x₁ y
  iter₁ := alg.iter_cor1
  iter₂ := alg.iter_cor2
  iter₃ := alg.iter_cor3
  initial₁ := alg.initial₁
  initial₂ := alg.initial₂
  initial₃ := alg.initial₃

structure Lasso.Lasso_KKT (self : Lasso μ A b)
    (x : EuclideanSpace ℝ (Fin n)) (z : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin n)) : Prop where
  subgrad₁ :
    let Ax : EuclideanSpace ℝ (Fin m) := A *ᵥ x
    y = - Aᵀ *ᵥ (Ax - b)
  subgrad₂ : ∀ i, (z i > 0 ∧ y i = μ) ∨ (z i < 0 ∧ y i = -μ) ∨ (z i = 0 ∧ y i ∈ Icc (-μ) μ)
  eq       :  x - z = 0

lemma IsConvexKKT.isLasso_KKT₁ (h : IsConvexKKT pro.isProblem x y z) :
    let Ax : EuclideanSpace ℝ (Fin m) := A *ᵥ x
    z = -Aᵀ *ᵥ (Ax - b)  := by
  sorry

lemma IsConvexKKT.isLasso_KKT₂ (h : IsConvexKKT pro.isProblem x y z) :
    ∀ i, y i > 0 ∧ z i = μ ∨ y i < 0 ∧ z i = -μ ∨ y i = 0 ∧ z i ∈ Set.Icc (-μ) μ  := by
  sorry

lemma IsConvexKKT.isLasso_KKT₃ (h : IsConvexKKT pro.isProblem x y z) :
    x - y = 0 := by
  sorry

instance isConvexKKT.isLasso_KKT (h : IsConvexKKT pro.isProblem x y z) :
    pro.Lasso_KKT x y z where
  subgrad₁ := h.isLasso_KKT₁
  subgrad₂ := h.isLasso_KKT₂
  eq :=  h.isLasso_KKT₃

theorem lasso_ADMM_converges (alg : Lasso_ADMM pro x₀ y₀ u₀) (hρ : alg.ρ > 0) (hτ : 0 < alg.τ ∧ alg.τ < (1 + √5) / 2) :
    ∃ (x y z : EuclideanSpace ℝ (Fin n)), (pro.Lasso_KKT x y z) ∧
    Tendsto (fun k ↦ (alg.x k, alg.y k, alg.u k)) atTop (nhds (x, y, z)) := by
  obtain ⟨x, y, z, hx⟩ := admm_converges (alg := alg.toADMM) hρ hτ pro.injA₁ pro.injA₂ pro.f₁_lsc pro.f₂_lsc pro.f₁_conv pro.f₂_conv
    pro.feasible pro.slater
  use x; use y; use z
  constructor
  · apply isConvexKKT.isLasso_KKT
    exact hx.1
  exact hx.2

end Lasso
