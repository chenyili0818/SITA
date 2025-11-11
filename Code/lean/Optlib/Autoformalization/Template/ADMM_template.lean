import Optlib.Autoformalization.Lemmas
import Mathlib.Analysis.Convex.Intrinsic
import Optlib.Convex.Subgradient

open Real InnerProductSpace Filter Function Set ContinuousLinearMap

local notation "⟪" a₁ ", " a₂ "⟫" => @inner ℝ _ _ a₁ a₂

noncomputable section ADMM

variable {E₁ E₂ F : Type*}
[NormedAddCommGroup E₁] [InnerProductSpace ℝ E₁] [CompleteSpace E₁]
[NormedAddCommGroup E₂] [InnerProductSpace ℝ E₂] [CompleteSpace E₂]
[NormedAddCommGroup F ] [InnerProductSpace ℝ F ] [CompleteSpace F ]

/-- Definitions of ADMM -/
structure Problem (f₁ : E₁ → ℝ) (f₂ : E₂ → ℝ) (A₁ : E₁ →L[ℝ] F) (A₂ : E₂ →L[ℝ] F) (b : F) where

variable {f₁ : E₁ → ℝ} {f₂ : E₂ → ℝ} {A₁ : E₁ →L[ℝ] F} {A₂ : E₂ →L[ℝ] F} {b : F}

noncomputable def Problem.augLag (_ : Problem f₁ f₂ A₁ A₂ b) : ℝ → E₁ → E₂ → F → ℝ :=
  fun ρ x₁ x₂ y ↦ f₁ x₁ + f₂ x₂ + ⟪y, A₁ x₁ + A₂ x₂ - b⟫ + (ρ / 2) * ‖A₁ x₁ + A₂ x₂ - b‖ ^ 2

def HasUniqueMin {V : Type*} (φ : V → ℝ) : Prop :=
  ∃! x : V, ∀ y : V, φ x ≤ φ y

def Problem.UniqSubX (P : Problem f₁ f₂ A₁ A₂ b) (ρ : ℝ) (x₂ : E₂) (y : F) : Prop :=
  HasUniqueMin (fun u : E₁ => P.augLag ρ u x₂ y)

def Problem.UniqSubY (P : Problem f₁ f₂ A₁ A₂ b) (ρ : ℝ) (x₁ : E₁) (y : F) : Prop :=
  HasUniqueMin (fun u : E₂ => P.augLag ρ x₁ u y)

def Problem.HasOptimum (_ : Problem f₁ f₂ A₁ A₂ b) : Prop :=
  ∃ x z, A₁ x + A₂ z = b  ∧ ∀ x' z', A₁ x' + A₂ z' = b → f₁ x + f₂ z ≤ f₁ x' + f₂ z'

/-- ADMM Method -/
class ADMM (P : Problem f₁ f₂ A₁ A₂ b) (x0 : E₁) (y0 : E₂) (u0 : F) where
  ρ : ℝ
  τ : ℝ
  x : ℕ → E₁
  y : ℕ → E₂
  u : ℕ → F
  hX : ∀ x₂ y, Problem.UniqSubX P ρ x₂ y
  hY : ∀ x₁ y, Problem.UniqSubY P ρ x₁ y
  iter₁ : ∀ k, IsMinOn (fun q : E₁ => P.augLag ρ q (y k) (u k)) univ (x (k + 1))
  iter₂ : ∀ k, IsMinOn (fun q : E₂ => P.augLag ρ (x (k + 1)) q (u k)) univ (y (k + 1))
  iter₃ : ∀ k, u (k + 1) = u k + A₁ (x (k + 1)) + A₂ (y (k + 1)) - b
  initial₁ : x 0 = x0
  initial₂ : y 0 = y0
  initial₃ : u 0 = u0

variable {x0 : E₁} {y0 : E₂} {u0 : F} {P : Problem f₁ f₂ A₁ A₂ b} {alg : ADMM P x0 y0 u0}

def Slater_condition {α V : Type* }[TopologicalSpace α] [AddCommGroup V][Module ℝ V][AddTorsor V α]
    (fr: Set α) : Prop := ∃ x , x ∈ intrinsicInterior ℝ fr

/-- Slater Condition -/
def Problem.Slater (_ : Problem f₁ f₂ A₁ A₂ b) : Prop := Slater_condition {(x₁, x₂) | A₁ x₁ + A₂ x₂ = b}

class IsConvexKKT (P : Problem f₁ f₂ A₁ A₂ b) (x₁ : E₁) (x₂ : E₂) (y : F) : Prop where
   subgrad₁ : - adjoint A₁ y ∈ SubderivAt f₁ x₁
   subgrad₂ : - adjoint A₂ y ∈ SubderivAt f₂ x₂
   eq       :  (A₁ x₁) + (A₂ x₂) = b

theorem admm_converges {P : Problem f₁ f₂ A₁ A₂ b} (alg : ADMM P x0 y0 u0)
    (hρ : alg.ρ > 0)
    (hτ : 0 < alg.τ ∧ alg.τ < ( 1 + √ 5 ) / 2)
    (fullrank₁ : Injective A₁) (fullrank₂ : Injective A₂)
    (lscf₁ : LowerSemicontinuous f₁) (lscf₂ : LowerSemicontinuous f₂)
    (cf₁ : ConvexOn ℝ univ f₁) (cf₂ : ConvexOn ℝ univ f₂)
    (hP : P.HasOptimum) (hs : P.Slater) :
    ∃ (z : E₁) (v : E₂) (w : F),
      IsConvexKKT P z v w ∧ Tendsto (fun n ↦ (alg.x n, alg.y n, alg.u n)) atTop (nhds (z, v, w)) := by
  sorry

end ADMM
