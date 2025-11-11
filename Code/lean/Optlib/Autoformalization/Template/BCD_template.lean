import Mathlib.Analysis.InnerProductSpace.ProdL2
import Mathlib.Analysis.Normed.Lp.ProdLp
import Mathlib.Topology.MetricSpace.Sequences
import Mathlib.Analysis.Convex.Deriv
import Optlib.Function.KL
import Optlib.Function.Proximal
import Optlib.Differential.Subdifferential
import Mathlib.Topology.EMetricSpace.Lipschitz

open Set Real Matrix Finset Filter Bornology BigOperators Topology

noncomputable section BCD

variable {E F : Type*}
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E] [ProperSpace E]
variable [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F] [ProperSpace F]
variable {f : E → ℝ} {g : F → ℝ}
variable {H : (WithLp 2 (E × F)) → ℝ} {x0 : E} {y0 : F} {l : NNReal}

/- The gradient of the first component -/
def grad_fst (H : WithLp 2 (E × F) → ℝ) (y : F) : E → E := gradient (fun t ↦ H (t, y))

/- The gradient function of the second component -/
def grad_fun_fst (H : WithLp 2 (E × F) → ℝ) := fun (x, y) ↦ (grad_fst H y x)

/- The gradient of the second component -/
def grad_snd (H : WithLp 2 (E × F) → ℝ) (x : E) : F → F := gradient (fun t ↦ H (x, t))

/- The gradient function of the second component -/
def grad_fun_snd (H : WithLp 2 (E × F) → ℝ) := fun (x, y) ↦ (grad_snd H x y)

/- The gradient of the prod domain -/
def grad_comp (H : WithLp 2 (E × F) → ℝ) (z : WithLp 2 (E × F)) : WithLp 2 (E × F) :=
    (WithLp.equiv 2 (E × F)).symm (grad_fst H z.2 z.1, grad_snd H z.1 z.2)

class ProblemData (f : E → ℝ) (g : F → ℝ) (H : (WithLp 2 (E × F)) → ℝ) (l : NNReal) : Prop where
  lbdf : BddBelow (f '' univ)
  lbdg : BddBelow (g '' univ)
  hf : LowerSemicontinuous f
  hg : LowerSemicontinuous g
  conH : ContDiff ℝ 1 H
  lpos : l > (0 : ℝ)
  lip : LipschitzWith l (gradient H)

/--
  The definition of block coordinate descent
-/
structure BCD (pro : ProblemData f g H l) (x0 : E) (y0 : F) where
  x : ℕ → E
  y : ℕ → F
  x0 : x 0 = x0
  y0 : y 0 = y0
  c : ℕ → ℝ
  d : ℕ → ℝ
  s₁ : ∀ k, prox_prop (c k • f) (x k - c k • (grad_fst H (y k) (x k))) (x (k + 1))
  s₂ : ∀ k, prox_prop (d k • g) (y k - d k • (grad_snd H (x (k + 1)) (y k))) (y (k + 1))

open BCD

variable {pro : ProblemData f g H l} {x0 : E} {y0 : F}
/- the notation z in BCD -/
def BCD.z {self : BCD pro x0 y0} : ℕ → WithLp 2 (E × F) :=
  fun n ↦ (WithLp.equiv 2 (E × F)).symm (self.x n, self.y n)

/- the notation ψ in BCD -/
def BCD.ψ {_ : BCD pro x0 y0} := fun z : WithLp 2 (E × F) ↦ f z.1 + g z.2 + H z

variable {alg : BCD pro x0 y0}

theorem Sufficient_Descent1 (γ : ℝ) (hγ : γ > 1) (ck : ∀ k, alg.c k = 1 / (γ * l)) (dk : ∀ k, alg.d k = 1 / (γ * l)) :
    ∃ ρ₁ > 0, ρ₁ = (γ - 1) * l ∧ ∀ k, ρ₁ / 2 * ‖alg.z (k + 1) - alg.z k‖ ^ 2 ≤ alg.ψ (alg.z k) - alg.ψ (alg.z (k + 1)) := by sorry

theorem Sufficient_Descent2 (γ : ℝ) (hγ : γ > 1) (ck : ∀ k, alg.c k = 1 / (γ * l)) (dk : ∀ k, alg.d k = 1 / (γ * l)) :
    ∀ (k : ℕ), alg.ψ (alg.z (k + 1)) ≤ alg.ψ (alg.z k) := by sorry

theorem Sufficient_Descent3 (γ : ℝ) (hγ : γ > 1) (ck : ∀ k, alg.c k = 1 / (γ * l)) (dk : ∀ k, alg.d k = 1 / (γ * l))
    (lbdψ : BddBelow (alg.ψ '' univ)) :
    ∃ (A : ℝ), Tendsto (fun (n : ℕ) ↦ ∑ k ∈ Finset.range n, ‖alg.z (k + 1) - alg.z k‖ ^ 2) atTop (𝓝 A) := by sorry

theorem Sufficient_Descent4 (γ : ℝ) (hγ : γ > 1) (ck : ∀ k, alg.c k = 1 / (γ * l)) (dk : ∀ k, alg.d k = 1 / (γ * l))
    (lbdψ : BddBelow (alg.ψ '' univ)) : Tendsto (fun k ↦ ‖alg.z (k + 1) - alg.z k‖) atTop (𝓝 0) := by sorry

theorem Convergence_to_critpt (γ : ℝ) (hγ : γ > 1) (ck : ∀ k, alg.c k = 1 / (γ * l)) (dk : ∀ k, alg.d k = 1 / (γ * l))
    (bd : Bornology.IsBounded (alg.z '' univ)) (hψ : KL_function alg.ψ) (lbdψ : BddBelow (alg.ψ '' univ)) :
    ∃ z_ : (WithLp 2 (E × F)), z_ ∈ (critial_point alg.ψ) ∧ Tendsto alg.z atTop (𝓝 z_) := by sorry

end BCD
