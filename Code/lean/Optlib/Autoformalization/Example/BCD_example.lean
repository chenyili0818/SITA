import Optlib.Autoformalization.Template.BCD_template

open Set Real Matrix Finset Filter Bornology BigOperators Topology

noncomputable section JointSparseCoding

variable {p n m : ℕ}
variable {A : Matrix (Fin p) (Fin n) ℝ} {B : Matrix (Fin p) (Fin m) ℝ} {b : Fin p → ℝ} {lambda1 lambda2 : ℝ}

local notation "‖" x "‖₂" => @Norm.norm _ (PiLp.instNorm 2 fun _ ↦ ℝ) x
local notation "‖" x "‖₁" => @Norm.norm _ (PiLp.instNorm 1 fun _ ↦ ℝ) x
local notation "|‖" A "|‖" => ‖(Matrix.toEuclideanLin ≪≫ₗ LinearMap.toContinuousLinearMap) A‖₊

class JointSparseCoding_problem (A : Matrix (Fin p) (Fin n) ℝ)
    (B : Matrix (Fin p) (Fin m) ℝ) (b : Fin p → ℝ) (lambda1 lambda2 : ℝ) where
  hA : A ≠ 0
  hB : B ≠ 0
  hlambda1 : lambda1 > 0
  hlambda2 : lambda2 > 0

def JointSparseCoding_problem.f (_ : JointSparseCoding_problem A B b lambda1 lambda2) :
    EuclideanSpace ℝ (Fin n) → ℝ := fun x ↦ lambda1 * ‖x‖₁

def JointSparseCoding_problem.g (_ : JointSparseCoding_problem A B b lambda1 lambda2) :
    EuclideanSpace ℝ (Fin m) → ℝ := fun y ↦ lambda2 * ‖y‖₁

def JointSparseCoding_problem.H (_ : JointSparseCoding_problem A B b lambda1 lambda2) :
    WithLp 2 (EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m)) → ℝ :=
  fun z ↦ 1/2 * ‖A *ᵥ z.1 + B *ᵥ z.2 - b‖₂ ^ 2

def JointSparseCoding_problem.ψ (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    WithLp 2 (EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m)) → ℝ :=
  fun z ↦ self.f z.1 + self.g z.2 + self.H z

def JointSparseCoding_problem.l (_ : JointSparseCoding_problem A B b lambda1 lambda2) : NNReal :=
  |‖Aᵀ * A|‖ + |‖Bᵀ * B|‖

lemma JointSparseCoding_problem.lbdf (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    BddBelow (self.f '' Set.univ) := by
  unfold JointSparseCoding_problem.f
  rw [bddBelow_def]
  use 0
  simp
  intro a
  obtain hlam1 := self.hlambda1
  positivity

lemma JointSparseCoding_problem.lbdg (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    BddBelow (self.g '' Set.univ) := by
  unfold JointSparseCoding_problem.g
  rw [bddBelow_def]
  use 0
  simp
  intro a
  obtain hlam2 := self.hlambda2
  positivity

lemma JointSparseCoding_problem.hf (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    LowerSemicontinuous self.f := by
  unfold JointSparseCoding_problem.f
  apply Continuous.lowerSemicontinuous
  have : Continuous (fun (x : EuclideanSpace ℝ (Fin n)) ↦ ‖x‖₁) := by
    exact Continuous.norm continuous_id'
  obtain h1 := Continuous.const_smul this lambda1
  simpa

lemma JointSparseCoding_problem.hg (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    LowerSemicontinuous self.g := by
  unfold JointSparseCoding_problem.g
  apply Continuous.lowerSemicontinuous
  have : Continuous (fun (y : EuclideanSpace ℝ (Fin m)) ↦ ‖y‖₁) := by
    exact Continuous.norm continuous_id'
  obtain h2 := Continuous.const_smul this lambda2
  simpa

lemma JointSparseCoding_problem.conH (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    ContDiff ℝ 1 self.H := by
  sorry

lemma JointSparseCoding_problem.lpos (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    self.l > 0 := by
  sorry

lemma JointSparseCoding_problem.lip (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    LipschitzWith self.l (gradient self.H) := by
  sorry

instance JointSparseCoding_problem.ProblemData (self : JointSparseCoding_problem A B b lambda1 lambda2) :
    ProblemData self.f self.g self.H self.l where
  lbdf := self.lbdf
  lbdg := self.lbdg
  hf := self.hf
  hg := self.hg
  conH := self.conH
  lpos := self.lpos
  lip := self.lip

open JointSparseCoding_problem

class BCD_JointSparse (pro : JointSparseCoding_problem A B b lambda1 lambda2)
    (x0 : EuclideanSpace ℝ (Fin n)) (y0 : EuclideanSpace ℝ (Fin m)) where
  c : ℕ → ℝ
  d : ℕ → ℝ
  x : ℕ → EuclideanSpace ℝ (Fin n)
  y : ℕ → EuclideanSpace ℝ (Fin m)
  s₁ : ∀ k,
      let grad_fst : EuclideanSpace ℝ (Fin n) := Aᵀ *ᵥ (A *ᵥ x k + B *ᵥ y k - b)
      let aux : EuclideanSpace ℝ (Fin n) := x k - c k • grad_fst
      ∀ i, x (k + 1) i = (Real.sign (aux i) * (max (abs (aux i) - t * (c k)) 0))
  s₂ : ∀ k,
      let grad_snd : EuclideanSpace ℝ (Fin m) := Bᵀ *ᵥ (A *ᵥ x (k + 1) + B *ᵥ y k - b)
      let aux : EuclideanSpace ℝ (Fin m) := y k - d k • grad_snd
      ∀ i, y (k + 1) i = (Real.sign (aux i) * (max (abs (aux i) - t * (d k)) 0))
  init₁ : x 0 = x0
  init₂ : y 0 = y0

variable {pro : JointSparseCoding_problem A B b lambda1 lambda2}
variable {x0 : EuclideanSpace ℝ (Fin n)} {y0 : EuclideanSpace ℝ (Fin m)}
variable {alg : BCD_JointSparse pro x0 y0}

def BCD_JointSparse.z {self : BCD_JointSparse pro x0 y0} :=
  fun k ↦ (WithLp.equiv 2 (EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m))).symm (self.x k, self.y k)

def BCD_JointSparse.ψ {_ : BCD_JointSparse pro x0 y0} :
    WithLp 2 (EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m)) → ℝ :=
  fun z ↦ pro.f z.1 + pro.g z.2 + pro.H z

lemma BCD_JointSparse.update_cor1 (self : BCD_JointSparse pro x0 y0) :
    ∀ (k : ℕ), prox_prop (self.c k • pro.f)
    (self.x k - self.c k • grad_fst pro.H (self.y k) (self.x k)) (self.x (k + 1)) := by
  sorry

lemma BCD_JointSparse.update_cor2 (self : BCD_JointSparse pro x0 y0) :
    ∀ (k : ℕ), prox_prop (self.d k • pro.g)
    (self.y k - self.d k • grad_snd pro.H (self.x (k + 1)) (self.y k)) (self.y (k + 1)) := by
  sorry

instance BCD_JointSparse.BCD (self : BCD_JointSparse pro x0 y0) :
    BCD pro.ProblemData x0 y0 where
  c := self.c
  d := self.d
  x := self.x
  y := self.y
  s₁ := self.update_cor1
  s₂ := self.update_cor2
  x0 := self.init₁
  y0 := self.init₂

theorem JointSparse_Sufficient_Descent1 (γ : ℝ) (hγ : γ > 1)
    (ck : ∀ k, alg.c k = 1 / (γ * pro.l))
    (dk : ∀ k, alg.d k = 1 / (γ * pro.l)) :
    ∃ ρ₁ > 0, ρ₁ = (γ - 1) * pro.l ∧
    ∀ k, ρ₁ / 2 * ‖alg.z (k + 1) - alg.z k‖ ^ 2 ≤ alg.ψ (alg.z k) - alg.ψ (alg.z (k + 1)) := by
  apply Sufficient_Descent1 (alg := alg.BCD) γ hγ ck dk

theorem JointSparse_Sufficient_Descent2 (γ : ℝ) (hγ : γ > 1)
    (ck : ∀ k, alg.c k = 1 / (γ * pro.l)) (dk : ∀ k, alg.d k = 1 / (γ * pro.l)) :
    ∀ (k : ℕ), alg.ψ (alg.z (k + 1)) ≤ alg.ψ (alg.z k) := by
  apply Sufficient_Descent2 (alg := alg.BCD) γ hγ ck dk

lemma BCD_JointSparse.lbdψ (alg : BCD_JointSparse pro x0 y0) : BddBelow (alg.ψ '' univ) := by
  sorry

theorem JointSparse_Sufficient_Descent3 (γ : ℝ) (hγ : γ > 1)
    (ck : ∀ k, alg.c k = 1 / (γ * pro.l)) (dk : ∀ k, alg.d k = 1 / (γ * pro.l)):
    ∃ (A : ℝ), Tendsto (fun (n : ℕ) ↦ ∑ k ∈ Finset.range n, ‖alg.z (k + 1) - alg.z k‖ ^ 2) atTop (𝓝 A) := by
  apply Sufficient_Descent3 (alg := alg.BCD) γ hγ ck dk alg.lbdψ

theorem JointSparse_Sufficient_Descent4 (γ : ℝ) (hγ : γ > 1)
    (ck : ∀ k, alg.c k = 1 / (γ * pro.l)) (dk : ∀ k, alg.d k = 1 / (γ * pro.l)) :
    Tendsto (fun k ↦ ‖alg.z (k + 1) - alg.z k‖) atTop (𝓝 0) := by
  apply Sufficient_Descent4 (alg := alg.BCD) γ hγ ck dk alg.lbdψ

lemma BCD_JointSparse.bd (alg : BCD_JointSparse pro x0 y0) :
    Bornology.IsBounded (alg.z '' univ) := by
  sorry

lemma BCD_JointSparse.hψ (alg : BCD_JointSparse pro x0 y0) :
    KL_function alg.ψ := by
  sorry

theorem JointSparse_Convergence_to_critpt (γ : ℝ) (hγ : γ > 1)
    (ck : ∀ k, alg.c k = 1 / (γ * pro.l)) (dk : ∀ k, alg.d k = 1 / (γ * pro.l)) :
    ∃ z_ : (WithLp 2 (EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m))),
      z_ ∈ (critial_point alg.ψ) ∧ Tendsto alg.z atTop (𝓝 z_) := by
  apply Convergence_to_critpt (alg := alg.BCD) γ hγ ck dk alg.bd alg.hψ alg.lbdψ

end JointSparseCoding
