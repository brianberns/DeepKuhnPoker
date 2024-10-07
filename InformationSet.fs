namespace DeepKuhnPoker

open MathNet.Numerics.LinearAlgebra

module InformationSet =

    /// Uniform strategy: All actions have equal probability.
    let private uniformStrategy =
        DenseVector.create
            KuhnPoker.actions.Length
            (1.0f / float32 KuhnPoker.actions.Length)

    /// Normalizes a strategy such that its elements sum to
    /// 1.0 (to represent action probabilities).
    let private normalize strategy =

            // assume no negative values during normalization
        assert(Vector.forall (fun x -> x >= 0.0f) strategy)

        let sum = Vector.sum strategy
        if sum > 0.0f then strategy / sum
        else uniformStrategy

    /// Computes regret-matching strategy from given regrets.
    let getStrategy regrets =
        regrets
            |> Vector.map (max 0.0f)   // clamp negative regrets
            |> normalize
