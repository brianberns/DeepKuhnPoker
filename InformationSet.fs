namespace DeepKuhnPoker

open MathNet.Numerics.LinearAlgebra

module InformationSet =

    /// Normalizes a strategy such that its elements sum to
    /// 1.0 (to represent action probabilities).
    let private normalize strategy =

            // assume no negative values during normalization
        assert(Vector.forall (fun x -> x >= 0.0f) strategy)

        let sum = Vector.sum strategy
        if sum > 0.0f then strategy / sum
        else
            let idx = Vector.maxIndex strategy
            DenseVector.init strategy.Count (fun i ->
                if i = idx then 1.0f
                else 0.0f)

    /// Computes regret-matching strategy from given regrets.
    let getStrategy regrets =
        regrets
            |> Vector.map (max 0.0f)   // clamp negative regrets
            |> normalize
