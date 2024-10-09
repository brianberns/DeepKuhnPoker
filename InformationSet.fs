namespace DeepKuhnPoker

open MathNet.Numerics.LinearAlgebra

module InformationSet =

    /// Computes strategy from the given regrets. A strategy
    /// is normalized so that its elements sum to 1.0 (to
    /// represent action probabilities).
    let getStrategy regrets =

            // find highest-value action
        let idx = Vector.maxIndex regrets

            // scale if possible, or choose highest-value action
        if regrets[idx] > 0.0f then
            let clamped = Vector.map (max 0.0f) regrets   // clamp negative regrets
            clamped / Vector.sum clamped
        else
            DenseVector.init regrets.Count (fun i ->
                if i = idx then 1.0f
                else 0.0f)
