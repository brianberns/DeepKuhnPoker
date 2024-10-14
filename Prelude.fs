namespace DeepKuhnPoker

module Choice =

    /// Unzips a sequence of choices.
    let unzip choices =
        let opts =
            choices
                |> Seq.map (function
                    | Choice1Of2 ch -> Some ch, None
                    | Choice2Of2 ch -> None, Some ch)
                |> Seq.cache
        Seq.choose fst opts,
        Seq.choose snd opts

module Vector =

    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Distributions

    /// Samples a strategy.
    let inline sample rng (strategy : Vector<_>) =
        let strategy' =
            strategy
                |> Seq.map float
                |> Seq.toArray
        Categorical.Sample(rng, strategy')
