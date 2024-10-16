namespace DeepKuhnPoker

open MathNet.Numerics.LinearAlgebra
open TorchSharp

module Trainer =

    /// Computes strategy for the given info set using the
    /// given advantage model.
    let getStrategy infoSetKey model =
        use _ = torch.no_grad()   // use model.eval() instead?
        (AdvantageModel.getAdvantage infoSetKey model)
            .data<float32>()
            |> DenseVector.ofSeq
            |> InformationSet.getStrategy

    /// Negates opponent's utilties (assuming a zero-zum game).
    let private getActiveUtilities utilities =
        utilities
            |> Seq.map (~-)
            |> DenseVector.ofSeq

    /// Evaluates the utility of the given deal.
    let private traverse iter deal updatingPlayer (models : _[]) =

        /// Appends an item to the end of an array.
        let append items item =
            [| yield! items; yield item |]

        /// Top-level loop.
        let rec loop history =
            match KuhnPoker.getPayoff deal history with
                | Some payoff ->
                    float32 payoff, Array.empty   // game is over
                | None ->
                    loopNonTerminal history

        /// Recurses for non-terminal game state.
        and loopNonTerminal history =

                // get info set for current state from this player's point of view
            let activePlayer = KuhnPoker.getActivePlayer history
            let infoSetKey = deal[activePlayer] + history

                // get active player's current strategy for this info set
            let strategy =
                getStrategy infoSetKey models[activePlayer]

                // get utility of this info set
            if activePlayer = updatingPlayer then

                    // get utility of each action
                let actionUtilities, samples =
                    let utilities, sampleArrays =
                        KuhnPoker.actions
                            |> Array.map (fun action ->
                                loop (history + action))
                            |> Array.unzip
                    getActiveUtilities utilities,
                    Array.concat sampleArrays

                    // utility of this info set is action utilities weighted by action probabilities
                let utility = actionUtilities * strategy
                let sample =
                    AdvantageSample.create
                        infoSetKey
                        (actionUtilities - utility)
                        iter |> Choice1Of2
                utility, append samples sample

            else
                    // sample a single action according to the strategy
                let utility, samples =
                    let action =
                        strategy
                            |> Vector.sample settings.Random
                            |> Array.get KuhnPoker.actions
                    loop (history + action)
                let sample =
                    StrategySample.create
                        infoSetKey
                        strategy
                        iter |> Choice2Of2
                -utility, append samples sample

        loop "" |> snd

    /// Adds the given samples to the given reservoir and
    /// then uses the reservoir to train the given model.
    let private trainAdvantageModel resv newSamples model =

            // update reservoir
        let resv = Reservoir.addMany newSamples resv

            // train model
        let losses =
            Array.init settings.NumAdvantageModelTrainSteps
                (fun _ ->
                    AdvantageModel.train resv.Items model)

        resv, losses

    /// Trains a single iteration.
    let private trainIteration iter models (resvMap : Map<_, _>) =

            // train each player's model once
        let stratSampleSeqs, resvMap =
            (resvMap, seq { 0 .. KuhnPoker.numPlayers - 1})
                ||> Seq.mapFold (fun resvMap updatingPlayer ->

                        // generate training data for this player
                    let advSamples, stratSamples =
                        Choice.unzip [|
                            for _ = 1 to settings.NumTraversals do
                                let deal =
                                    let iDeal =
                                        settings.Random.Next(
                                            KuhnPoker.allDeals.Length)
                                    KuhnPoker.allDeals[iDeal]
                                yield! traverse
                                    iter deal updatingPlayer models
                        |]

                        // train model
                    let resv, losses =
                        trainAdvantageModel
                            resvMap[updatingPlayer]
                            advSamples
                            models[updatingPlayer]
                    let resvMap =
                        Map.add updatingPlayer resv resvMap

                        // log losses
                    for step = 0 to losses.Length - 1 do
                        settings.Writer.add_scalar(
                            $"iter%04d{iter}/player{updatingPlayer}",
                            losses[step], step)

                    stratSamples, resvMap)

        resvMap, Seq.concat stratSampleSeqs

    let private trainStrategyModel (resv : Reservoir<StrategySample>) =

        let model =
            StrategyModel.create
                settings.HiddenSize
                settings.LearningRate

            // train model
        for _ = 1 to settings.NumStrategyModelTrainSteps do
            StrategyModel.train resv.Items model

        model

    /// Trains for the given number of iterations.
    let train () =

        torch.manual_seed(0) |> ignore

            // create advantage models
        let advModels =
            Array.init KuhnPoker.numPlayers (fun _ ->
                AdvantageModel.create
                    settings.HiddenSize
                    settings.LearningRate)
        let advResvMap =
            Map [
                for player = 0 to KuhnPoker.numPlayers - 1 do
                    let resv =
                        Reservoir.create
                            settings.Random
                            settings.NumAdvantageSamples
                    player, resv
            ]

            // run the iterations
        let _, stratResv =
            let stratResv =
                Reservoir.create
                    settings.Random
                    settings.NumStrategySamples
            ((advResvMap, stratResv), seq { 0 .. settings.NumIterations - 1 })
                ||> Seq.fold (fun (advResvMap, stratResv) iter ->
                    let advResvMap, stratSamples =
                        trainIteration iter advModels advResvMap
                    let stratResv =
                        Reservoir.addMany stratSamples stratResv
                    advResvMap, stratResv)

        trainStrategyModel stratResv
