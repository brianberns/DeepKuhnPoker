﻿namespace DeepKuhnPoker

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

    /// Generates training data for the given player.
    let private generateSamples iter updatingPlayer models =
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

    /// Adds the given samples to the given reservoir and then
    /// uses the reservoir to train the given advantage model.
    let private trainAdvantageModel resv newSamples model =
        let resv = Reservoir.addMany newSamples resv
        let losses =
            AdvantageModel.train
                settings.NumAdvantageModelTrainSteps
                resv.Items
                model
        resv, losses

    /// Trains a single iteration.
    let private trainIteration iter models (resvMap : Map<_, _>) =

            // train each player's model
        let stratSampleSeqs, resvMap =
            (resvMap, seq { 0 .. KuhnPoker.numPlayers - 1 })
                ||> Seq.mapFold (fun resvMap updatingPlayer ->

                        // generate training data for this player
                    let advSamples, stratSamples =
                        generateSamples iter updatingPlayer models

                        // train this player's model
                    let resv, losses =
                        trainAdvantageModel
                            resvMap[updatingPlayer]
                            advSamples
                            models[updatingPlayer]
                    let resvMap =
                        Map.add updatingPlayer resv resvMap

                        // log inputs and losses
                    settings.Writer.add_scalar(
                        $"advantage reservoir/player{updatingPlayer}",
                        float32 resv.Items.Count,
                        iter)
                    for step = 0 to losses.Length - 1 do
                        settings.Writer.add_scalar(
                            $"advantage loss/iter%04d{iter}/player{updatingPlayer}",
                            losses[step], step)

                    stratSamples, resvMap)

            // log betting behavior
        for infoSetKey in [ "J"; "K"; "Jc"; "Qb"; "Qcb" ] do
            let player = (infoSetKey.Length - 1) % 2
            let betProb = (getStrategy infoSetKey models[player])[0]
            settings.Writer.add_scalar(
                $"advantage bet probability/{infoSetKey}",
                betProb,
                iter)

        resvMap, Seq.concat stratSampleSeqs

    /// Trains a strategy model using the given samples.
    let private trainStrategyModel (resv : Reservoir<StrategySample>) =
        let model =
            StrategyModel.create
                settings.HiddenSize
                settings.LearningRate
        let losses =
            StrategyModel.train
                settings.NumStrategyModelTrainSteps
                resv.Items
                model
        for step = 0 to losses.Length - 1 do
            settings.Writer.add_scalar(
                "strategy loss", losses[step], step)
        model

    /// Trains for the given number of iterations.
    let train () =

            // create advantage models
        let advModels =
            [|
                for _ = 1 to KuhnPoker.numPlayers do
                    AdvantageModel.create
                        settings.HiddenSize
                        settings.LearningRate
            |]
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
            let iterNums = seq { 0 .. settings.NumIterations - 1 }
            ((advResvMap, stratResv), iterNums)
                ||> Seq.fold (fun (advResvMap, stratResv) iter ->
                    let advResvMap, stratSamples =
                        trainIteration iter advModels advResvMap
                    let stratResv =
                        Reservoir.addMany stratSamples stratResv
                    settings.Writer.add_scalar(
                        $"strategy reservoir",
                        float32 stratResv.Items.Count,
                        iter)
                    advResvMap, stratResv)

            // train the final strategy model
        trainStrategyModel stratResv
