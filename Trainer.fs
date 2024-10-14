namespace DeepKuhnPoker

open System
open MathNet.Numerics.LinearAlgebra
open TorchSharp

[<AutoOpen>]
module Settings =

    /// Hyperparameters.
    let settings =
        {|
            /// Random number generator.
            Random = Random(0)

            /// Maximum number of samples stored in a reservoir.
            ReservoirCapacity = 10_000_000

            /// Size of a neural network hidden layer.
            HiddenSize = 16

            /// Optimizer learning rate.
            LearningRate = 1e-3

            /// Number of steps to use when training models.
            NumModelTrainSteps = 20

            /// Number of samples to use from the reservoir at each
            /// step of training.
            NumSamples = 128

            /// Number of deals to traverse during each iteration.
            NumTraversals = 40

            /// Number of iterations to perform.
            NumIterations = 40
        |}

module Trainer =

    /// Computes strategy for the given info set using the
    /// given advantage model.
    let getStrategy infoSetKey (advModel : AdvantageModel) =
        use _ = torch.no_grad()   // use model.eval() instead?
        (AdvantageModel.getAdvantage infoSetKey advModel)
            .data<float32>()
            |> DenseVector.ofSeq
            |> InformationSet.getStrategy

    /// Negates opponent's utilties (assuming a zero-zum game).
    let private getActiveUtilities utilities =
        utilities
            |> Seq.map (~-)
            |> DenseVector.ofSeq

    /// Evaluates the utility of the given deal.
    let private traverse
        iter deal updatingPlayer (advModels : AdvantageModel[]) =

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
                getStrategy infoSetKey advModels[activePlayer]

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
    let private trainAdvantageModel
        resv newSamples (model : AdvantageModel) =

            // update reservoir
        let resv = Reservoir.addMany newSamples resv

            // train model
        use optim =
            torch.optim.Adam(
                model.Network.parameters(),
                lr = settings.LearningRate)
        use loss = torch.nn.MSELoss()
        for _ = 1 to settings.NumModelTrainSteps do
            let samples =
                Reservoir.sample settings.NumSamples resv
            AdvantageModel.train
                samples optim loss model

        resv

    /// Trains a single iteration.
    let private trainIteration
        iter advModels (advResvMap : Map<_, _>) =

            // train each player's model once
        let stratSampleSeqs, advResvMap =
            (advResvMap, seq { 0 .. KuhnPoker.numPlayers - 1})
                ||> Seq.mapFold (fun advResvMap updatingPlayer ->

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
                                    iter deal updatingPlayer advModels
                        |]

                        // train model
                    let advResMap =
                        let advResv =
                            trainAdvantageModel
                                advResvMap[updatingPlayer]
                                advSamples
                                advModels[updatingPlayer]
                        Map.add updatingPlayer advResv advResvMap

                    stratSamples, advResMap)

        advResvMap, Seq.concat stratSampleSeqs

    /// Trains for the given number of iterations.
    let train () =

        torch.manual_seed(0) |> ignore

            // create advantage model
        let advModels =
            Array.init KuhnPoker.numPlayers (fun _ ->
                AdvantageModel.create settings.HiddenSize)
        let advResvMap =
            Seq.init KuhnPoker.numPlayers (fun player ->
                let resv =
                    Reservoir.create
                        settings.Random
                        settings.ReservoirCapacity
                player, resv)
                |> Map

            // create strategy model
        let stratModel =
            StrategyModel.create settings.HiddenSize
        let stratResv =
            Reservoir.create
                settings.Random
                settings.ReservoirCapacity

            // run the iterations
        let stratResv, _ =
            ((advResvMap, stratResv), seq { 0 .. settings.NumIterations - 1 })
                ||> Seq.fold (fun (advResvMap, stratResv) iter ->
                    let advResvMap, stratSamples =
                        trainIteration iter advModels advResvMap
                    let stratResv =
                        Reservoir.addMany stratSamples stratResv
                    advResvMap, stratResv)

        stratModel
