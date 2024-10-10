﻿namespace DeepKuhnPoker

open System

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

open TorchSharp

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

type private AdvantageStateMap =
    Map<int,
        {|
            Model : AdvantageModel
            Optimizer : torch.optim.Optimizer
            Reservoir : Reservoir<AdvantageSample>
        |}>

module private AdvantageStateMap =

    let create
        hiddenSize learningRate rng reservoirCapacity =
        Seq.init KuhnPoker.numPlayers (fun i ->
            let model = AdvantageModel.create hiddenSize
            let optim : torch.optim.Optimizer =
                torch.optim.Adam(
                    model.Network.parameters(),
                    lr = learningRate)
            let resv = Reservoir.create rng reservoirCapacity
            let state =
                {|
                    Model = model
                    Optimizer = optim
                    Reservoir = resv
                |}
            i, state)
            |> Map

    let update
        player model optimizer reservoir stateMap =
        let state =
            {|
                Model = model
                Optimizer = optimizer
                Reservoir = reservoir
            |}
        Map.add player state stateMap

module KuhnCfrTrainer =

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

    /// Random number generator.
    let private rng = Random(0)

    /// Evaluates the utility of the given deal.
    let private traverse iter deal updatingPlayer model =

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

                // get player's current strategy for this info set
            let strategy = getStrategy infoSetKey model

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
                let action =
                    let strategy' =
                        strategy
                            |> Seq.map float   // ugly
                            |> Seq.toArray
                    Categorical.Sample(rng, strategy')
                        |> Array.get KuhnPoker.actions
                let utility, samples =
                    loop (history + action)
                let sample =
                    StrategySample.create
                        infoSetKey
                        strategy
                        iter |> Choice2Of2
                -utility, append samples sample

        loop "" |> snd

    let private hiddenSize = 16
    let private learningRate = 0.01
    let private reservoirCapacity = 1000
    let private numModelTrainSteps = 20
    let private numSamples = 10

    let private updateAdvantageModel
        reservoir newSamples optim loss model =

            // update reservoir
        let resv =
            (reservoir, newSamples)
                ||> Seq.fold (fun resv (advSample : AdvantageSample) ->
                    Reservoir.add advSample resv)

            // train model
        let model =
            (model, Seq.init numModelTrainSteps id)
                ||> Seq.fold (fun model _ ->
                    resv
                        |> Reservoir.trySample numSamples
                        |> Option.map (fun samples ->
                            AdvantageModel.train
                                samples optim loss model)
                        |> Option.defaultValue model)

        resv, model

    /// Trains for the given number of iterations.
    let train numIterations numTraversals =

            // gather chunks of deals
        let chunkPairs =
            let numDeals = numIterations * numTraversals
            Seq.init numDeals (fun i ->
                KuhnPoker.allDeals[i % KuhnPoker.allDeals.Length])
                |> Seq.chunkBySize numTraversals
                |> Seq.indexed

        let advStateMap =
            AdvantageStateMap.create
                hiddenSize learningRate rng reservoirCapacity
        let advLoss = torch.nn.MSELoss()

        let advStateMap =
            (advStateMap, chunkPairs)
                ||> Seq.fold (fun advState (iter, chunk) ->

                        // traverse this chunk of deals
                    let updatingPlayer = iter % KuhnPoker.numPlayers
                    let advModel, advOptim, advResv =
                        let advState = advStateMap[updatingPlayer]
                        advState.Model, advState.Optimizer, advState.Reservoir
                    let newSamples =
                        chunk
                            |> Array.collect (fun deal ->
                                traverse
                                    iter deal updatingPlayer advModel)

                        // update advantages
                    let advSamples, stratSamples = Choice.unzip newSamples
                    let advResv, advModel =
                        updateAdvantageModel
                            advResv advSamples advOptim advLoss advModel
                    AdvantageStateMap.update
                        updatingPlayer advModel advOptim advResv advState)

        advStateMap.Values
            |> Seq.map (fun state -> state.Model)
            |> Seq.toArray

module Program =

    let playerInfoSetKeys =
        [|
            [| "J"; "Q"; "K"; "Jcb"; "Qcb"; "Kcb" |]
            [| "Jb"; "Jc"; "Qb"; "Qc"; "Kb"; "Kc" |]
        |]

    let run () =

        torch.manual_seed(0) |> ignore

            // train
        let numIterations = 50
        let numTraversals = KuhnPoker.allDeals.Length
        printfn $"Running Kuhn Poker Deep CFR for {numIterations} iterations"
        let advModels = KuhnCfrTrainer.train numIterations numTraversals

        for player = 0 to KuhnPoker.numPlayers - 1 do
            let advModel = advModels[player]
            let infoSetKeys = playerInfoSetKeys[player]
            printfn $"\nPlayer {player}"
            for infoSetKey in infoSetKeys do
                let strategy =
                    KuhnCfrTrainer.getStrategy infoSetKey advModel
                printfn "   %-3s: %s = %0.3f, %s = %0.3f"
                    infoSetKey
                    KuhnPoker.actions[0]
                    strategy[0]
                    KuhnPoker.actions[1]
                    strategy[1]

    let timer = Diagnostics.Stopwatch.StartNew()
    run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    