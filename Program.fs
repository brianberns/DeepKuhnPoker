namespace DeepKuhnPoker

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

[<AutoOpen>]
module Settings =

    let settings =
        {|
            /// Random number generator.
            Random = Random(0)

            HiddenSize = 16
            LearningRate = 1e-3
            ReservoirCapacity = int 1e7
            NumModelTrainSteps = 20

            /// Number of samples to use from the reservoir at each
            /// step of training.
            NumSamples = 128

            /// Number of deals to traverse during each iteration.
            NumTraversals = 40

            NumIterations = 50
        |}

/// State required to train advantage models.
type private AdvantageState =
    {
        /// Current model.
        Model : AdvantageModel

        /// Current model's optimizer.
        Optimizer : torch.optim.Optimizer

        /// Training data.
        Reservoir : Reservoir<AdvantageSample>
    }

module private AdvantageState =

    /// Initializes advantage state for each player.
    let createMap () =
        Seq.init KuhnPoker.numPlayers (fun player ->
            let state =
                let model = AdvantageModel.create settings.HiddenSize
                {
                    Model = model
                    Optimizer =
                        torch.optim.Adam(
                            model.Network.parameters(),
                            lr = settings.LearningRate)
                    Reservoir =
                        Reservoir.create
                            settings.Random
                            settings.ReservoirCapacity
                }
            player, state)
            |> Map

    /// Updates advantage state for the given player.
    let updateMap
        (player : int) model optimizer reservoir stateMap =
        let state =
            {
                Model = model
                Optimizer = optimizer
                Reservoir = reservoir
            }
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
                    Categorical.Sample(settings.Random, strategy')
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


    /// Adds the given samples to the given reservoir and
    /// then uses the reservoir to train the given model.
    let private updateAdvantageModel
        reservoir newSamples optim loss model =

            // update reservoir
        let resv =
            (reservoir, newSamples)
                ||> Seq.fold (fun resv advSample ->
                    Reservoir.add advSample resv)

            // train model
        let model =
            (model, seq { 1 .. settings.NumModelTrainSteps })
                ||> Seq.fold (fun model _ ->
                    let samples =
                        Reservoir.sample settings.NumSamples resv
                    AdvantageModel.train
                        samples optim loss model)

        resv, model

    /// Trains a single iteration.
    let private trainIteration
        iter advLoss (advStateMap : Map<_, _>) =

            // train each player's model
        (advStateMap, seq { 0 .. KuhnPoker.numPlayers - 1})
            ||> Seq.fold (fun advStateMap updatingPlayer ->

                    // run traversals for this player
                let advModel, advOptim, advResv =
                    let state = advStateMap[updatingPlayer]
                    state.Model, state.Optimizer, state.Reservoir
                let advSamples, stratSamples =
                    Choice.unzip [|
                        for _ = 1 to settings.NumTraversals do
                            let deal =
                                let iDeal =
                                    settings.Random.Next(KuhnPoker.allDeals.Length)
                                KuhnPoker.allDeals[iDeal]
                            yield! traverse
                                iter deal updatingPlayer advModel
                    |]

                    // update model
                let advResv, advModel =
                    updateAdvantageModel
                        advResv advSamples advOptim advLoss advModel
                AdvantageState.updateMap
                    updatingPlayer advModel advOptim advResv advStateMap)

    /// Trains for the given number of iterations.
    let train () =

            // create advantage model
        let advStateMap = AdvantageState.createMap ()
        let advLoss = torch.nn.MSELoss()

            // run the iterations
        let advStateMap =
            (advStateMap, seq { 0 .. settings.NumIterations - 1 })
                ||> Seq.fold (fun advStateMap iter ->
                    trainIteration iter advLoss advStateMap)

        advStateMap.Values
            |> Seq.map (fun advState -> advState.Model)
            |> Seq.toArray

module Program =

    let private playerInfoSetKeys =
        [|
            [| "J"; "Q"; "K"; "Jcb"; "Qcb"; "Kcb" |]
            [| "Jb"; "Jc"; "Qb"; "Qc"; "Kb"; "Kc" |]
        |]

    let run () =

        torch.manual_seed(0) |> ignore

            // train
        printfn "Running Kuhn Poker Deep CFR for %A iterations"
            settings.NumIterations
        let advModels = KuhnCfrTrainer.train ()

        for player = 0 to KuhnPoker.numPlayers - 1 do
            printfn $"\nPlayer {player}"
            let advModel = advModels[player]
            for infoSetKey in playerInfoSetKeys[player] do
                let strategy =
                    KuhnCfrTrainer.getStrategy infoSetKey advModel
                printfn "   %-3s: %s = %.3f, %s = %.3f"
                    infoSetKey
                    KuhnPoker.actions[0]
                    strategy[0]
                    KuhnPoker.actions[1]
                    strategy[1]

    let timer = Diagnostics.Stopwatch.StartNew()
    run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    