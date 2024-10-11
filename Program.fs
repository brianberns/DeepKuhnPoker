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
    let createMap
        hiddenSize learningRate rng reservoirCapacity =
        Seq.init KuhnPoker.numPlayers (fun player ->
            let state =
                let model = AdvantageModel.create hiddenSize
                {
                    Model = model
                    Optimizer =
                        torch.optim.Adam(
                            model.Network.parameters(),
                            lr = learningRate)
                    Reservoir =
                        Reservoir.create rng reservoirCapacity
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

    /// Random number generator.
    let private rng = Random(0)

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
    let private learningRate = 1e-3
    let private reservoirCapacity = int 1e7
    let private numModelTrainSteps = 20

    /// Number of samples to use from the reservoir at each
    /// step of training.
    let private numSamples = 128

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
            (model, seq { 1 .. numModelTrainSteps })
                ||> Seq.fold (fun model _ ->
                    let samples =
                        Reservoir.sample numSamples resv
                    AdvantageModel.train
                        samples optim loss model)

        resv, model

    /// Trains for the given number of iterations.
    let train numIterations numTraversals =

            // create advantage model
        let advStateMap =
            AdvantageState.createMap
                hiddenSize learningRate rng reservoirCapacity
        let advLoss = torch.nn.MSELoss()

        let advStateMap =

                // for each iteration
            (advStateMap, seq { 0 .. numIterations - 1 })
                ||> Seq.fold (fun advStateMap iter ->

                        // for each player
                    (advStateMap, seq { 0 .. KuhnPoker.numPlayers - 1})
                        ||> Seq.fold (fun advStateMap updatingPlayer ->

                            let advModel, advOptim, advResv =
                                let state = advStateMap[updatingPlayer]
                                state.Model, state.Optimizer, state.Reservoir

                            let advSamples, stratSamples =
                                Choice.unzip [|
                                    for _ = 0 to numTraversals - 1 do
                                        let deal =
                                            let iDeal = rng.Next(KuhnPoker.allDeals.Length)
                                            KuhnPoker.allDeals[iDeal]
                                        yield! traverse
                                            iter deal updatingPlayer advModel
                                |]

                                // update advantages
                            let advResv, advModel =
                                updateAdvantageModel
                                    advResv advSamples advOptim advLoss advModel

                            AdvantageState.updateMap
                                updatingPlayer advModel advOptim advResv advStateMap))

        advStateMap.Values
            |> Seq.map (fun advState -> advState.Model)
            |> Seq.toArray

module Program =

    /// Number of CFR iterations to perform.
    let private numIterations = 50

    /// Number of deals to traverse during each iteration.
    let private numTraversals = 40

    let private playerInfoSetKeys =
        [|
            [| "J"; "Q"; "K"; "Jcb"; "Qcb"; "Kcb" |]
            [| "Jb"; "Jc"; "Qb"; "Qc"; "Kb"; "Kc" |]
        |]

    let run () =

        torch.manual_seed(0) |> ignore

            // train
        printfn $"Running Kuhn Poker Deep CFR for {numIterations} iterations"
        let advModels = KuhnCfrTrainer.train numIterations numTraversals

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
    