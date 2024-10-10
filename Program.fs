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
    let private learningRate = 0.01
    let private reservoirCapacity = 1000
    let private numModelTrainSteps = 20

    /// Number of samples to use from the reservoir at each
    /// step of training.
    let private numSamples = reservoirCapacity

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

            // gather chunks of deals
        let chunkPairs =
            let numDeals = numIterations * numTraversals
            Seq.init numDeals (fun i ->
                KuhnPoker.allDeals[i % KuhnPoker.allDeals.Length])
                |> Seq.chunkBySize numTraversals
                |> Seq.indexed

            // create advantage model
        let advStateMap =
            AdvantageState.createMap
                hiddenSize learningRate rng reservoirCapacity
        let advLoss = torch.nn.MSELoss()

        let playerInfoSetKeys =
            [|
                [| "J"; "Q"; "K"; "Jcb"; "Qcb"; "Kcb" |]
                [| "Jb"; "Jc"; "Qb"; "Qc"; "Kb"; "Kc" |]
            |]

            // train the model on each chunk of deals
        let advStateMap =
            (advStateMap, chunkPairs)
                ||> Seq.fold (fun advStateMap (iter, chunk) ->

                        // traverse this chunk of deals
                    let updatingPlayer = iter % KuhnPoker.numPlayers
                    let advModel, advOptim, advResv =
                        let state = advStateMap[updatingPlayer]
                        state.Model, state.Optimizer, state.Reservoir
                    let advSamples, stratSamples = 
                        chunk
                            |> Array.collect (fun deal ->
                                traverse
                                    iter deal updatingPlayer advModel)
                            |> Choice.unzip

                        // update advantages
                    let advResv, advModel =
                        updateAdvantageModel
                            advResv advSamples advOptim advLoss advModel

                    printfn $"\nIter {iter}, Player {updatingPlayer}, Trained {advModel.IsTrained}"
                    printfn "   Training data:"
                    let sorted =
                        advSamples
                            |> Seq.sortBy (fun sample ->
                                sample.InfoSetKey.Length,
                                List.findIndex (fun card ->
                                    card = sample.InfoSetKey[0..0])
                                    KuhnPoker.deck,
                                sample.InfoSetKey)
                    for sample in sorted do
                        printfn "      %-3s: %s = %6.3f, %s = %6.3f (%d)"
                            sample.InfoSetKey
                            KuhnPoker.actions[0]
                            sample.Regrets[0]
                            KuhnPoker.actions[1]
                            sample.Regrets[1]
                            sample.Iteration
                    printfn "   Resulting model:"
                    for infoSetKey in playerInfoSetKeys[updatingPlayer] do
                        let advantages =
                            (AdvantageModel.getAdvantage infoSetKey advModel)
                                .data<float32>()
                                |> Seq.toArray
                        printfn "      %-3s: %s = %6.3f, %s = %6.3f (advantage)"
                            infoSetKey
                            KuhnPoker.actions[0]
                            advantages[0]
                            KuhnPoker.actions[1]
                            advantages[1]
                        let strategy =
                            getStrategy infoSetKey advModel
                        printfn "      %-3s: %s = %6.3f, %s = %6.3f (strategy)"
                            infoSetKey
                            KuhnPoker.actions[0]
                            strategy[0]
                            KuhnPoker.actions[1]
                            strategy[1]

                    AdvantageState.updateMap
                        updatingPlayer advModel advOptim advResv advStateMap)
        ()

module Program =

    /// Number of CFR iterations to perform.
    let private numIterations = 10

    /// Number of deals to traverse during each iteration.
    let private numTraversals = 10 * KuhnPoker.allDeals.Length

    let run () =

        torch.manual_seed(0) |> ignore

            // train
        printfn $"Running Kuhn Poker Deep CFR for {numIterations} iterations"
        KuhnCfrTrainer.train numIterations numTraversals

    let timer = Diagnostics.Stopwatch.StartNew()
    run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    