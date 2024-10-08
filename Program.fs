namespace DeepKuhnPoker

open System

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

open TorchSharp

module KuhnCfrTrainer =

    /// Repeating sequence of possible deals.
    let getDeals numDeals =
        let permutations =
            [|
                for card0 in KuhnPoker.deck do
                    for card1 in KuhnPoker.deck do
                        if card0 <> card1 then
                            [| card0; card1 |]
            |]
        Seq.init numDeals (fun i ->
            permutations[i % permutations.Length])

    /// Computes strategy for the given info set using the
    /// given advantage model.
    let private getStrategy infoSetKey model =
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
                    Choice1Of2 {
                        InfoSetKey = infoSetKey
                        Regrets = actionUtilities - utility
                        Iteration = iter
                    }
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
                    Choice2Of2 {
                        InfoSetKey = infoSetKey
                        Strategy = strategy
                        Iteration = iter
                    }
                -utility, append samples sample

        loop "" |> snd

    /// Creates an advantage model and optimizer.
    let private createAdvantageModel hiddenSize learningRate =
        let model = AdvantageModel.create hiddenSize
        let optim =
            torch.optim.Adam(
                model.parameters(),
                lr = learningRate)
        model, optim

    let private hiddenSize = 16
    let private learningRate = 0.01
    let private reservoirCapacity = 1000
    let private numModelTrainSteps = 20
    let private numSamples = 10

    let private updateAdvantageModel
        reservoir
        newSamples
        trainModel =

            // update reservoir
        let resv =
            (reservoir, newSamples)
                ||> Seq.fold (fun resv (advSample : AdvantageSample) ->
                    Reservoir.add advSample resv)

            // train model
        for _ = 1 to numModelTrainSteps do
            resv
                |> Reservoir.trySample numSamples
                |> Option.iter trainModel

        resv

    /// Trains for the given number of iterations.
    let train numIterations numTraversals =

            // gather chunks of deals
        let chunkPairs =
            getDeals (numIterations * numTraversals)
                |> Seq.chunkBySize numTraversals
                |> Seq.indexed

        let advModelPairs =
            Array.init KuhnPoker.numPlayers
                (fun _ ->
                    createAdvantageModel hiddenSize learningRate)
        let advLoss = torch.nn.MSELoss()
        let advReservoir = Reservoir.create rng reservoirCapacity

        (advReservoir, chunkPairs)
            ||> Seq.fold (fun resv (iter, chunk) ->

                    // traverse this chunk of deals
                let updatingPlayer = iter % KuhnPoker.numPlayers
                let advModel, advOptim = advModelPairs[updatingPlayer]
                let newSamples =
                    chunk
                        |> Array.collect (fun deal ->
                            traverse
                                iter deal updatingPlayer advModel)

                    // update advantages
                let advSamples =
                    newSamples
                        |> Seq.choose (function
                            | Choice1Of2 advSample -> Some advSample
                            | Choice2Of2 _ -> None)
                updateAdvantageModel
                    resv
                    advSamples
                    (fun samples ->
                        AdvantageModel.train
                            samples
                            advOptim
                            advLoss
                            advModel))
            |> ignore

module Program =

    let run () =

        torch.manual_seed(0) |> ignore

            // train
        let numIterations = 50
        let numTraversals = 6   // all possible deals
        printfn $"Running Kuhn Poker Deep CFR for {numIterations} iterations\n"
        KuhnCfrTrainer.train numIterations numTraversals

        (*
            // expected overall utility
        printfn $"Average game value for first player: %0.5f{util}\n"
        assert(abs(util - -1.0/18.0) <= 0.02)

            // strategy
        printfn "Strategy:"
        for (KeyValue(key, infoSet)) in infoSetMap do
            let str =
                let strategy =
                    InformationSet.getAverageStrategy infoSet
                (strategy.ToArray(), KuhnPoker.actions)
                    ||> Array.map2 (fun prob action ->
                        sprintf "%s: %0.5f" action prob)
                    |> String.concat ", "
            printfn $"%-3s{key}:    {str}"
        assert(
            let betAction =
                Array.IndexOf(KuhnPoker.actions, "b")
            let prob key =
                let strategy =
                    infoSetMap[key]
                        |> InformationSet.getAverageStrategy
                strategy[betAction]
            let k = prob "K"
            let j = prob "J"
            j >= 0.0 && j <= 1.0/3.0            // bet frequency for a Jack should be between 0 and 1/3
                && abs((k / j) - 3.0) <= 0.1)   // bet frequency for a King should be three times a Jack
        *)

    let timer = Diagnostics.Stopwatch.StartNew()
    run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    