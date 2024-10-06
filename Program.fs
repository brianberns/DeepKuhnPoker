namespace DeepKuhnPoker

open System

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

open TorchSharp

module InformationSet =

    /// Uniform strategy: All actions have equal probability.
    let private uniformStrategy =
        DenseVector.create
            KuhnPoker.actions.Length
            (1.0f / float32 KuhnPoker.actions.Length)

    /// Normalizes a strategy such that its elements sum to
    /// 1.0 (to represent action probabilities).
    let private normalize strategy =

            // assume no negative values during normalization
        assert(Vector.forall (fun x -> x >= 0.0f) strategy)

        let sum = Vector.sum strategy
        if sum > 0.0f then strategy / sum
        else uniformStrategy

    /// Computes regret-matching strategy from given regrets.
    let getStrategy regrets =
        regrets
            |> Vector.map (max 0.0f)   // clamp negative regrets
            |> normalize

module KuhnCfrTrainer =

    /// Random number generator.
    let private rng = Random(0)

    let private getStrategy infoSetKey advantageNetwork =
        use _ = torch.no_grad()   // use model.eval() instead?
        (Network.getAdvantage infoSetKey advantageNetwork)
            .data<float32>()
            |> DenseVector.ofSeq
            |> InformationSet.getStrategy

    /// Negates opponent's utilties (assuming a zero-zum game).
    let private getActiveUtilities utilities =
        utilities
            |> Seq.map (~-)
            |> DenseVector.ofSeq

    /// Evaluates the utility of the given deal.
    let private traverse iter deal updatingPlayer advantageNetwork =

        /// Appends an item to the end of an array.
        let append items item =
            Array.append items (Array.singleton item)

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
            let strategy = getStrategy infoSetKey advantageNetwork

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

        loop ""

    /// Trains for the given number of iterations.
    let train numIterations numTraversals =

            // each iteration evaluates one possible deal
        let deals =
            let permutations =
                [|
                    for card0 in KuhnPoker.deck do
                        for card1 in KuhnPoker.deck do
                            if card0 <> card1 then
                                [| card0; card1 |]
                |]
            let nDeals = numIterations * numTraversals
            seq {
                for i = 0 to nDeals - 1 do
                    yield permutations[i % permutations.Length]
            }

        let chunkPairs =
            deals
                |> Seq.chunkBySize numTraversals
                |> Seq.indexed

        let advantageNetworks =
            Array.init KuhnPoker.numPlayers
                (fun _ ->
                    let network =
                        Network.createAdvantageNetwork 16
                    network,
                    torch.optim.Adam(network.parameters(), lr = 0.01))
        let advantageLoss = torch.nn.MSELoss()

        (Reservoir.create rng 1000, chunkPairs)
            ||> Seq.fold (fun resv (iter, chunk) ->

                let updatingPlayer = iter % KuhnPoker.numPlayers
                let advNetwork, advOptim = advantageNetworks[updatingPlayer]
                let samples =
                    [|
                        for deal in chunk do
                            let _, samples =
                                traverse
                                    iter
                                    deal
                                    updatingPlayer
                                    advNetwork
                            yield! samples
                    |]

                let advSamples =
                    samples
                        |> Seq.choose (function
                            | Choice1Of2 advSample -> Some advSample
                            | Choice2Of2 _ -> None)
                let resv =
                    (resv, advSamples)
                        ||> Seq.fold (fun resv advSample ->
                            Reservoir.add advSample resv)
                match Reservoir.trySample 100 resv with
                    | Some samples ->
                        Network.trainAdvantageNetwork
                            samples
                            advNetwork
                            advOptim
                            advantageLoss
                    | None -> ()
                resv)
            |> ignore

module Program =

    let run () =

        torch.manual_seed(0) |> ignore

            // train
        let numIterations = 50
        let numTraversals = 5
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
    