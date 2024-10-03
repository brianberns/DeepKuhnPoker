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

type AdvantageExperience =
    {
        InfoSetKey : string
        Regrets : Vector<float32>
        Iteration : int
    }

module KuhnCfrTrainer =

    /// Random number generator.
    let private rng = Random(0)

    let private numTraversals = 40

    let private advantageNetwork = Network.createAdvantageNetwork 32

    let private getStrategy infoSetKey =
        use _ = torch.no_grad()   // use model.eval() instead?
        (infoSetKey
            |> Network.encodeInput
            |> torch.tensor
            |> advantageNetwork.forward)
            .data<float32>()
            |> DenseVector.ofSeq
            |> InformationSet.getStrategy

    /// Negates opponent's utilties (assuming a zero-zum game).
    let private getActiveUtilities utilities =
        utilities
            |> Seq.map (~-)
            |> DenseVector.ofSeq

    let private advantageReservior = Reservoir.create rng 10000

    let private traverse deal updatingPlayer =

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
            let strategy = getStrategy infoSetKey

                // get utility of this info set
            let utility, keyedInfoSets =

                if activePlayer = updatingPlayer then

                        // get utility of each action
                    let actionUtilities, experiences =
                        let utilities, experienceArrays =
                            KuhnPoker.actions
                                |> Array.map (fun action ->
                                    loop (history + action))
                                |> Array.unzip
                        getActiveUtilities utilities,
                        Array.concat experienceArrays

                        // utility of this info set is action utilities weighted by action probabilities
                    let utility = actionUtilities * strategy

                    let experiences =
                        [|
                            yield! experiences
                            yield {
                                InfoSetKey = infoSetKey
                                Regrets = actionUtilities - utility
                                Iteration = -1
                            }
                        |]

                    utility, experiences

                else
                        // sample a single action according to the strategy
                    let action =
                        let strategy' =
                            strategy
                                |> Seq.map float   // ugly
                                |> Seq.toArray
                        Categorical.Sample(rng, strategy')
                            |> Array.get KuhnPoker.actions
                    let utility, experiences =
                        loop (history + action)
                    -utility, experiences

            utility, keyedInfoSets

        for _ = 1 to numTraversals do
            loop "" |> ignore

    /// Trains for the given number of iterations.
    let train numIterations =

        let utilities, () =

                // each iteration evaluates one possible deal
            let deals =
                let permutations =
                    [|
                        for card0 in KuhnPoker.deck do
                            for card1 in KuhnPoker.deck do
                                if card0 <> card1 then
                                    [| card0; card1 |]
                    |]
                seq {
                    for _ = 1 to numIterations do
                        yield permutations[rng.Next(permutations.Length)]   // avoid bias
                }

            ((), Seq.indexed deals)
                ||> Seq.mapFold (fun acc (i, deal) ->

                        // evaluate one game starting with this deal
                    let utility =
                        let updatingPlayer = i % KuhnPoker.numPlayers
                        traverse deal updatingPlayer

                    utility, acc)

            // compute average utility per deal
        let utility =
            Seq.sum utilities / float numIterations
        utility, infoSetMap

module Program =

    let run () =

            // train
        let numIterations = 500000
        printfn $"Running Kuhn Poker Monte Carlo CFR for {numIterations} iterations\n"
        let util, infoSetMap = KuhnCfrTrainer.train numIterations

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

    let timer = Diagnostics.Stopwatch.StartNew()
    run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    