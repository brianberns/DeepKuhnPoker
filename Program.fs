namespace DeepKuhnPoker

open System

module Program =

    let private infoSetKeys =
        [|
            "J"; "Q"; "K"; "Jcb"; "Qcb"; "Kcb"   // player 0
            "Jb"; "Jc"; "Qb"; "Qc"; "Kb"; "Kc"   // player 1
        |]

    let run () =

            // train the model
        printfn "Running Kuhn Poker Deep CFR for %A iterations"
            settings.NumIterations
        let stratModel = Trainer.train ()

            // examine resulting strategy
        let strategyMap =
            infoSetKeys
                |> Seq.map (fun infoSetKey ->
                    let strategy =
                        (StrategyModel.getStrategy infoSetKey stratModel)
                            .data<float32>()
                            .ToArray()
                    infoSetKey, strategy)
                |> Map
        let betIdx = Array.IndexOf(KuhnPoker.actions, "b")
        let lookup infoSetKey =
            strategyMap[infoSetKey][betIdx]
        let alpha = lookup "J"
        printfn ""
        printfn "Player 0"
        printfn "   J   bet: %.3f (should be between 0 and 1/3)" alpha
        printfn "   Q   bet: %.3f (should be 0)" (lookup "Q")
        printfn "   K   bet: %.3f (should be %.3f)" (lookup "K") (3f * alpha)
        printfn "   Jcb bet: %.3f (should be 0)" (lookup "Jcb")
        printfn "   Qcb bet: %.3f (should be %.3f)" (lookup "Qcb") (alpha + 1.f/3.f)
        printfn "   Kcb bet: %.3f (should be 1)" (lookup "Kcb")
        printfn ""
        printfn "Player 1"
        printfn "   Jb  bet: %.3f (should be 0)" (lookup "Jb")
        printfn "   Jc  bet: %.3f (should be 1/3)" (lookup "Jc")
        printfn "   Qb  bet: %.3f (should be 1/3)" (lookup "Qb")
        printfn "   Qc  bet: %.3f (should be 0)" (lookup "Qc")
        printfn "   Kb  bet: %.3f (should be 1)" (lookup "Kb")
        printfn "   Kc  bet: %.3f (should be 1)" (lookup "Kc")

        strategyMap

    let runTournament strategyMap =
        let player : Player =
            let strategyMap =
                Map.map (fun _ strat ->
                    MathNet.Numerics.LinearAlgebra
                        .DenseVector.ofArray strat)
                    strategyMap
            fun infoSetKey ->
                Vector.sample
                    settings.Random
                    strategyMap[infoSetKey]
        Player.runTournament
            [| player; Champion.player |]
            1000000

    let timer = Diagnostics.Stopwatch.StartNew()
    let strategyMap = run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    printfn ""
    printfn $"Average payoff: {runTournament strategyMap}"
