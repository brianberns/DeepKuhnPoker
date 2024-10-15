namespace DeepKuhnPoker

open System
open TorchSharp

module Program =

    let private playerInfoSetKeys =
        [|
            [| "J"; "Q"; "K"; "Jcb"; "Qcb"; "Kcb" |]
            [| "Jb"; "Jc"; "Qb"; "Qc"; "Kb"; "Kc" |]
        |]

    let run () =

            // train
        printfn "Running Kuhn Poker Deep CFR for %A iterations"
            settings.NumIterations
        let stratModel = Trainer.train ()

        let strategyMap =
            playerInfoSetKeys
                |> Seq.concat 
                |> Seq.map (fun infoSetKey ->
                    let strategy =
                        (StrategyModel.getStrategy infoSetKey stratModel)
                            .data<float32>()
                            .ToArray()
                    infoSetKey, strategy)
                |> Map

        for player = 0 to KuhnPoker.numPlayers - 1 do
            printfn $"\nPlayer {player}"
            for infoSetKey in playerInfoSetKeys[player] do
                let strategy = strategyMap[infoSetKey]
                printfn "   %-3s: %s = %.3f, %s = %.3f"
                    infoSetKey
                    KuhnPoker.actions[0]
                    strategy[0]
                    KuhnPoker.actions[1]
                    strategy[1]

        let betIdx = Array.IndexOf(KuhnPoker.actions, "b")
        let alpha = strategyMap["J"][betIdx]
        printfn ""
        printfn "Player 0"
        printfn "   J   bet: %.3f (should be between 0 and 1/3)" alpha
        printfn "   Q   bet: %.3f (should be 0)" (strategyMap["Q"][betIdx])
        printfn "   K   bet: %.3f (should be %.3f)" (strategyMap["K"][betIdx]) (3f * alpha)
        printfn "   Jcb bet: %.3f (should be 0)" (strategyMap["Jcb"][betIdx])
        printfn "   Qcb bet: %.3f (should be %.3f)" (strategyMap["Qcb"][betIdx]) (alpha + 1.f/3.f)
        printfn "   Kcb bet: %.3f (should be 1)" (strategyMap["Kcb"][betIdx])
        printfn ""
        printfn "Player 1"
        printfn "   Jb  bet: %.3f (should be 0)" (strategyMap["Jb"][betIdx])
        printfn "   Jc  bet: %.3f (should be 1/3)" (strategyMap["Jc"][betIdx])
        printfn "   Qb  bet: %.3f (should be 1/3)" (strategyMap["Qb"][betIdx])
        printfn "   Qc  bet: %.3f (should be 0)" (strategyMap["Qc"][betIdx])
        printfn "   Kb  bet: %.3f (should be 1)" (strategyMap["Kb"][betIdx])
        printfn "   Kc  bet: %.3f (should be 1)" (strategyMap["Kc"][betIdx])

    let timer = Diagnostics.Stopwatch.StartNew()
    run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    