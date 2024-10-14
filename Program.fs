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

        for player = 0 to KuhnPoker.numPlayers - 1 do
            printfn $"\nPlayer {player}"
            (*
            for infoSetKey in playerInfoSetKeys[player] do
                let strategy =
                    Trainer.getStrategy infoSetKey stratModel
                printfn "   %-3s: %s = %.3f, %s = %.3f"
                    infoSetKey
                    KuhnPoker.actions[0]
                    strategy[0]
                    KuhnPoker.actions[1]
                    strategy[1]
            *)

    let timer = Diagnostics.Stopwatch.StartNew()
    run ()
    printfn ""
    printfn $"Elapsed time: {timer}"
    