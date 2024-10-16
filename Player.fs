namespace DeepKuhnPoker

open MathNet.Numerics.LinearAlgebra

type Player = string (*info set*) -> int (*action*)

module Player =

    let private play cards (players : Player[]) =

        let rec loop history =
            match KuhnPoker.getPayoff cards history with
                | Some payoff -> payoff
                | None ->
                    let playerIdx = history.Length % players.Length
                    let infoSetKey = cards[playerIdx] + history
                    let actionIdx = players[playerIdx] infoSetKey
                    let history = history + KuhnPoker.actions[actionIdx]
                    -(loop history)

        loop ""

    let runTournament players numGames =
        let payoff =
            Seq.init numGames (fun _ ->
                let deal =
                    let iDeal =
                        settings.Random.Next(
                            KuhnPoker.allDeals.Length)
                    KuhnPoker.allDeals[iDeal]
                play deal players)
                |> Seq.sum
        float payoff / float numGames

module Champion =

    let player : Player =

        let alpha = settings.Random.NextDouble() / 3.

        let strategyMap =
            [
                "J", alpha
                "Q", 0
                "K", 3. * alpha
                "Jb", 0
                "Jc", 1./3.
                "Qb", 1./3.
                "Qc", 0
                "Kb", 1
                "Kc", 1
                "Jcb", 0
                "Qcb", alpha + 1./3.
                "Kcb", 1
            ]
                |> Seq.map (fun (infoSetKey, bet) ->
                    let strategy =
                        seq { bet; 1. - bet }
                            |> DenseVector.ofSeq
                    infoSetKey, strategy)
                |> Map

        fun infoSetKey ->
            strategyMap[infoSetKey]
                |> Vector.sample settings.Random
