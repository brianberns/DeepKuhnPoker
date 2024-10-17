namespace DeepKuhnPoker

type Deal = string[]   // cards indexed by player

/// Kuhn poker
module KuhnPoker =

    /// Number of players.
    let numPlayers = 2

    /// Available player actions.
    let actions =
        [|
            "b"   // bet/call
            "c"   // check/fold
        |]

    /// Cards in the deck.
    let deck =
        [
            "J"   // Jack
            "Q"   // Queen
            "K"   // King
        ]

    /// All possible deals.
    let allDeals : Deal[] =
        [|
            for card0 in deck do
                for card1 in deck do
                    if card0 <> card1 then
                        [| card0; card1 |]
        |]

    /// Gets zero-based index of active player.
    let getActivePlayer (history : string) =
        history.Length % numPlayers

    /// Gets payoff for the active player if the game is over.
    let getPayoff (cards : string[]) = function

            // opponent folds - active player wins
        | "bc" | "cbc" -> Some 1

            // showdown
        | "cc" | "bb" | "cbb" as history ->
            let payoff =
                if history.Contains('b') then 2 else 1
            let activePlayer = getActivePlayer history
            let playerCard = cards[activePlayer]
            let opponentCard =
                cards[(activePlayer + 1) % numPlayers]
            match playerCard, opponentCard with
                | "K", _
                | _, "J" -> payoff   // active player wins
                | _ -> -payoff       // opponent wins
                |> Some

            // game not over
        | _ -> None

    module Encoding =

        /// Length of a one-hot vector.
        let private oneHotLength =
            max
                actions.Length   // 2
                deck.Length      // 3

        /// Length of longest info set key. E.g. "Jcb".
        let private maxInfoSetKeyLength = 3

        /// Length of an encoded info set key.
        let encodedLength = maxInfoSetKeyLength * oneHotLength

        /// Encodes the given info set key as a vector.
        let encodeInput (infoSetKey : string) =

            let toOneHot c =
                let oneHot =
                    match c with
                        | 'J' -> [| 1.0f; 0.0f; 0.0f |]
                        | 'Q' -> [| 0.0f; 1.0f; 0.0f |]
                        | 'K' -> [| 0.0f; 0.0f; 1.0f |]
                        | 'b' -> [| 1.0f; 0.0f; 0.0f |]
                        | 'c' -> [| 0.0f; 1.0f; 0.0f |]
                        | _ -> failwith "Unexpected"
                assert(oneHot.Length = oneHotLength)
                oneHot

            let encoded =
                [|
                    for c in infoSetKey do
                        yield! toOneHot c
                    yield! Array.zeroCreate
                        (oneHotLength * (maxInfoSetKeyLength - infoSetKey.Length))
                |]
            assert(encoded.Length = encodedLength)
            encoded
