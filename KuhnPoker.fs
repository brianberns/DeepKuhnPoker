/// Kuhn poker
module KuhnPoker

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
