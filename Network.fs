namespace DeepKuhnPoker

open TorchSharp
open type torch.nn

type Network = Module<torch.Tensor, torch.Tensor>

module Network =

    /// Length of longest info set key. E.g. "Jcb".
    let private maxInfoSetKeyLength = 3

    /// Length of a one-hot vector.
    let private oneHotLength =
        max
            KuhnPoker.actions.Length   // 2
            KuhnPoker.deck.Length      // 3

    /// Length of neural network input.
    let private inputSize = maxInfoSetKeyLength * oneHotLength

    /// Length of neural network output.
    let private outputSize = KuhnPoker.actions.Length

    /// Advantage network.
    let createAdvantageNetwork hiddenSize : Network =
        Sequential(
            Linear(inputSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, outputSize))

    /// Strategy network.
    let createStrategyNetwork hiddenSize : Network =
        Sequential(
            Linear(inputSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, outputSize),
            Softmax(dim = 1))

    /// Encodes the given info set key as a vector.
    let private encodeInput (infoSetKey : string) =

        let toOneHot c =
            let oneHot =
                match c with
                    | 'J' -> [| 1.0f; 0.0f; 0.0f |]
                    | 'K' -> [| 0.0f; 1.0f; 0.0f |]
                    | 'Q' -> [| 0.0f; 0.0f; 1.0f |]
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
        assert(encoded.Length = inputSize)
        encoded

    /// Gets the advantage for the given info set.
    let getAdvantage infoSetKey (advantageNetwork : Network) =
        (infoSetKey
            |> encodeInput
            |> torch.tensor)
            --> advantageNetwork
