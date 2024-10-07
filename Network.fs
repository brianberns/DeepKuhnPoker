namespace DeepKuhnPoker

open TorchSharp
open type torch.nn

open MathNet.Numerics.LinearAlgebra

type Network = Module<torch.Tensor, torch.Tensor>

type AdvantageSample =
    {
        InfoSetKey : string
        Regrets : Vector<float32>
        Iteration : int
    }

type StrategySample =
    {
        InfoSetKey : string
        Strategy : Vector<float32>
        Iteration : int
    }

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

    let trainAdvantageNetwork
        samples
        network
        (optimizer : torch.optim.Optimizer)
        (criterion : Loss<_, _, torch.Tensor>) =

            // forward pass
        let loss =
            let inputs =
                samples
                    |> Seq.map (fun (sample : AdvantageSample) ->
                        sample.InfoSetKey
                            |> encodeInput)
                    |> array2D
                    |> torch.tensor
            let targets =
                samples
                    |> Seq.map (fun sample ->
                        sample.Regrets)
                    |> array2D
                    |> torch.tensor
            let outputs = inputs --> network
            criterion.forward(outputs, targets)

            // backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() |> ignore
