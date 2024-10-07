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

    /// Length of neural network input.
    let private inputSize = KuhnPoker.Encoding.encodedLength

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

    /// Gets the advantage for the given info set.
    let getAdvantage infoSetKey (advantageNetwork : Network) =
        (infoSetKey
            |> KuhnPoker.Encoding.encodeInput
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
                            |> KuhnPoker.Encoding.encodeInput)
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
