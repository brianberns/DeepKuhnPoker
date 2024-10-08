namespace DeepKuhnPoker

open TorchSharp
open type torch.nn

open MathNet.Numerics.LinearAlgebra

type Model = Module<torch.Tensor, torch.Tensor>

module private Model =

    /// Length of neural network input.
    let inputSize = KuhnPoker.Encoding.encodedLength

    /// Length of neural network output.
    let outputSize = KuhnPoker.actions.Length

type AdvantageSample =
    {
        InfoSetKey : string
        Regrets : Vector<float32>
        Iteration : int
    }

module AdvantageSample =

    let create infoSetKey regrets iteration =
        {
            InfoSetKey = infoSetKey
            Regrets = regrets
            Iteration = iteration
        }

type AdvantageModel = Model

module AdvantageModel =

    /// Creates an advantage model.
    let create hiddenSize : AdvantageModel =
        Sequential(
            Linear(Model.inputSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, Model.outputSize))

    /// Gets the advantage for the given info set.
    let getAdvantage infoSetKey (model : AdvantageModel) =
        (infoSetKey
            |> KuhnPoker.Encoding.encodeInput
            |> torch.tensor)
            --> model

    let train samples
        (optimizer : torch.optim.Optimizer)
        (criterion : Loss<_, _, torch.Tensor>)
        (model : AdvantageModel) =

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
            let outputs = inputs --> model
            criterion.forward(outputs, targets)

            // backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() |> ignore

type StrategySample =
    {
        InfoSetKey : string
        Strategy : Vector<float32>
        Iteration : int
    }

module StrategySample =

    let create infoSetKey regrets iteration =
        {
            InfoSetKey = infoSetKey
            Strategy = regrets
            Iteration = iteration
        }

type StrategyModel = Model

module StrategyModel =

    /// Creates a strategy model.
    let create hiddenSize : StrategyModel =
        Sequential(
            Linear(Model.inputSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, Model.outputSize),
            Softmax(dim = 1))
