namespace DeepKuhnPoker

open TorchSharp
open type torch.nn

open MathNet.Numerics.LinearAlgebra

type Model = Module<torch.Tensor, torch.Tensor>

module Model =

    /// Length of neural network input.
    let inputSize = KuhnPoker.Encoding.encodedLength

    /// Length of neural network output.
    let outputSize = KuhnPoker.actions.Length

    /// Invokes the given model for the given info set.
    let invoke infoSetKey (model : Model) =
        (infoSetKey
            |> KuhnPoker.Encoding.encodeInput
            |> torch.tensor)
            --> model

/// An observed advantage event.
type AdvantageSample =
    {
        /// Key of info set.
        InfoSetKey : string

        /// Observed regrets.
        Regrets : Vector<float32>

        /// 0-based iteration number.
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
            Linear(hiddenSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, Model.outputSize))

    /// Trains the given model using the given samples.
    let train
        (samples : seq<AdvantageSample>)
        (optimizer : torch.optim.Optimizer)
        (criterion : Loss<_, _, torch.Tensor>)
        (model : AdvantageModel) =

            // forward pass
        let loss =
            let inputs =
                samples
                    |> Seq.map (fun sample ->
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
            let iters =
                samples
                    |> Seq.map (fun sample ->
                        (sample.Iteration + 1)   // make 1-based
                            |> float32
                            |> sqrt
                            |> Seq.singleton )
                    |> array2D
                    |> torch.tensor
            let outputs = inputs --> model
            criterion.forward(
                iters * outputs,   // favor newer iterations
                iters * targets)

            // backward pass and optimize
        (optimizer : torch.optim.Optimizer).zero_grad()
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
            Linear(hiddenSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, Model.outputSize),
            Softmax(dim = 1))
