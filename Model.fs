namespace DeepKuhnPoker

open TorchSharp
open TorchSharp.Modules
open type torch
open type torch.nn
open type torch.optim
open FSharp.Core.Operators   // reclaim "float32" and other F# operators

open MathNet.Numerics.LinearAlgebra

/// Neural network that maps an input tensor to an output
/// tensor.
type Network = Module<Tensor, Tensor>

/// Loss function.
type Loss = Loss<Tensor, Tensor, Tensor>

module Network =

    /// Length of neural network input.
    let inputSize = KuhnPoker.Encoding.encodedLength

    /// Length of neural network output.
    let outputSize = KuhnPoker.actions.Length

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

    /// Creates an advantage sample.
    let create infoSetKey regrets iteration =
        {
            InfoSetKey = infoSetKey
            Regrets = regrets
            Iteration = iteration
        }

/// Model used for learning advantages.
type AdvantageModel =
    {
        /// Neural network.
        Network : Network

        /// Training optimizer.
        Optimizer : Optimizer

        /// Training loss function.
        Loss : Loss
    }

module AdvantageModel =

    /// Creates an advantage model.
    let create hiddenSize learningRate =
        let network =
            Sequential(
                Linear(Network.inputSize, hiddenSize),
                ReLU(),
                Linear(hiddenSize, Network.outputSize))
        {
            Network = network
            Optimizer =
                Adam(
                    network.parameters(),
                    lr = learningRate)
            Loss = MSELoss()
        }

    /// Gets the advantage for the given info set.
    let getAdvantage infoSetKey model =
        (infoSetKey
            |> KuhnPoker.Encoding.encodeInput
            |> tensor)
            --> model.Network

    /// Trains the given model using the given samples.
    let train numSteps samples model =

            // prepare training data
        let inputs =
            samples
                |> Seq.map (fun sample ->
                    sample.InfoSetKey
                        |> KuhnPoker.Encoding.encodeInput)
                |> array2D
                |> tensor
        let targets =
            samples
                |> Seq.map (fun sample ->
                    sample.Regrets)
                |> array2D
                |> tensor
        let iters =
            samples
                |> Seq.map (fun sample ->
                    (sample.Iteration + 1)   // make 1-based
                        |> float32
                        |> sqrt
                        |> Seq.singleton )
                |> array2D
                |> tensor

        [|
            for _ = 1 to numSteps do

                    // forward pass
                let loss =
                    let outputs = inputs --> model.Network
                    model.Loss.forward(
                        iters * outputs,   // favor later iterations
                        iters * targets)

                    // backward pass and optimize
                model.Optimizer.zero_grad()
                loss.backward()
                model.Optimizer.step() |> ignore

                loss.item<float32>()
        |]

/// An observed strategy event.
type StrategySample =
    {
        /// Key of info set.
        InfoSetKey : string

        /// Observed strategy.
        Strategy : Vector<float32>

        /// O-based iteration number.
        Iteration : int
    }

module StrategySample =

    /// Creates a strategy sample.
    let create infoSetKey strategy iteration =
        {
            /// Key of info set.
            InfoSetKey = infoSetKey

            /// Observed strategy.
            Strategy = strategy

            /// 0-based iteration number.
            Iteration = iteration
        }

/// Model used for learning strategy.
type StrategyModel =
    {
        /// Neural network.
        Network : Network

        /// Training optimizer.
        Optimizer : Optimizer

        /// Training loss function.
        Loss : Loss

        /// Softmax layer.
        Softmax : Softmax
    }

module StrategyModel =

    /// Creates a strategy model.
    let create hiddenSize learningRate =
        let network =
            Sequential(
                Linear(Network.inputSize, hiddenSize),
                ReLU(),
                Linear(hiddenSize, Network.outputSize))
        {
            Network = network
            Optimizer =
                Adam(
                    network.parameters(),
                    lr = learningRate)
            Loss = MSELoss()
            Softmax = Softmax(dim = -1)
        }

    /// Trains the given model using the given samples.
    let train numSteps samples model =

            // prepare training data
        let inputs =
            samples
                |> Seq.map (fun sample ->
                    sample.InfoSetKey
                        |> KuhnPoker.Encoding.encodeInput)
                |> array2D
                |> tensor
        let targets =
            samples
                |> Seq.map (fun sample ->
                    sample.Strategy)
                |> array2D
                |> tensor
        let iters =
            samples
                |> Seq.map (fun sample ->
                    (sample.Iteration + 1)   // make 1-based
                        |> float32
                        |> sqrt
                        |> Seq.singleton )
                |> array2D
                |> tensor

        [|
            for _ = 1 to numSteps do

                    // forward pass
                let loss =
                    let outputs =
                        (inputs --> model.Network)
                            |> model.Softmax.forward
                    model.Loss.forward(
                        iters * outputs,   // favor later iterations
                        iters * targets)

                    // backward pass and optimize
                model.Optimizer.zero_grad()
                loss.backward()
                model.Optimizer.step() |> ignore

                loss.item<float32>()
        |]

    /// Gets the strategy for the given info set.
    let getStrategy infoSetKey model =
        (infoSetKey
            |> KuhnPoker.Encoding.encodeInput
            |> tensor)
            --> model.Network
            |> model.Softmax.forward
