namespace DeepKuhnPoker

open TorchSharp
open type torch.nn

open MathNet.Numerics.LinearAlgebra

type Network = Module<torch.Tensor, torch.Tensor>

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

    let create infoSetKey regrets iteration =
        {
            InfoSetKey = infoSetKey
            Regrets = regrets
            Iteration = iteration
        }

type AdvantageModel =
    {
        Network : Network
        Optimizer : torch.optim.Optimizer
        Loss : Loss<torch.Tensor, torch.Tensor, torch.Tensor>
    }

module AdvantageModel =

    /// Creates an advantage model.
    let create hiddenSize learningRate =
        let network =
            Sequential(
                Linear(Network.inputSize, hiddenSize),
                ReLU(),
                Linear(hiddenSize, hiddenSize),
                ReLU(),
                Linear(hiddenSize, Network.outputSize))
        {
            Network = network
            Optimizer =
                torch.optim.Adam(
                    network.parameters(),
                    lr = learningRate)
            Loss = torch.nn.MSELoss()
        }

    /// Gets the advantage for the given info set.
    let getAdvantage infoSetKey model =
        (infoSetKey
            |> KuhnPoker.Encoding.encodeInput
            |> torch.tensor)
            --> model.Network

    /// Trains the given model a single step using the given
    /// samples.
    let train samples model =

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
            let outputs = inputs --> model.Network
            model.Loss.forward(
                iters * outputs,   // favor newer iterations
                iters * targets)

            // backward pass and optimize
        model.Optimizer.zero_grad()
        loss.backward()
        model.Optimizer.step() |> ignore

        loss.item<float32>()

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

type StrategyModel =
    {
        Network : Network
        Optimizer : torch.optim.Optimizer
        Loss : Loss<torch.Tensor, torch.Tensor, torch.Tensor>
        Softmax : Modules.Softmax
    }

module StrategyModel =

    /// Creates a strategy model.
    let create hiddenSize learningRate =
        let network =
            Sequential(
                Linear(Network.inputSize, hiddenSize),
                ReLU(),
                Linear(hiddenSize, hiddenSize),
                ReLU(),
                Linear(hiddenSize, Network.outputSize))
        {
            Network = network
            Optimizer =
                torch.optim.Adam(
                    network.parameters(),
                    lr = learningRate)
            Loss = torch.nn.MSELoss()
            Softmax = torch.nn.Softmax(dim = -1)
        }

    /// Trains the given model using the given samples.
    let train samples model =

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
                        sample.Strategy)
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
            let outputs =
                (inputs --> model.Network)
                    |> model.Softmax.forward
            model.Loss.forward(
                iters * outputs,   // favor newer iterations
                iters * targets)

            // backward pass and optimize
        model.Optimizer.zero_grad()
        loss.backward()
        model.Optimizer.step() |> ignore

        loss.item<float32>()

    /// Gets the strategy for the given info set.
    let getStrategy infoSetKey model =
        (infoSetKey
            |> KuhnPoker.Encoding.encodeInput
            |> torch.tensor)
            --> model.Network
            |> model.Softmax.forward
