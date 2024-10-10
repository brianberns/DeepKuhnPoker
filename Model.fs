namespace DeepKuhnPoker

open TorchSharp
open type torch.nn

open MathNet.Numerics.LinearAlgebra

type Network = Module<torch.Tensor, torch.Tensor>

module private Network =

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
        IsTrained : bool
        Network : Network
    }

module AdvantageModel =

    /// Creates an advantage model.
    let create hiddenSize : AdvantageModel =
        {
            IsTrained = false
            Network =
                Sequential(
                    Linear(Network.inputSize, hiddenSize),
                    ReLU(),
                    Linear(hiddenSize, Network.outputSize))
        }

    /// Gets the advantage for the given info set.
    let getAdvantage infoSetKey model =
        if model.IsTrained then
            (infoSetKey
                |> KuhnPoker.Encoding.encodeInput
                |> torch.tensor)
                --> model.Network
        else
            torch.ones(Network.outputSize)

    let train samples optimizer criterion model =

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
            (criterion : Loss<_, _, torch.Tensor>).forward(
                iters * outputs,   // favor newer iterations
                iters * targets)

            // backward pass and optimize
        (optimizer : torch.optim.Optimizer).zero_grad()
        loss.backward()
        optimizer.step() |> ignore

        { model with IsTrained = true }

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

type StrategyModel = Network

module StrategyModel =

    /// Creates a strategy model.
    let create hiddenSize : StrategyModel =
        Sequential(
            Linear(Network.inputSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, Network.outputSize),
            Softmax(dim = 1))
