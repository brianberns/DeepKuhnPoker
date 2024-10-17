namespace DeepKuhnPoker

open System

[<AutoOpen>]
module Settings =

    /// Hyperparameters.
    let settings =
        let seed = 0
        TorchSharp.torch.manual_seed(seed) |> ignore
        {|
            /// Random number generator.
            Random = Random(seed)

            /// Size of a neural network hidden layer.
            HiddenSize = 32

            /// Optimizer learning rate.
            LearningRate = 1e-3

            /// Number of steps to use when training advantage models.
            NumAdvantageModelTrainSteps = 40

            /// Number of advantage samples to keep.
            NumAdvantageSamples = 2048

            /// Number of deals to traverse during each iteration.
            NumTraversals = 40

            /// Number of iterations to perform.
            NumIterations = 400

            /// Number of steps to use when training the strategy model.
            NumStrategyModelTrainSteps = 400

            /// Number of strategy samples to keep.
            NumStrategySamples = 1_000_000

            /// Tensorboard writer.
            Writer =
                let timespan = DateTime.Now - DateTime.Today
                TorchSharp.torch.utils.tensorboard.SummaryWriter(
                    $"runs/run%05d{int timespan.TotalSeconds}")
        |}
