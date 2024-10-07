namespace DeepKuhnPoker

open System

/// Reservoir sampler.
/// https://en.wikipedia.org/wiki/Reservoir_sampling
type Reservoir<'t> =
    {
        /// Random number generator.
        Random : Random

        /// Capacity of this reservoir.
        Capacity : int

        /// Items stored in this reservoir, indexed from
        /// 0 to Capacity - 1.
        Items : Map<int, 't>
    }

    /// Number of items stored in this reservoir.
    member this.Count = this.Items.Count

module Reservoir =

    /// Creates an empty reservoir.
    let create rng capacity =
        {
            Random = rng
            Capacity = capacity
            Items = Map.empty
        }

    /// Adds the given item to the given reservior, replacing
    /// an existing item at random if necessary.
    let add item (reservoir : Reservoir<_>) =
        let idx =
            if reservoir.Count < reservoir.Capacity then
                reservoir.Count
            else
                reservoir.Random.Next(reservoir.Count)
        { reservoir with
            Items = Map.add idx item reservoir.Items }

    /// Answers the given number of items from the given
    /// reservoir at random, if possible.
    let trySample numSamples (reservoir : Reservoir<_>) =
        if numSamples <= reservoir.Count then
            let idxs = [| 0 .. reservoir.Count - 1 |]
            reservoir.Random.Shuffle(idxs)
            Seq.init numSamples (fun i ->
                reservoir.Items[idxs[i]])
                |> Some
        else None
