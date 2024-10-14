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
        /// 0 to Count - 1.
        Items : Map<int, 't>
    }

    /// Number of items stored in this reservoir.
    member this.Count = this.Items.Count

module Reservoir =

    /// Validates the given reservoir.
    let private isValid reservoir =
        Seq.toArray reservoir.Items.Keys
            = [| 0 .. reservoir.Count - 1 |]

    /// Creates an empty reservoir.
    let create rng capacity =
        let reservoir =
            {
                Random = rng
                Capacity = capacity
                Items = Map.empty
            }
        assert(isValid reservoir)
        reservoir

    /// Adds the given item to the given reservior, replacing
    /// an existing item at random if necessary.
    let add item reservoir =
        assert(isValid reservoir)
        let idx =
            if reservoir.Count < reservoir.Capacity then
                reservoir.Count
            else
                reservoir.Random.Next(reservoir.Count)
        { reservoir with
            Items = Map.add idx item reservoir.Items }

    /// Adds the given items to the given reservior, replacing
    /// existing items at random if necessary.
    let addMany items reservoir =
        (reservoir, items)
            ||> Seq.fold (fun reservoir item ->
                add item reservoir)

    /// Answers up to the given number of items from the given
    /// reservoir at random.
    let sample numSamples reservoir =
        assert(isValid reservoir)
        let numSamples = min numSamples reservoir.Count
        let idxs = [| 0 .. reservoir.Count - 1 |]
        reservoir.Random.Shuffle(idxs)
        Seq.init numSamples (fun i ->
            reservoir.Items[idxs[i]])
