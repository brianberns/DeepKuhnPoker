namespace DeepKuhnPoker

open System

type Reservoir<'t> =
    {
        Random : Random
        Capacity : int
        Items : Map<int, 't>
    }

module Reservoir =

    let create rng capacity =
        {
            Random = rng
            Capacity = capacity
            Items = Map.empty
        }

    let add item reservoir =
        let idx =
            if reservoir.Items.Count < reservoir.Capacity then
                reservoir.Items.Count
            else
                reservoir.Random.Next(reservoir.Items.Count)
        { reservoir with
            Items = Map.add idx item reservoir.Items }

    let trySample numSamples reservoir =
        let count = reservoir.Items.Count
        if numSamples >= count then
            let indexes = [| 0 .. count-1 |]
            reservoir.Random.Shuffle(indexes)
            Seq.init numSamples (fun i -> indexes[i])
                |> Some
        else None
