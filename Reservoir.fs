module Reservoir

open System

type Reservoir<'t> =
    {
        Random : Random
        Capacity : int
        Items : Map<int, 't>
    }

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
