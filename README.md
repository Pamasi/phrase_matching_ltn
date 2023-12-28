# phrase_matching_ltn
The neuro-symbolic model used to participate in the competition "U.S. Patent Phrase to Phrase Matching"

## TODO:
- [] text augmentation
- [] nesy integration: 
    - USE A KG taken from the patent
    - USE synonym as rule:
        Forall <x,y> IsSyn(x,y) =>IsScore(x, 0.25)
        Forall <x,y> IsContrary(x,y) =>IsScore(x, 0.0)
