# phrase_matching_ltn
Neuro-symbolic model used to participate for the competition "U.S. Patent Phrase to Phrase Matching"

## TODO:
- [] text augmentation
- [] nesy integration: 
    - USE A KG takent from patent
    - USE synonim as rule:
        Forall <x,y> IsSyn(x,y) =>IsScore(x, 0.25)
        Forall <x,y> IsContrary(x,y) =>IsScore(x, 0.0)