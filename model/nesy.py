import torch
import torch.nn.functional as F
import ltn


class NeSyLoss():
    """neurosymbolic module
    """
    def __init__(self, aggr_p:int=2) -> None:
        self.p_is_similar = ltn.Predicate(func=lambda anchor, candidate : 0.5 + F.cosine_similarity(anchor, candidate, dim = -1 )*0.5 )
        self.p_is_score = ltn.Predicate(func=lambda pred, tgt : torch.sum( pred * tgt , dim = -1 ))

        self.forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
        self.implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.sat_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

    def __call__(self, anchor_emb:torch.Tensor, cand_emb:torch.Tensor,
                        pred_scores:torch.Tensor, tgt_scores:torch.Tensor) ->torch.Tensor:
        """compute loss of criterion satisfability, i.e. if two embedding are similary, then the score is high

        Args:
            anchor_emb (torch.Tensor): embedding of the anchor
            cand_emb (torch.Tensor): _embedding of the candidate
            pred_scores (torch.Tensor): predicted score
            tgt_scores (torch.Tensor): target score

        Returns:
            torch.Tensor: _description_
        """
        
        anchor_emb_var = ltn.Variable('anchor_emb', anchor_emb)
        cand_emb_var = ltn.Variable('anchor_emb', cand_emb)

        pred_scores_var = ltn.Variable('pred_score', pred_scores)
        tgt_scores_var = ltn.Variable('tgt_score', tgt_scores)

        out = self.forall( ltn.diag(anchor_emb_var, cand_emb_var, pred_scores_var, tgt_scores_var),
                          self.implies(self.p_is_similar(anchor_emb_var, cand_emb_var), 
                                        self.p_is_score(pred_scores_var, tgt_scores_var)
                            ) 
        )

        

        return 1 - self.sat_axiom(out)