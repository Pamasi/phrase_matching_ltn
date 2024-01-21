import torch
import torch.nn.functional as F
import ltn


class NeSyLoss():
    """neurosymbolic module
    """
    def __init__(self, aggr_p:int=2, pos_idx:int=2, strategy:int=0) -> None:
        """create the neuro-symbolic loss

        Args:
            aggr_p (int, optional): aggregator norm value. Defaults to 2.
            pos_idx (int, optional): index value of positive score. Defaults to 2.
            strategy (int, optional): strategy to be use for computing the nesy loss. Defaults to 0.

        Raises:
            ValueError: Invalid stragegy
        """
        self.p_is_similar = ltn.Predicate(func=lambda anchor, candidate : 0.5 + F.cosine_similarity(anchor, candidate, dim = -1 )*0.5 )
        self.p_is_score = ltn.Predicate(func=lambda pred, tgt : torch.sum( pred * tgt , dim = -1 ))


        self.forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
        self.implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.sat_axiom = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=aggr_p))

        self.pos_idx = pos_idx

        if strategy > 2:
            raise ValueError(f'{strategy} is an invalid strategy')   
        
        elif strategy == 2:
            self.and_luk = ltn.Connective(ltn.fuzzy_ops.AndLuk())
        
        self.strategy = strategy


           

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
     


        if self.strategy == 0:
                anchor_emb_var = ltn.Variable('anchor_emb', anchor_emb)
                cand_emb_var = ltn.Variable('anchor_emb', cand_emb)

                pred_scores_var = ltn.Variable('pred_score', pred_scores)
                tgt_scores_var = ltn.Variable('tgt_score', tgt_scores)

                out = self.forall(  ltn.diag(anchor_emb_var, cand_emb_var, pred_scores_var, tgt_scores_var),
                                    self.implies(self.p_is_similar(anchor_emb_var, cand_emb_var), 
                                                self.p_is_score(pred_scores_var, tgt_scores_var)
                                    ) 
                )

                loss = 1 - self.sat_axiom(out)
        else:
            #  filter positive score to use imply
            pos_idx_t = torch.tensor([ 1 if t.argmax() >= self.pos_idx else 0 for b in tgt_scores for t in b])
            neg_idx_t = torch.tensor([ 1 if t.argmax() <  self.pos_idx else 0 for b in tgt_scores for t in b])

    

            anchor_emb_var_p = ltn.Variable('anchor_emb', anchor_emb[pos_idx_t])
            cand_emb_var_p = ltn.Variable('anchor_emb', cand_emb[pos_idx_t])

            pred_scores_var_p = ltn.Variable('pred_score', pred_scores[pos_idx_t])
            tgt_scores_var_p = ltn.Variable('tgt_score', tgt_scores[pos_idx_t])

            anchor_emb_var_n = ltn.Variable('anchor_emb', anchor_emb[neg_idx_t])
            cand_emb_var_n = ltn.Variable('anchor_emb', cand_emb[neg_idx_t])

            pred_scores_var_n = ltn.Variable('pred_score', pred_scores[neg_idx_t])
            tgt_scores_var_n = ltn.Variable('tgt_score', tgt_scores[neg_idx_t])


            if self.strategy == 1:
                out_pos  = self.forall( ltn.diag(anchor_emb_var_p, cand_emb_var_p, pred_scores_var_p, tgt_scores_var_p),
                                        self.implies(self.p_is_similar(anchor_emb_var_p, cand_emb_var_p), 
                                                        self.p_is_score(pred_scores_var_p, tgt_scores_var_p)
                                            ) 
                ) 


                out_neg  = self.forall( ltn.diag(anchor_emb_var_n, cand_emb_var_n, pred_scores_var_n, tgt_scores_var_n),
                                        self.implies(self.p_is_similar(anchor_emb_var_n, cand_emb_var_n), 
                                                        self.p_is_score(pred_scores_var_n, tgt_scores_var_n)
                                            ) 
                ) 

            else:
                out_pos  = self.forall( ltn.diag(anchor_emb_var_p, cand_emb_var_p, pred_scores_var_p, tgt_scores_var_p),
                                       self.and_luk(self.p_is_score(pred_scores_var_p, tgt_scores_var_p),
                                                    self.implies(self.p_is_similar(anchor_emb_var_p, cand_emb_var_p), 
                                                                    self.p_is_score(pred_scores_var_p, tgt_scores_var_p)
                                                    )
                                    ) 
                ) 


                out_neg  = self.forall( ltn.diag(anchor_emb_var_n, cand_emb_var_n, pred_scores_var_n, tgt_scores_var_n),
                                       self.and_luk(self.p_is_score(pred_scores_var_n, tgt_scores_var_n), 
                                                    self.implies(self.p_is_similar(anchor_emb_var_n, cand_emb_var_n), 
                                                                        self.p_is_score(pred_scores_var_n, tgt_scores_var_n)
                                                            ) 
                                       )
                ) 

            loss = 1 -  self.sat_axiom(out_pos, out_neg)

                   

        

        return loss