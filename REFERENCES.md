# References

- Yan, Germaine, Wang, Régin, Pelsser — *The complexity of Solitaire*,
  Theoretical Computer Science 410(38), 2009. Klondike deciding-win is
  NP-complete. https://doi.org/10.1016/j.tcs.2009.06.022
- Mnih et al. — *Human-level control through deep reinforcement learning*
  (DQN), Nature 518, 2015. https://doi.org/10.1038/nature14236
- van Hasselt et al. — *Deep Reinforcement Learning with Double Q-learning*
  (Double DQN), AAAI 2016. https://arxiv.org/abs/1509.06461
- Schulman et al. — *Proximal Policy Optimization Algorithms*, 2017.
  https://arxiv.org/abs/1707.06347
- sb3-contrib `MaskablePPO` — action-masked PPO used in M3.
  https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- Farama Gymnasium API. https://gymnasium.farama.org/
- RLCard — card-game RL toolkit (patterns for state encoding + masking;
  contains no Klondike). https://github.com/datamllab/rlcard
- Pubrant et al., *Learning to play Solitaire from human gameplay*,
  arXiv:2305.11449 (2023) — behavior cloning + MLP value baselines on
  Klondike; reports ~3.9%–4.9% win-rate for greedy value play on draw-3,
  useful context for why heuristic-level RL here is hard.
- Bjarnason, Tadepalli, Fern — *Searching Solitaire in Real Time*,
  ICAPS 2007 — search-based Klondike play; thought-not-RL but the strongest
  published line of work for automated Klondike play.
