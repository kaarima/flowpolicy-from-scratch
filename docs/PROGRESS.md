# Progress

One line per session: date | milestone | result | next step

2026-11-26 | M0 setup | repo + venv (Py 3.10.12, gym-pusht 0.1.8, pymunk 6.11.1); lerobot/pusht: 206 eps, 25650 frames, len 49/124.5/246, state=(2,) agent xy only, action=(2,) in [12,511]; block pose via lerobot/pusht_keypoints environment_state (16,) = 8 T keypoints; env obs_type='state' (5,) = agent xy, block xy, theta | M1: 2D flow-matching toy in NumPy (own MLP + backprop + Adam)
2026-11-26 | M1 2D toy | NumPy MLP (SiLU) + manual backprop + Adam + linear-path flow matching + Euler sampler; 3 grad tests pass (torch, finite diff, Adam vs torch.optim) and fail on a broken SiLU grad; SW samples/floor: moons 0.054/0.046, checkerboard 0.027/0.021, 8gauss 0.024/0.017; checkerboard 30k steps visibly sharper squares but SW 0.028 (SW blind to corners) | M2: flow-matching action policy on PushT keypoints (obs -> action chunk), normalisation, eval in gym-pusht
