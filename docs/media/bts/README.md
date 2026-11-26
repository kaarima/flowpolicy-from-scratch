# Behind the scenes: GIFs for the write-up

All from M2 (2026-11-26), PushT keypoints, test seeds 200000+. Panel border: green = success, red = fail.
Outcomes can differ from `best_eval500.json` for the same seed, because the flow sampling noise differs.

| GIF | What it shows |
|---|---|
| `01_same_start_base_B_film.gif` | Same start (seed 200000), three policies. Base (concat MLP, 8 % success) pushes the T into the top wall (max coverage 0.00); B (wide concat, 21 %) and FiLM (49.6 %) both succeed. |
| `02_film_near_miss.gif` | FiLM's most common failure: gets the T almost aligned and hovers (max coverage 0.95, just under the > 0.95 threshold). 172 of its 252 failures in the 500-episode eval are 0.90-0.95. |
| `03_film_worst.gif` | FiLM's other failure mode: pushes the T away from the goal into the bottom wall and never recovers (max coverage 0.00). |
| `04_base_8pct_typical.gif` | The 8 % baseline: coarse pushing in the right direction, no fine alignment (max coverage 0.80). |
