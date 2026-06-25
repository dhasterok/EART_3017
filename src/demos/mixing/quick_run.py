from EART_3017.src.demos.mixing.mixing_law_transport_sim_v2 import batch_times, save_example_and_hist# Random (geometric), Parallel (arithmetic), Series (harmonic)
for label, geom in [('RANDOM','random'), ('PARALLEL','parallel'), ('SERIES','series')]:   
    times, example = batch_times(geom, trials=300, H=60, W=90, logsigma=0.6,
                            corr_len=0,  # try 3--7 for grainy look
                            corr_frac=0.25, barrier_frac=0.18)
    print(f"{label}: mean={times.mean():.3f}, median={times.mean():.3f}")
    save_example_and_hist(times, example, label)