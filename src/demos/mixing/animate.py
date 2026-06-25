from EART_3017.src.demos.mixing.mixing_law_transport_sim_v2 import simulate_once, animate_path

# e.g., random case with modest correlation length
t, k, path = simulate_once('random', H=60, W=90, logsigma=0.6, corr_len=5)
animate_path(k, path, 'RANDOM', fps=20, step=1)  # saves mixing_v2_RANDOM_anim.gif