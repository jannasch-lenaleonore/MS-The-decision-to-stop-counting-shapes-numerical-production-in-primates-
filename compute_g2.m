function g2 = compute_g2(params, fix_drift, bound, obs_quantiles, obs_n, ...
        bins, n_bins, q_probs, n_sim, t_vec, n_t, dt, noise, period, ...
        resp_time_scaling, min_trials_per_cat)
% G² goodness-of-fit on RT quantile bins (Ratcliff & Tuerlinckx, 2002).
%
% obs_quantiles    - n_bins × numel(q_probs) empirical RT quantile boundaries
% obs_n            - n_bins × 1 observed trial counts per offset category
% q_probs          - quantile probability vector (e.g. [.10 .30 .50 .70 .90])

    [sim_rt, sim_off] = run_diffusion_sim(params, fix_drift, bound, n_sim, ...
        t_vec, n_t, dt, noise, period, resp_time_scaling);

    obs_bin_p = diff([0, q_probs, 1]);   % [.10 .20 .20 .20 .20 .10]

    g2 = 0;
    for b = 1 : n_bins
        if obs_n(b) < min_trials_per_cat || any(isnan(obs_quantiles(b,:)))
            continue
        end
        sim_cat   = sim_rt(sim_off == bins(b) & ~isnan(sim_rt));
        n_sim_cat = numel(sim_cat);
        if n_sim_cat < 10
            g2 = g2 + 1e6;
            continue
        end
        edges  = [-inf, obs_quantiles(b,:), inf];
        pred_p = histcounts(sim_cat, edges) / n_sim_cat;
        pred_p = max(pred_p, 1e-10);
        g2 = g2 + 2 * obs_n(b) * sum(obs_bin_p .* log(obs_bin_p ./ pred_p));
    end
end
