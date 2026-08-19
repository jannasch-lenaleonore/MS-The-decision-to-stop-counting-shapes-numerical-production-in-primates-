function [rts, offsets] = run_diffusion_sim(params, fix_drift, bound, ...
        n_trials, t_vec, n_t, dt, noise, period, resp_time_scaling)
% Simulate the counting diffusion model; return scaled RTs and offset bins.
%
% params   - [drift_mean, sigma_z]              when fix_drift = true
%            [drift_mean, drift_sigma, sigma_z] when fix_drift = false
% rts      - 1×n_trials; NaN for trials that never crossed the bound
% offsets  - 1×n_trials of (stop_step - 3); NaN for no-stop trials

    drift_mean = params(1);
    if fix_drift
        drift_sigma = 0;
        sigma_z     = max(params(2), 1e-6);
    else
        drift_sigma = max(params(2), 1e-6);
        sigma_z     = max(params(3), 1e-6);
    end

    drift_vals = drift_mean + drift_sigma * randn(1, n_trials);
    dv_curr    = sigma_z * abs(randn(1, n_trials));   % folded-normal start
    stop_k     = NaN(1, n_trials);
    stop_t_arr = NaN(1, n_trials);
    stopped    = false(1, n_trials);

    for i = 2 : n_t
        active = ~stopped;
        if ~any(active), break; end
        n_act = sum(active);
        dv_curr(active) = max( ...
            dv_curr(active) + drift_vals(active) .* dt + ...
            noise * sqrt(dt) * randn(1, n_act), 0);
        k_i                 = ceil(t_vec(i) / period);
        crossed             = active & (dv_curr >= bound);
        stop_k(crossed)     = k_i;
        stop_t_arr(crossed) = t_vec(i);
        stopped(crossed)    = true;
    end

    offsets = stop_k - 3;

    valid  = ~isnan(stop_k);
    rt_au  = stop_t_arr(valid) - (stop_k(valid) - 1) * period;
    scales = resp_time_scaling(randi(numel(resp_time_scaling), 1, sum(valid)));

    rts        = NaN(1, n_trials);
    rts(valid) = rt_au .* scales;
end
