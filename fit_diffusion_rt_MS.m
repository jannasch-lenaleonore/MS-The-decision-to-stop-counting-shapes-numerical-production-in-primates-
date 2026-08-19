% fit_diffusion_rt_MS.m
%
% Fits drift_mean (v) [and optionally drift_sigma (eta)], sigma_z
% to monkey RT distributions using the G² statistic on RT quantile bins.
%
% Model matches diffusion_model_ratcliff_MS.m exactly:
%   - folded-normal starting point: |N(0, sigma_z)|
%   - reflecting lower border at 0
%   - fixed bound = 1
%
% Procedure (Ratcliff & Tuerlinckx, 2002):
%   For each of the 5 offset categories (-2:+2):
%     1. Compute empirical RT quantiles [.1 .3 .5 .7 .9] from monkey data.
%        These divide the distribution into 6 bins with expected proportions
%        [.10  .20  .20  .20  .20  .10].
%     2. Simulate model RTs; compute fraction of simulated RTs in each bin.
%     3. Accumulate G² = 2 * sum_c sum_k  obs_ck * log(obs_ck / exp_ck)
%        where exp_ck = N_c * pred_p_ck.
%   Minimise G² using fminsearch with sigmoid transform for bounded search.

clear; clc;
rng(42)

% helpers/ holds run_diffusion_sim.m and compute_g2.m.  Added here relative
% to this file so the script runs from any working directory without the
% reviewer having to set up a path.
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'helpers'));

% =========================================================================
% 1.  Toggle
% =========================================================================
fix_drift = true;   % true: fix drift_sigma = 0 (only drift_mean + sigma_z fitted)

% =========================================================================
% 2.  Fixed model parameters
% =========================================================================
noise    = 0.1;
dt       = 0.01;
step_dur = 1;
bound    = 1;
RT_cutoff         = 1100;   % must match every other script (performance fit,
                            % Gaussian fits, CV, comparison figures)
resp_time_scaling = [200, 500, 800, 1100, 200, 200, 800, 400];

n_steps_rel = 7;
period      = step_dur;
t_vec       = (0 : dt : n_steps_rel * period)';
n_t         = numel(t_vec);

% =========================================================================
% 3.  Load monkey data (RT + offset per trial)
% =========================================================================
targets_monk = [3, 4];
project_dir  = fileparts(fileparts(mfilename('fullpath')));
monkeys      = {'m1', 'm2'};
monk_rt      = [];
monk_off_rt  = [];

for mk = 1 : numel(monkeys)
    load(fullfile(project_dir, 'data', ...
        sprintf('bhv_data_tbl_%s', monkeys{mk})));
    for sess = 1 : height(dataTable)
        RM = dataTable.RespMat{sess};
        RM(all(RM == 9, 2), :) = [];

        corr_mask    = RM(:,5) == 0;
        err_mask     = RM(:,5) == 6;
        rt_conf      = dataTable.RT_conf{sess};
        rt_err_conf  = dataTable.RT_err_conf{sess};
        corr_counter = cumsum(corr_mask);
        err_counter  = cumsum(err_mask);

        for tg = targets_monk
            tg_mask = RM(:,2) == tg;

            tg_corr_rows = find(corr_mask & tg_mask);
            if ~isempty(tg_corr_rows) && ~isempty(rt_conf)
                rt_c  = rt_conf(corr_counter(tg_corr_rows));
                valid = ~isnan(rt_c) & rt_c <= RT_cutoff;
                monk_rt     = [monk_rt;     rt_c(valid)];            
                monk_off_rt = [monk_off_rt; zeros(sum(valid), 1)];   
            end

            tg_err_rows = find(err_mask & tg_mask);
            if ~isempty(tg_err_rows) && ~isempty(rt_err_conf)
                rt_e  = rt_err_conf(err_counter(tg_err_rows));
                off_e = RM(tg_err_rows, 3) - tg;
                valid = ~isnan(rt_e) & rt_e <= RT_cutoff;
                monk_rt     = [monk_rt;     rt_e(valid)];   
                monk_off_rt = [monk_off_rt; off_e(valid)];   
            end
        end
    end
end

% =========================================================================
% 4.  Compute observed RT quantile boundaries per category (fixed for fit)
%
%     Quantiles are computed once from the data and never change during
%     optimisation.  The 5 quantile boundaries divide each category's RT
%     distribution into 6 bins with nominal proportions
%     [.10  .20  .20  .20  .20  .10].
% =========================================================================
bins    = -2 : 2;
n_bins  = numel(bins);
q_probs = [.10, .30, .50, .70, .90];    % 5 boundaries -> 6 bins
min_trials_per_cat = 20;

obs_quantiles = NaN(n_bins, numel(q_probs));
obs_n         = zeros(n_bins, 1);

fprintf('Observed RT quantiles (ms) per category:\n');
fprintf('  %-8s  %-6s   Q10    Q30    Q50    Q70    Q90\n', 'Offset', 'N');
for b = 1 : n_bins
    cat_rt     = monk_rt(monk_off_rt == bins(b) & ~isnan(monk_rt));
    obs_n(b)   = numel(cat_rt);
    if obs_n(b) >= min_trials_per_cat
        obs_quantiles(b,:) = quantile(cat_rt, q_probs);
        if bins(b) == 0, lbl = ' 0 (corr)'; else, lbl = sprintf('%+d', bins(b)); end
        fprintf('  %-8s  %-6d   ', lbl, obs_n(b));
        fprintf('%5.0f  ', obs_quantiles(b,:));
        fprintf('\n');
    else
        fprintf('  %+d         %-6d   (skipped — too few trials)\n', bins(b), obs_n(b));
    end
end
fprintf('\n');

% =========================================================================
% 5.  Optimisation setup
% =========================================================================
if fix_drift
    %              drift_mean   sigma_z
    x0 = [         0.4,         0.2  ];
    lb = [         0.1,        0.01 ];
    ub = [         1.0,         1.00 ];
    param_names = {'drift_mean','sigma_z'};
else
    %              drift_mean   drift_sigma   sigma_z
    x0 = [         0.4,        0.10,         0.2  ];
    lb = [         0.1,        0.001,        0.01 ];
    ub = [         1.0,        2.00,         1.00 ];
    param_names = {'drift_mean','drift_sigma','sigma_z'};
end
n_params = numel(x0);

% Larger n_sim needed here than for performance fitting: each category's
% RT distribution must be well-sampled for stable quantile bin probabilities.
n_sim = 20000;

to_raw    = @(p) log((p - lb) ./ (ub - p));
to_params = @(r) lb + (ub - lb) ./ (1 + exp(-r));

obj_raw = @(r) compute_g2(to_params(r), fix_drift, bound, ...
    obs_quantiles, obs_n, bins, n_bins, q_probs, n_sim, ...
    t_vec, n_t, dt, noise, period, resp_time_scaling, min_trials_per_cat);

n_starts = 20;

opts = optimset( ...
    'Display',     'off', ...
    'MaxFunEvals', 600,   ...
    'MaxIter',     300,   ...
    'TolX',        1e-3,  ...
    'TolFun',      1e-3);

rng(42)
starts = [x0; lb + (ub - lb) .* rand(n_starts - 1, n_params)];

fprintf('Multi-start fitting: %d starts, %d params, n_sim=%d...\n\n', ...
    n_starts, n_params, n_sim);

best_g2 = Inf;
best_r  = to_raw(x0);
all_g2  = NaN(n_starts, 1);

tic
for s = 1 : n_starts
    r0_s = to_raw(starts(s,:));
    [r_s, g2_s] = fminsearch(obj_raw, r0_s, opts);
    all_g2(s)   = g2_s;
    fprintf('  start %2d/%d  G² = %.2f  [%s]\n', s, n_starts, g2_s, ...
        strjoin(arrayfun(@(v) sprintf('%.3f', v), to_params(r_s), ...
        'UniformOutput', false), '  '));
    if g2_s < best_g2
        best_g2 = g2_s;
        best_r  = r_s;
    end
end
t_fit = toc;

x_opt  = to_params(best_r);
g2_opt = best_g2;

fprintf('\n--- Best result (%.0f s total) ---\n', t_fit);
for i = 1 : n_params
    fprintf('  %-14s = %.4f\n', param_names{i}, x_opt(i));
end
fprintf('  G²             = %.2f\n', g2_opt);
fprintf('  G² range       = [%.2f  %.2f]\n\n', min(all_g2), max(all_g2));

% =========================================================================
% 6.  Final simulation for plotting (3x larger for stable quantiles)
% =========================================================================
rng(0)
[sim_rt, sim_off] = run_diffusion_sim(x_opt, fix_drift, bound, n_sim * 3, ...
    t_vec, n_t, dt, noise, period, resp_time_scaling);

% =========================================================================
% 7.  Plot: RT quantile curves + histogram overlays
% =========================================================================
lw  = 1.5;
fs  = 12;
clr = [0.75 0.15 0.10;   % offset -2
       0.90 0.50 0.10;   % offset -1
       0.15 0.55 0.20;   % offset  0 (correct)
       0.10 0.40 0.80;   % offset +1
       0.50 0.10 0.70];  % offset +2

figure('Color','w','Units','centimeter','Position',[3 3 26 13])

be_rt = linspace(0, RT_cutoff, 21);

for b = 1 : n_bins
    monk_cat = monk_rt(monk_off_rt == bins(b) & ~isnan(monk_rt));
    sim_cat  = sim_rt(sim_off == bins(b) & ~isnan(sim_rt));
    if bins(b) == 0, lbl = 'correct'; else, lbl = sprintf('%+d', bins(b)); end

    % --- Top row: RT quantile curves --------------------------------------
    subplot(2, n_bins, b);  hold on

    if ~isempty(monk_cat)
        q_m = quantile(monk_cat, q_probs);
        plot(q_probs, q_m, 'o-', 'Color', clr(b,:), ...
            'MarkerFaceColor', clr(b,:), 'MarkerSize', 5, 'LineWidth', lw, ...
            'DisplayName', 'Monkey')
    end
    if ~isempty(sim_cat)
        q_s = quantile(sim_cat, q_probs);
        plot(q_probs, q_s, 's--', 'Color', clr(b,:), ...
            'MarkerFaceColor', 'w', 'MarkerSize', 5, 'LineWidth', lw, ...
            'DisplayName', 'Model')
    end

    xlim([0 1]);  ylim([0 RT_cutoff])
    xlabel('Quantile', 'FontSize', fs-1)
    if b == 1, ylabel('RT (ms)', 'FontSize', fs-1); end
    title(sprintf('Offset %s  (N=%d)', lbl, numel(monk_cat)), 'FontSize', fs-1)
    legend('Location', 'northwest', 'Box', 'off', 'FontSize', fs-2)
    set(gca, 'TickDir','out', 'LineWidth', lw, 'FontSize', fs-1, 'Box','off')

    % --- Bottom row: RT histogram overlays --------------------------------
    subplot(2, n_bins, n_bins + b);  hold on

    if ~isempty(monk_cat)
        n_m = histcounts(monk_cat, be_rt, 'Normalization', 'probability');
        stairs(be_rt(1:end-1), n_m, '-',  'Color', clr(b,:), 'LineWidth', lw)
    end
    if ~isempty(sim_cat)
        n_s = histcounts(sim_cat, be_rt, 'Normalization', 'probability');
        stairs(be_rt(1:end-1), n_s, '--', 'Color', clr(b,:), 'LineWidth', lw)
    end

    xlabel('RT (ms)', 'FontSize', fs-1)
    if b == 1, ylabel('Proportion', 'FontSize', fs-1); end
    set(gca, 'TickDir','out', 'LineWidth', lw, 'FontSize', fs-1, 'Box','off')
end

if fix_drift
    ttl = sprintf('RT fit: v=%.3f  \\sigma_z=%.3f  (\\eta fixed=0, \\theta=%.1f,  G^2=%.1f)', ...
        x_opt(1), x_opt(2), bound, g2_opt);
else
    ttl = sprintf('RT fit: v=%.3f  \\eta=%.3f  \\sigma_z=%.3f  (\\theta=%.1f,  G^2=%.1f)', ...
        x_opt(1), x_opt(2), x_opt(3), bound, g2_opt);
end
sgtitle(ttl, 'FontSize', fs + 1)

% =========================================================================
% Save fitted parameters
% =========================================================================
results_dir = fullfile(project_dir, 'results');
if ~isfolder(results_dir), mkdir(results_dir); end
save(fullfile(results_dir, 'params_rt.mat'), 'x_opt', 'g2_opt', 'fix_drift', 'param_names');
fprintf('Parameters saved to %s\n', fullfile(results_dir, 'params_rt.mat'));
