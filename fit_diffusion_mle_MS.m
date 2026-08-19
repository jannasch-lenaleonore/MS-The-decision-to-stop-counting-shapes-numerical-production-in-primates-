% fit_diffusion_mle_MS.m
%
% MLE fitting of diffusion_model_ratcliff parameters to monkey
% performance data (multinomial over offset bins -2:+2 + no-stop).
% The no-stop / outside-range category has observed count = 0: every
% no-stop trial the model produces reduces pred_p(1:5), implicitly
% penalising the model for failing to stop within -2:+2.
%
% Free parameters:
%   drift_mean  (v)    - mean of trial-to-trial drift distribution
%   drift_sigma (eta)  - SD of trial-to-trial drift distribution
%   sigma_z            - SD of folded-normal starting-point distribution
%
% All other model parameters are held fixed.
%
% The fit is run three times — pooled over both animals, and separately for
% m1 and m2 — and all three results are written to results/params_perf.mat.
% Within a fit the parameters are shared across target numerosities: the
% model is invariant to target by construction, so one parameter set has to
% account for both targets. Observed counts are stored per target as well,
% so model_comparison_fig4.m can evaluate the fit target by target.

clear; clc;
rng(42)

% =========================================================================
% 1.  Toggle
% =========================================================================
fix_drift = true;   % set to true to fix drift_sigma = 0 (only drift_mean is fitted)

% true: tiny run to check the script end to end (~1 min).  The full fit
% takes ~25 min per set, so verify changes with this before committing.
quick_test = false;

% =========================================================================
% 2.  Fixed model parameters
% =========================================================================
noise    = 0.1;
dt       = 0.01;
step_dur = 1;
bound    = 1;

n_steps_rel = 7;
period      = step_dur;
t_vec       = (0 : dt : n_steps_rel * period)';
n_t         = numel(t_vec);

% =========================================================================
% 3.  Data selection, response categories, and fit sets
% =========================================================================
targets_monk = [3, 4];
RT_cutoff    = 1100;    % exclude monkey RTs above this value (ms) --> invalid
project_dir  = fileparts(fileparts(mfilename('fullpath')));

fit_sets   = {{'m1', 'm2'}, {'m1'}, {'m2'}};
set_labels = {'pooled', 'm1', 'm2'};
n_sets     = numel(fit_sets);
n_targets  = numel(targets_monk);

bins   = -2 : 2;
n_bins = numel(bins);
labels = {'-2','-1','0','+1','+2','no-stop'};

% =========================================================================
% 4.  Optimization bounds and starting values
% =========================================================================
if fix_drift
    %              drift_mean   sigma_z
    x0 = [         0.4,         0.2  ];
    lb = [         0.1,         0.01 ];
    ub = [         1,           1.00 ];
    param_names = {'drift_mean','sigma_z'};
else
    %              drift_mean   drift_sigma   sigma_z
    x0 = [         0.4,        0.10,         0.2  ];
    lb = [         0.25,       0.01,         0.01 ];
    ub = [         0.5,        2.00,         1.00  ];
    param_names = {'drift_mean','drift_sigma','sigma_z'};
end
n_params = numel(x0);

% =========================================================================
% 5.  Number of simulation trials per likelihood evaluation
%
%     Higher n_sim → smoother likelihood surface → better convergence,
%     but slower per evaluation.  Increase to 50 000–100 000 if stalling.
% =========================================================================
n_sim = 20000;

% Trials used for the FINAL evaluation of the fitted parameters.  This is a
% separate, much larger simulation: the value reported for model comparison
% must not carry the Monte-Carlo noise of a single small run.  At n_eval =
% 5e5 the seed-to-seed SD of the neg-LL is ~2 units, against model
% differences of hundreds (see the seed check printed below).
n_eval       = 500000;
n_seed_check = 10;      % re-evaluations under different seeds

if quick_test
    n_sim = 3000;  n_eval = 20000;  n_seed_check = 2;
    fprintf('*** QUICK TEST — results are not usable ***\n\n');
end

% =========================================================================
% 6.  Multi-start fminsearch with sigmoid transform to enforce bounds
%
%     Bounds are enforced via a logistic (sigmoid) transform:
%       raw (unconstrained) -> bounded:  p = lb + (ub-lb) ./ (1+exp(-r))
%       bounded -> raw:                  r = log((p-lb) ./ (ub-p))
%
%     Multi-start: n_starts random starting points drawn uniformly from
%     the bounded parameter space.  The run with the lowest neg-LL wins.
% =========================================================================
n_starts = 20;
if quick_test, n_starts = 2; end

to_raw    = @(p) log((p - lb) ./ (ub - p));
to_params = @(r) lb + (ub - lb) ./ (1 + exp(-r));

max_fun_evals = 600;
if quick_test, max_fun_evals = 40; end

opts = optimset( ...
    'Display',     'off', ...
    'MaxFunEvals', max_fun_evals, ...
    'MaxIter',     300,   ...
    'TolX',        1e-3,  ...
    'TolFun',      1e-3);

lw = 1.5;  fs = 12;
clr_monk  = [1.0 1.0 1.0];
clr_model = [0.4 0.4 0.4];

% Field names and their order must match the struct assigned below exactly.
fit_perf = struct('label', {}, 'monkeys', {}, 'param_names', {}, ...
    'x_opt', {}, 'nll_opt', {}, 'nll_range', {}, ...
    'nll_eval', {}, 'nll_eval_sd', {}, 'nll_eval_range', {}, 'n_eval', {}, ...
    'k', {}, 'pred_p', {}, 'obs_counts', {}, 'obs_counts_tg', {}, ...
    'n_obs', {}, 'n_dropped', {});

% Checkpointing: each completed fit set is written to a partial file, and a
% re-run skips sets that are already in it.  A full run takes ~80 min, so an
% interruption should cost one fit set, not all three.  Delete the partial
% file to force a clean refit.
results_dir = fullfile(project_dir, 'results');
if ~isfolder(results_dir), mkdir(results_dir); end
partial_file = fullfile(results_dir, 'params_perf_partial.mat');
if quick_test, partial_file = fullfile(results_dir, 'params_perf_partial_QUICKTEST.mat'); end

if isfile(partial_file)
    P = load(partial_file);
    if isfield(P, 'fit_perf') && isequal(P.fix_drift, fix_drift)
        fit_perf = P.fit_perf;
        fprintf('Resuming: %s already contains [%s].\n\n', ...
            partial_file, strjoin({fit_perf.label}, ', '));
    end
end

for fs_i = 1 : n_sets
    set_monkeys = fit_sets{fs_i};

    if any(strcmp({fit_perf.label}, set_labels{fs_i}))
        fprintf('Fit set "%s" already done — skipping.\n\n', set_labels{fs_i});
        continue
    end

    % ---------------------------------------------------------------------
    % Observed counts, tallied per target and pooled over targets
    % ---------------------------------------------------------------------
    obs_counts_tg = zeros(n_targets, n_bins + 1);
    n_dropped     = 0;

    for mk = 1 : numel(set_monkeys)
        S = load(fullfile(project_dir, 'data', ...
            sprintf('bhv_data_tbl_%s', set_monkeys{mk})));
        dataTable = S.dataTable;
        for sess = 1 : height(dataTable)
            RM = dataTable.RespMat{sess};
            RM(all(RM == 9, 2), :) = [];

            corr_mask    = RM(:,5) == 0;
            err_mask     = RM(:,5) == 6;
            rt_conf      = dataTable.RT_conf{sess};
            rt_err_conf  = dataTable.RT_err_conf{sess};
            corr_counter = cumsum(corr_mask);
            err_counter  = cumsum(err_mask);

            for ti = 1 : n_targets
                tg       = targets_monk(ti);
                tg_mask  = RM(:,2) == tg;
                off_sess = [];

                tg_corr_rows = find(corr_mask & tg_mask);
                if ~isempty(tg_corr_rows) && ~isempty(rt_conf)
                    rt_c     = rt_conf(corr_counter(tg_corr_rows));
                    valid    = ~isnan(rt_c) & rt_c <= RT_cutoff;
                    off_sess = [off_sess; zeros(sum(valid), 1)]; %#ok<AGROW>
                end

                tg_err_rows = find(err_mask & tg_mask);
                if ~isempty(tg_err_rows) && ~isempty(rt_err_conf)
                    rt_e     = rt_err_conf(err_counter(tg_err_rows));
                    off_e    = RM(tg_err_rows, 3) - tg;
                    valid    = ~isnan(rt_e) & rt_e <= RT_cutoff;
                    off_sess = [off_sess; off_e(valid)]; %#ok<AGROW>
                end

                for b = 1 : n_bins
                    obs_counts_tg(ti, b) = obs_counts_tg(ti, b) + sum(off_sess == bins(b));
                end
                n_dropped = n_dropped + sum(abs(off_sess) > 2);
            end
        end
    end

    obs_counts      = sum(obs_counts_tg, 1);
    obs_counts(end) = 0;       % monkey never fails to stop within -2:+2
    n_obs           = sum(obs_counts);

    fprintf('=========================================================\n');
    fprintf('Fit set "%s"  (%s)\n', set_labels{fs_i}, strjoin(set_monkeys, ', '));
    fprintf('=========================================================\n');
    fprintf('Loaded %d trials within -2:+2 (%d dropped outside range)\n', n_obs, n_dropped);
    fprintf('Observed proportions  ');
    for i = 1 : n_bins
        fprintf('%s=%.1f%%  ', labels{i}, 100*obs_counts(i)/n_obs);
    end
    fprintf('%s=%.1f%%', labels{end}, 0.0);
    fprintf('\n\n');

    % ---------------------------------------------------------------------
    % Multi-start fit
    % ---------------------------------------------------------------------
    obj_raw = @(r) neg_log_lik(to_params(r), fix_drift, bound, obs_counts, n_sim, ...
        n_bins, bins, t_vec, n_t, dt, noise, period);

    rng(42)
    starts = [x0; lb + (ub - lb) .* rand(n_starts - 1, n_params)];

    fprintf('Multi-start fitting: %d starts, %d params, n_sim=%d...\n\n', ...
        n_starts, n_params, n_sim);

    best_nll = Inf;
    best_r   = to_raw(x0);
    all_nll  = NaN(n_starts, 1);

    tic
    for s = 1 : n_starts
        r0_s = to_raw(starts(s,:));
        [r_s, nll_s] = fminsearch(obj_raw, r0_s, opts);
        all_nll(s)   = nll_s;
        fprintf('  start %2d/%d  neg-LL = %.2f  [%s]\n', s, n_starts, nll_s, ...
            strjoin(arrayfun(@(v) sprintf('%.3f', v), to_params(r_s), ...
            'UniformOutput', false), '  '));
        if nll_s < best_nll
            best_nll = nll_s;
            best_r   = r_s;
        end
    end
    t_fit = toc;

    x_opt   = to_params(best_r);
    nll_opt = best_nll;

    fprintf('\n--- Best result (%.0f s total) ---\n', t_fit);
    for i = 1 : n_params
        fprintf('  %-14s = %.4f\n', param_names{i}, x_opt(i));
    end
    fprintf('  neg-LL         = %.2f\n', nll_opt);
    fprintf('  neg-LL range   = [%.2f  %.2f]\n\n', min(all_nll), max(all_nll));

    % ---------------------------------------------------------------------
    % Final model evaluation (much larger n_eval for stable comparison)
    %
    % nll_opt above is the minimum of a STOCHASTIC objective, so it is
    % optimistically biased: fminsearch partly chases favourable noise
    % draws.  nll_eval re-evaluates the same parameters on a fresh, large
    % simulation and is the value that should be used for model comparison.
    % ---------------------------------------------------------------------
    rng(0)
    pred_p = simulate_model(x_opt, fix_drift, bound, n_eval, ...
        n_bins, bins, t_vec, n_t, dt, noise, period);
    nll_eval = -sum(obs_counts .* log(pred_p));

    % Seed check: how much does Monte-Carlo noise alone move the neg-LL?
    nll_seeds = NaN(1, n_seed_check);
    for s = 1 : n_seed_check
        rng(1000 + s)
        pp_s = simulate_model(x_opt, fix_drift, bound, n_eval, ...
            n_bins, bins, t_vec, n_t, dt, noise, period);
        nll_seeds(s) = -sum(obs_counts .* log(pp_s));
    end

    fprintf('  neg-LL (fit, n_sim=%d)    = %.2f   <- optimistically biased\n', ...
        n_sim, nll_opt);
    fprintf('  neg-LL (eval, n=%d)   = %.2f\n', n_eval, nll_eval);
    fprintf('  seed check (%d seeds)      : SD = %.2f, range [%.2f  %.2f]\n\n', ...
        n_seed_check, std(nll_seeds), min(nll_seeds), max(nll_seeds));

    monk_pct = 100 * obs_counts / n_obs;
    pred_pct = 100 * pred_p;

    fprintf('%-10s', 'Category:');  fprintf('%-9s', labels{:});    fprintf('\n');
    fprintf('%-10s', 'Monkey %:');  fprintf('%-9.1f', monk_pct);   fprintf('\n');
    fprintf('%-10s', 'Model %:');   fprintf('%-9.1f', pred_pct(1:n_bins)); fprintf('\n');

    % The model is target-invariant, so the same pred_p applies to each
    % target; report the fit per target to make that assumption checkable.
    fprintf('\nPer target (same 2 parameters for both):\n');
    for ti = 1 : n_targets
        oc_t = obs_counts_tg(ti, :);
        fprintf('  target %d (N=%5d):  ', targets_monk(ti), sum(oc_t));
        fprintf('%-8.1f', 100 * oc_t(1:n_bins) / sum(oc_t));
        fprintf('  | neg-LL = %.2f\n', -sum(oc_t .* log(pred_p)));
    end
    fprintf('\n');

    fit_perf(end + 1) = struct( ... %#ok<SAGROW>
        'label',         set_labels{fs_i}, ...
        'monkeys',       {set_monkeys}, ...
        'param_names',   {param_names}, ...
        'x_opt',         x_opt, ...
        'nll_opt',       nll_opt, ...
        'nll_range',     [min(all_nll), max(all_nll)], ...
        'nll_eval',      nll_eval, ...
        'nll_eval_sd',   std(nll_seeds), ...
        'nll_eval_range',[min(nll_seeds), max(nll_seeds)], ...
        'n_eval',        n_eval, ...
        'k',             n_params, ...
        'pred_p',        pred_p, ...
        'obs_counts',    obs_counts, ...
        'obs_counts_tg', obs_counts_tg, ...
        'n_obs',         n_obs, ...
        'n_dropped',     n_dropped);

    save(partial_file, 'fit_perf', 'fix_drift', 'param_names');

    % ---------------------------------------------------------------------
    % Plot
    % ---------------------------------------------------------------------
    figure('Color','w','Units','centimeter','Position',[5 5 20 8])

    x_ext        = [bins, 3.5];
    xlbl_ext     = {'-2','-1','0','+1','+2','NS'};
    monk_pct_ext = [monk_pct(1:n_bins), 0];
    pred_pct_ext = [pred_pct(1:n_bins), pred_pct(end)];

    subplot(1,2,1);  hold on
    bar_w = 0.38;
    bar(x_ext - bar_w/2, monk_pct_ext, bar_w, ...
        'FaceColor', clr_monk, 'EdgeColor','k','LineWidth',0.8)
    bar(x_ext + bar_w/2, pred_pct_ext, bar_w, ...
        'FaceColor', clr_model, 'EdgeColor','k','LineWidth',0.8)
    xline(3.0, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1, 'HandleVisibility','off')
    legend('Monkey','Model fit','Location','northeast','Box','off')
    xticks(x_ext);  xticklabels(xlbl_ext)
    xlabel('Offset');  ylabel('Percentage of trials (%)')
    title('Performance: monkey vs model')
    set(gca,'TickDir','out','LineWidth',lw,'FontSize',fs,'Box','off')

    subplot(1,2,2);  hold on
    resid = pred_pct_ext - monk_pct_ext;
    bar(x_ext, resid, 0.7, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor','k','LineWidth',0.8)
    yline(0, '-k', 'LineWidth', 1.5)
    xline(3.0, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1)
    xticks(x_ext);  xticklabels(xlbl_ext)
    xlabel('Offset');  ylabel('\Delta% (model - monkey)')
    title('Residuals')
    set(gca,'TickDir','out','LineWidth',lw,'FontSize',fs,'Box','off')

    if fix_drift
        ttl = sprintf('%s:  v=%.3f  \\sigma_z=%.3f  (\\eta fixed=0, \\theta fixed=%.1f)', ...
            set_labels{fs_i}, x_opt(1), x_opt(2), bound);
    else
        ttl = sprintf('%s:  v=%.3f  \\eta=%.3f  \\sigma_z=%.3f  (\\theta fixed=%.1f)', ...
            set_labels{fs_i}, x_opt(1), x_opt(2), x_opt(3), bound);
    end
    sgtitle(ttl, 'FontSize', fs+1)
end

% =========================================================================
% 7.  Summary across fit sets
% =========================================================================
fprintf('=========================================================\n');
fprintf('Summary\n');
fprintf('=========================================================\n');
fprintf('%-8s %-8s', 'Set', 'N');
fprintf('%-14s', param_names{:});
fprintf('%-14s%-14s%-10s\n', 'neg-LL(fit)', 'neg-LL(eval)', 'seed SD');
for fs_i = 1 : numel(fit_perf)
    fprintf('%-8s %-8d', fit_perf(fs_i).label, fit_perf(fs_i).n_obs);
    fprintf('%-14.4f', fit_perf(fs_i).x_opt);
    fprintf('%-14.2f%-14.2f%-10.2f\n', fit_perf(fs_i).nll_opt, ...
        fit_perf(fs_i).nll_eval, fit_perf(fs_i).nll_eval_sd);
end
fprintf(['\nneg-LL(fit) is the optimiser minimum and is optimistically biased; ' ...
    'neg-LL(eval)\nis a clean re-evaluation at n = %d and is what ' ...
    'model_comparison_fig4 scores.\n\n'], n_eval);

% =========================================================================
% 8.  Save fitted parameters
%
%     fit_perf holds all three fits.  x_opt / nll_opt are kept as the
%     pooled fit at the top level so compare_rt_fit_MS.m keeps working.
% =========================================================================
results_dir = fullfile(project_dir, 'results');
if ~isfolder(results_dir), mkdir(results_dir); end

pooled  = fit_perf(strcmp({fit_perf.label}, 'pooled'));
x_opt   = pooled.x_opt;
nll_opt = pooled.nll_opt;

% A quick test must never overwrite the real fits
out_name = 'params_perf.mat';
if quick_test, out_name = 'params_perf_QUICKTEST.mat'; end

save(fullfile(results_dir, out_name), ...
    'fit_perf', 'set_labels', 'targets_monk', 'bins', 'labels', 'RT_cutoff', ...
    'x_opt', 'nll_opt', 'fix_drift', 'param_names');
fprintf('Parameters saved to %s\n', fullfile(results_dir, out_name));

% =========================================================================
% Local functions
% =========================================================================

function nll = neg_log_lik(params, fix_drift, bound, obs_counts, ...
        n_sim, n_bins, bins, t_vec, n_t, dt, noise, period)
    pred_p = simulate_model(params, fix_drift, bound, n_sim, ...
        n_bins, bins, t_vec, n_t, dt, noise, period);
    nll = -sum(obs_counts .* log(pred_p));
end


function pred_p = simulate_model(params, fix_drift, bound, n_trials, ...
        n_bins, bins, t_vec, n_t, dt, noise, period)
    % Returns predicted proportions [p(-2) p(-1) p(0) p(+1) p(+2) p(other)],
    % normalised over all 6 categories.  "other" is any outcome that is not a
    % response in -2:+2 — see the comment at the binning step below.

    drift_mean = params(1);
    if fix_drift
        drift_sigma = 0;
        sigma_z     = max(params(2), 1e-6);
    else
        drift_sigma = max(params(2), 1e-6);
        sigma_z     = max(params(3), 1e-6);
    end

    drift_vals = drift_mean + drift_sigma * randn(1, n_trials);
    dv_curr    = sigma_z * abs(randn(1, n_trials));
    stop_k     = NaN(1, n_trials);
    stopped    = false(1, n_trials);

    for i = 2 : n_t
        active = ~stopped;
        if ~any(active), break; end
        n_act = sum(active);
        dv_curr(active) = max( ...
            dv_curr(active) + drift_vals(active) .* dt + noise * sqrt(dt) * randn(1, n_act), 0);
        k_i = ceil(t_vec(i) / period);
        crossed         = active & (dv_curr >= bound);
        stop_k(crossed) = k_i;
        stopped(crossed) = true;
    end

    offsets = stop_k - 3;

    % The sixth category is everything that is not a response in -2:+2 —
    % trials that never reached the bound AND trials that stopped later than
    % +2.  Both are responses the monkey did not make (the 423 out-of-range
    % trials were excluded from the counts), so both must be charged for.
    % Counting only the never-stopped trials here would renormalise the late
    % stops away and score the diffusion model on a different support from
    % the Gaussian number lines, which route their out-of-range mass into
    % this same category.
    pred_p = zeros(1, n_bins + 1);
    for b = 1 : n_bins
        pred_p(b) = sum(offsets == bins(b));
    end
    pred_p(end) = n_trials - sum(pred_p(1 : n_bins));

    pred_p = max(pred_p, 1e-10);
    pred_p = pred_p / sum(pred_p);
end
