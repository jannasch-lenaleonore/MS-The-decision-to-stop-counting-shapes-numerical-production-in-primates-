% fit_gaussian_mle_MS.m
%
% MLE fitting of four descriptive numerosity models to monkey performance
% data, as comparison baselines for the accumulation-to-bound model fitted
% in fit_diffusion_mle_MS.m.
%
% All four place a Gaussian on a number line, either LINEAR (the curve is
% read at the produced number) or LOGARITHMIC (read at its logarithm, i.e. a
% Weber-Fechner representation), and take the probability of a response to
% be the HEIGHT of that curve at the response, normalised over the possible
% responses.  They differ in whether the number line is SHARED across target
% numerosities or refitted for each target:
%
%   'lin_shared'  ONE shared linear number line:  mu_n = n + b,
%                 one sigma in count units, so sigma_3 = sigma_4.
%                 2 free parameters (b, sigma) for both targets.
%   'log_shared'  ONE shared logarithmic number line:  mu_n = log(n) + b,
%                 one sigma in log units, so the distribution is symmetric
%                 on a logarithmic scale and its width in count units grows
%                 with n (scalar variability).
%                 2 free parameters (b, sigma) for both targets.
%   'lin_target'  linear, fitted SEPARATELY PER TARGET (mu, sigma each)
%   'log_target'  logarithmic, fitted SEPARATELY PER TARGET
%                 4 free parameters each.
%
% Parameter counts therefore run 2, 2, 4, 4 against the diffusion model's 2.
% The two shared models are matched to the diffusion model in flexibility;
% the two per-target models are strictly more flexible and can absorb any
% numerosity-specific difference.  AIC/BIC penalise that extra freedom.
%
% Why (b, sigma) and not an amplitude.  The likelihood is multinomial over
% the response categories, so the predicted probabilities are normalised
% and a free peak amplitude would scale all categories equally and cancel —
% it is not identifiable.  The two parameters are instead location and
% width, exactly the pair the diffusion model uses: drift_mean sets where
% the stopping-step distribution sits, sigma_z how wide it is.  In every
% model the proportion of correct trials is the offset-0 probability, a
% PREDICTION of those two parameters rather than a fitted quantity.
%
% Three fit sets are produced, mirroring fit_diffusion_mle_MS.m: the
% combined data of both animals ("pooled") and each animal separately.
%
% Everything else matches fit_diffusion_mle_MS.m:
%   - same trial selection (RT-valid trials, targets 3 and 4)
%   - same response distribution: multinomial over offset bins -2:+2 plus a
%     sixth "outside range" category with observed count 0
%   - same loss: nll = -sum(obs_counts .* log(pred_p))
%   - same optimiser: multi-start fminsearch with a sigmoid bound transform
%
% Unlike the diffusion model the predicted probabilities are analytic (the
% normal density at the integer responses, normalised), so no simulation is
% needed.

clear; clc;
rng(42)

% =========================================================================
% 1.  Models to fit
% =========================================================================
gauss_models = struct( ...
    'key',    {'lin_shared', 'log_shared', 'lin_target', 'log_target'}, ...
    'name',   {'Gaussian linear (shared)', 'Gaussian log (shared)', ...
               'Gaussian linear (per target)', 'Gaussian log (per target)'}, ...
    'short',  {'lin-sh', 'log-sh', 'lin-tg', 'log-tg'}, ...
    'scale',  {'linear', 'log', 'linear', 'log'}, ...
    'shared', {true, true, false, false});
n_models = numel(gauss_models);

% =========================================================================
% 2.  Response categories
% =========================================================================
bins   = -2 : 2;
n_bins = numel(bins);
labels = {'-2','-1','0','+1','+2','outside'};

% =========================================================================
% 3.  Load monkey data and build observed counts per monkey and target
% =========================================================================
targets_monk = [3, 4];
RT_cutoff    = 1100;    % exclude monkey RTs above this value (ms) --> invalid
project_dir  = fileparts(fileparts(mfilename('fullpath')));
monkeys      = {'m1', 'm2'};
n_monkeys    = numel(monkeys);
n_targets    = numel(targets_monk);

obs_counts = zeros(n_monkeys, n_targets, n_bins + 1);
n_dropped  = zeros(n_monkeys, n_targets);   % trials with |offset| > 2

for mk = 1 : n_monkeys
    S = load(fullfile(project_dir, 'data', ...
        sprintf('bhv_data_tbl_%s', monkeys{mk})));
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
            tg      = targets_monk(ti);
            tg_mask = RM(:,2) == tg;
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
                obs_counts(mk, ti, b) = obs_counts(mk, ti, b) + ...
                    sum(off_sess == bins(b));
            end
            n_dropped(mk, ti) = n_dropped(mk, ti) + sum(abs(off_sess) > 2);
        end
    end
end
% Sixth category: monkey never responds outside -2:+2 (verified below).
% Trials that do fall outside are dropped, exactly as in fit_diffusion_mle_MS.

fprintf('Observed response distributions (RT-valid trials)\n');
fprintf('%-8s %-8s %-7s', 'Monkey', 'Target', 'N');
fprintf('%-9s', labels{1:n_bins});  fprintf('%-10s\n', 'dropped');
for mk = 1 : n_monkeys
    for ti = 1 : n_targets
        oc    = squeeze(obs_counts(mk, ti, :))';
        n_obs = sum(oc);
        fprintf('%-8s %-8d %-7d', monkeys{mk}, targets_monk(ti), n_obs);
        fprintf('%-9.1f', 100 * oc(1:n_bins) / n_obs);
        fprintf('%-10d\n', n_dropped(mk, ti));
    end
end
if any(n_dropped(:) > 0)
    fprintf(['\nNote: %d trials fell outside -2:+2 and were dropped ' ...
        '(same convention as fit_diffusion_mle_MS).\n'], sum(n_dropped(:)));
end
fprintf('\n');

% Fit sets: combined data of both animals, then each animal separately.
% "pooled" pools over ANIMALS only.  Whether a fit also pools over TARGETS
% is what distinguishes the shared from the per-target models.
set_labels = {'pooled', 'm1', 'm2'};
n_sets     = numel(set_labels);
set_counts = zeros(n_sets, n_targets, n_bins + 1);
set_counts(1, :, :) = sum(obs_counts, 1);
for mk = 1 : n_monkeys
    set_counts(mk + 1, :, :) = obs_counts(mk, :, :);
end

% =========================================================================
% 4.  Optimisation settings (identical to fit_diffusion_mle_MS)
% =========================================================================
n_starts = 20;

opts = optimset( ...
    'Display',     'off', ...
    'MaxFunEvals', 600,   ...
    'MaxIter',     300,   ...
    'TolX',        1e-3,  ...
    'TolFun',      1e-3);

% =========================================================================
% 5.  Multi-start fitting per model and fit set
%
%     Shared models are fitted ONCE to both targets jointly (the two
%     targets' counts enter the same likelihood).  Per-target models are
%     fitted once per target and their neg-LLs added.
% =========================================================================
fit_gauss = struct( ...
    'key', {}, 'name', {}, 'short', {}, 'scale', {}, 'shared', {}, ...
    'label', {}, 'targets', {}, 'param_names', {}, 'x_opt', {}, ...
    'nll_opt', {}, 'nll_range', {}, 'k', {}, 'pred_p', {}, ...
    'p_correct', {}, 'obs_counts', {}, 'n_obs', {});

fprintf('Multi-start fitting: %d starts per fit...\n\n', n_starts);

tic
for mi = 1 : n_models
    M = gauss_models(mi);
    for si = 1 : n_sets
        oc_tg = reshape(set_counts(si, :, :), n_targets, n_bins + 1);
        n_obs = sum(oc_tg, 2)';

        if M.shared
            % ---- one number line for both targets: (bias, sigma) --------
            param_names = {'b', 'sigma'};
            switch M.scale
                case 'linear'
                    x0 = [0,     0.70];
                    lb = [-2.0,  0.05];
                    ub = [ 2.0,  3.00];
                case 'log'
                    x0 = [0,     0.25];
                    lb = [-0.7,  0.02];
                    ub = [ 0.7,  1.50];
            end

            obj = @(p) nll_shared(p, M.scale, targets_monk, oc_tg, bins, n_bins);
            [x_opt, nll_opt, nll_range] = multistart_fit(obj, x0, lb, ub, n_starts, opts);

            k      = numel(x0);
            pred_p = zeros(n_targets, n_bins + 1);
            for ti = 1 : n_targets
                mu = mu_shared(x_opt(1), M.scale, targets_monk(ti));
                pred_p(ti, :) = predict_gauss(mu, x_opt(2), M.scale, ...
                    targets_monk(ti), bins, n_bins);
            end

        else
            % ---- one number line per target: (mu, sigma) each -----------
            param_names = {'mu', 'sigma'};
            x_opt     = NaN(n_targets, 2);
            pred_p    = zeros(n_targets, n_bins + 1);
            nll_opt   = 0;
            nll_range = [0 0];

            for ti = 1 : n_targets
                tg = targets_monk(ti);
                % Bounds bracket the target on the relevant scale
                switch M.scale
                    case 'linear'
                        x0 = [tg,      0.70];
                        lb = [tg - 2,  0.05];
                        ub = [tg + 2,  3.00];
                    case 'log'
                        x0 = [log(tg),       0.25];
                        lb = [log(tg) - 0.7, 0.02];
                        ub = [log(tg) + 0.7, 1.50];
                end

                obj = @(p) nll_single(p, M.scale, tg, oc_tg(ti,:), bins, n_bins);
                [x_ti, nll_ti, rng_ti] = multistart_fit(obj, x0, lb, ub, n_starts, opts);

                x_opt(ti, :)  = x_ti;
                pred_p(ti, :) = predict_gauss(x_ti(1), x_ti(2), M.scale, tg, bins, n_bins);
                nll_opt       = nll_opt + nll_ti;
                nll_range     = nll_range + rng_ti;
            end
            k = 2 * n_targets;
        end

        p_correct = pred_p(:, bins == 0)';

        fprintf('  %-28s  %-6s  neg-LL = %9.2f  k = %d  [%s]\n', ...
            M.name, set_labels{si}, nll_opt, k, ...
            strjoin(arrayfun(@(v) sprintf('%.3f', v), x_opt(:)', ...
            'UniformOutput', false), '  '));
        fprintf('%s p(correct): ', repmat(' ', 1, 40));
        fprintf('target %d = %.3f   ', [targets_monk; p_correct]);
        fprintf('\n');

        fit_gauss(mi, si) = struct( ...
            'key',         M.key, ...
            'name',        M.name, ...
            'short',       M.short, ...
            'scale',       M.scale, ...
            'shared',      M.shared, ...
            'label',       set_labels{si}, ...
            'targets',     targets_monk, ...
            'param_names', {param_names}, ...
            'x_opt',       x_opt, ...
            'nll_opt',     nll_opt, ...
            'nll_range',   nll_range, ...
            'k',           k, ...
            'pred_p',      pred_p, ...
            'p_correct',   p_correct, ...
            'obs_counts',  oc_tg, ...
            'n_obs',       n_obs);
    end
    fprintf('\n');
end
t_fit = toc;
fprintf('Fitting finished in %.0f s.\n\n', t_fit);

% =========================================================================
% 6.  Observed vs fitted, per fit set and target
% =========================================================================
for si = 1 : n_sets
    for ti = 1 : n_targets
        oc    = reshape(set_counts(si, ti, :), 1, []);
        n_obs = sum(oc);
        fprintf('--- %s, target %d (N = %d) ---\n', set_labels{si}, targets_monk(ti), n_obs);
        fprintf('%-30s', 'Category:');  fprintf('%-9s', labels{:});  fprintf('\n');
        fprintf('%-30s', 'Monkey %:');  fprintf('%-9.1f', 100 * oc / n_obs);  fprintf('\n');
        for mi = 1 : n_models
            fprintf('%-30s', sprintf('%s %%:', gauss_models(mi).name));
            fprintf('%-9.1f', 100 * fit_gauss(mi, si).pred_p(ti, :));  fprintf('\n');
        end
        fprintf('\n');
    end
end

% =========================================================================
% 7.  Overview figure: observed vs all four Gaussian fits
% =========================================================================
lw = 1.5;  fs = 11;
clr_model = {[0.20 0.20 0.20], [0.20 0.20 0.20], [0.62 0.62 0.62], [0.62 0.62 0.62]};
sty_model = {':', '-.', ':', '-.'};
mrk_model = {'^', 'd', '^', 'd'};

figure('Color','w','Units','centimeter','Position',[5 5 22 19])
p = 0;
for si = 1 : n_sets
    for ti = 1 : n_targets
        p = p + 1;
        subplot(n_sets, n_targets, p);  hold on

        oc    = reshape(set_counts(si, ti, :), 1, []);
        n_obs = sum(oc);

        plot(bins, 100 * oc(1:n_bins) / n_obs, 'o-', 'Color', 'k', ...
            'MarkerFaceColor', 'k', 'MarkerSize', 6, 'LineWidth', lw, ...
            'DisplayName', 'Monkey')
        for mi = 1 : n_models
            plot(bins, 100 * fit_gauss(mi, si).pred_p(ti, 1:n_bins), ...
                [mrk_model{mi} sty_model{mi}], 'Color', clr_model{mi}, ...
                'MarkerFaceColor', 'w', 'MarkerSize', 5, 'LineWidth', lw, ...
                'DisplayName', gauss_models(mi).name)
        end

        xlim([-2.5 2.5])
        xticks(bins);  xticklabels({'-2','-1','0','+1','+2'})
        xlabel('Response offset', 'FontSize', fs)
        if ti == 1, ylabel('Trials (%)', 'FontSize', fs); end
        title(sprintf('%s — target %d  (N = %d)', set_labels{si}, targets_monk(ti), n_obs), ...
            'FontSize', fs)
        if p == 1
            legend('Location','northwest','Box','off','FontSize', fs-3)
        end
        set(gca, 'TickDir','out', 'LineWidth', lw, 'FontSize', fs, 'Box','off')
    end
end
sgtitle('Gaussian model fits per fit set and target numerosity', 'FontSize', fs+1)

% =========================================================================
% 8.  Save fitted parameters
% =========================================================================
results_dir = fullfile(project_dir, 'results');
if ~isfolder(results_dir), mkdir(results_dir); end
save(fullfile(results_dir, 'params_gauss.mat'), ...
    'fit_gauss', 'gauss_models', 'monkeys', 'set_labels', 'set_counts', ...
    'targets_monk', 'bins', 'labels', 'RT_cutoff', 'obs_counts', 'n_dropped');
fprintf('Parameters saved to %s\n', fullfile(results_dir, 'params_gauss.mat'));

% =========================================================================
% Local functions
% =========================================================================

function [x_opt, nll_opt, nll_range] = multistart_fit(obj, x0, lb, ub, n_starts, opts)
    % Multi-start fminsearch with a sigmoid transform enforcing the bounds:
    %   raw -> bounded:  p = lb + (ub-lb) ./ (1+exp(-r))
    %   bounded -> raw:  r = log((p-lb) ./ (ub-p))
    to_raw    = @(p) log((p - lb) ./ (ub - p));
    to_params = @(r) lb + (ub - lb) ./ (1 + exp(-r));

    rng(42)
    starts = [x0; lb + (ub - lb) .* rand(n_starts - 1, numel(x0))];

    best_nll = Inf;
    best_r   = to_raw(x0);
    all_nll  = NaN(n_starts, 1);

    for s = 1 : n_starts
        [r_s, nll_s] = fminsearch(@(r) obj(to_params(r)), to_raw(starts(s,:)), opts);
        all_nll(s)   = nll_s;
        if nll_s < best_nll
            best_nll = nll_s;
            best_r   = r_s;
        end
    end

    x_opt     = to_params(best_r);
    nll_opt   = best_nll;
    nll_range = [min(all_nll), max(all_nll)];
end


function mu = mu_shared(b, scale, target)
    % Location of target `target` on the shared number line.  b is the bias
    % in the units of that number line: additive in counts on the linear
    % scale, multiplicative (a gain of exp(b)) on the logarithmic scale.
    switch scale
        case 'linear', mu = target + b;
        case 'log',    mu = log(target) + b;
        otherwise,     error('Unknown scale "%s".', scale);
    end
end


function nll = nll_shared(params, scale, targets, obs_counts_tg, bins, n_bins)
    % One (b, sigma) pair scored against BOTH targets' counts.
    nll = 0;
    for ti = 1 : numel(targets)
        mu     = mu_shared(params(1), scale, targets(ti));
        pred_p = predict_gauss(mu, params(2), scale, targets(ti), bins, n_bins);
        nll    = nll - sum(obs_counts_tg(ti, :) .* log(pred_p));
    end
end


function nll = nll_single(params, scale, target, obs_counts, bins, n_bins)
    pred_p = predict_gauss(params(1), params(2), scale, target, bins, n_bins);
    nll = -sum(obs_counts .* log(pred_p));
end


function pred_p = predict_gauss(mu, sigma, scale, target, bins, n_bins)
    % Returns predicted proportions [p(-2) ... p(+2) p(outside)],
    % normalised over all 6 categories.
    %
    % The probability of producing r is the HEIGHT of the Gaussian at r —
    % at r itself on a linear number line, at log(r) on a logarithmic one —
    % normalised over the candidate responses.  This is the reading standard
    % in numerical cognition: the curve IS the response distribution, so a
    % curve symmetric on its own scale gives symmetric adjacent responses.
    %
    % The alternative reading integrates the curve between the half-integer
    % edges around r.  That carries a Jacobian which on a logarithmic scale
    % down-weights larger r by roughly 1/r, and it fitted these data clearly
    % worse (the shared log line lost ~1400 AIC at equal parameter count),
    % so it is not used anywhere in this project.
    %
    % R_MAX bounds the candidate set.  Responses below 1 are excluded, the
    % logarithm being undefined there; at any width the optimiser visits the
    % mass beyond R_MAX is negligible, so the bound itself does not matter.
    R_MAX = 40;

    sigma = max(sigma, 1e-6);
    r     = 1 : R_MAX;

    switch scale
        case 'linear', x = r;
        case 'log',    x = log(r);
        otherwise,     error('Unknown scale "%s".', scale);
    end

    dens = exp(-0.5 * ((x - mu) / sigma) .^ 2);   % constant factor cancels
    if sum(dens) > 0
        dens = dens / sum(dens);
    else
        % sigma so small that every height underflowed: all mass on the
        % nearest candidate response.
        [~, i_near] = min(abs(x - mu));
        dens = zeros(1, R_MAX);  dens(i_near) = 1;
    end

    resp   = target + bins;              % produced numerosities for bins -2:+2
    in_set = resp >= 1 & resp <= R_MAX;

    pred_p           = zeros(1, n_bins + 1);
    pred_p(in_set)   = dens(resp(in_set));
    pred_p(end)      = max(1 - sum(pred_p(1:n_bins)), 0);   % outside -2:+2

    pred_p = max(pred_p, 1e-10);
    pred_p = pred_p / sum(pred_p);
end
