% cv_models_MS.m
%
% Leave-one-session-out cross-validation of the five models on the combined
% data of both animals (targets 3 and 4).
%
%   accumulation to bound         2 parameters, shared across targets
%   Gaussian linear (shared)      2 parameters, one shared linear number line
%   Gaussian log (shared)         2 parameters, one shared log number line
%   Gaussian linear (per target)  2 parameters per target (4 total)
%   Gaussian log (per target)     2 parameters per target (4 total)
%
% Why sessions: trials within a session are correlated (see
% session_variability_MS.m — the response distribution is overdispersed by a
% factor of ~4 relative to independent trials), so a trial-level split would
% leak information between training and test and reproduce the same
% over-optimism as AIC/BIC computed on 16 537 "independent" trials.  Whole
% sessions are therefore held out, and the session is also the unit the
% statistics are reported over.
%
% Why one session at a time rather than k folds.  A k-fold split has to
% choose an arbitrary partition, which then has to be averaged away over
% repeated shuffles — and those repetitions re-cut the SAME sessions, so
% they inflate the apparent number of observations without adding data.
% Leave-one-session-out has no partition to choose: every session is held
% out exactly once, the result is deterministic, and the n of the paired
% test is exactly the number of sessions.  With 76 sessions it is also
% cheaper than 10 folds x 10 repetitions (76 refits instead of 100).
%
% Scoring.  Each trial contributes one number: the log of the probability
% the model assigned to the response category that actually occurred, with
% parameters estimated WITHOUT that trial's session.  Summed over the held
% out session this is that session's held-out log-likelihood; divided by its
% trial count it is the mean per-trial log-likelihood, whose exponential is
% the geometric-mean probability assigned to the observed response.
% Normalising per session is what lets long and short sessions carry equal
% weight in the paired test.  No AIC/BIC penalty is applied — out-of-sample
% evaluation already charges for flexibility.
%
% Run fit_diffusion_mle_MS.m first: the full-data optimum is used to warm
% start the per-fold diffusion fits.
%
% Saves results/cv_results.mat.

clear; clc;

% =========================================================================
% 1.  Settings
% =========================================================================
quick_test = false;   % true: tiny run to check the script end to end (~1 min)

% Leave-one-session-out: one fold per session, and nothing to repeat, since
% the partition is not random.  n_folds is set to the session count once the
% data are loaded (section 5); n_reps stays at 1 so the stored arrays keep
% the [model x rep x fold] shape the downstream scripts expect.
n_reps = 1;
rng(42)

% Per-fold diffusion refits are warm started from the full-data optimum, so
% far fewer starts are needed than for the cold multi-start full fit.  A
% cold 20-start fit takes ~25 min; at ~1.8 min per refit the 76 sessions
% come to roughly 2.3 h.
n_starts_cv   = 2;
max_fun_evals = 300;
n_sim         = 20000;    % per likelihood evaluation during fitting
n_eval        = 500000;   % for the held-out prediction — matches the
                          % full-data fit in fit_diffusion_mle_MS.m

max_folds = Inf;          % quick test only: stop after this many sessions

if quick_test
    max_folds = 3;  n_starts_cv = 1;
    max_fun_evals = 60;  n_sim = 4000;  n_eval = 20000;
    fprintf('*** QUICK TEST — results are not usable ***\n\n');
end

% =========================================================================
% 2.  Fixed model parameters — must match fit_diffusion_mle_MS exactly
% =========================================================================
noise    = 0.1;
dt       = 0.01;
step_dur = 1;
bound    = 1;

n_steps_rel = 7;
period      = step_dur;
t_vec       = (0 : dt : n_steps_rel * period)';
n_t         = numel(t_vec);

targets_monk = [3, 4];
RT_cutoff    = 1100;
bins         = -2 : 2;
n_bins       = numel(bins);
n_targets    = numel(targets_monk);

% Model 1 is the diffusion model; models 2:5 are the Gaussian variants and
% index gauss_spec(mi-1).  Definitions must match fit_gaussian_mle_MS.m.
model_names = {'Accumulation to bound', 'Gaussian linear (shared)', ...
    'Gaussian log (shared)', 'Gaussian linear (per target)', ...
    'Gaussian log (per target)'};
gauss_spec  = struct( ...
    'scale',  {'linear', 'log', 'linear', 'log'}, ...
    'shared', {true,     true,  false,    false});
n_models    = numel(model_names);

project_dir = fileparts(fileparts(mfilename('fullpath')));
monkeys     = {'m1', 'm2'};

% =========================================================================
% 3.  Observed counts per session and target
%
%     Trial selection identical to fit_diffusion_mle_MS / fit_gaussian_mle_MS.
%     Counts per session are enough: any fold's counts are just the sum over
%     the sessions it contains.
% =========================================================================
sess_counts = [];    % [n_sessions x n_targets x n_bins+1]
sess_animal = [];    % animal index per session
n_sess      = 0;

for mk = 1 : numel(monkeys)
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

        n_sess = n_sess + 1;
        sess_animal(n_sess) = mk; %#ok<SAGROW>
        sess_counts(n_sess, 1:n_targets, 1:n_bins+1) = 0; %#ok<SAGROW>

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
                sess_counts(n_sess, ti, b) = sum(off_sess == bins(b));
            end
        end
    end
end

n_per_sess = sum(sess_counts(:, :), 2);
fprintf('Loaded %d sessions (m1: %d, m2: %d), %d trials total.\n', ...
    n_sess, sum(sess_animal == 1), sum(sess_animal == 2), sum(n_per_sess));

has_both = all(squeeze(sum(sess_counts, 3)) > 0, 2);
fprintf('Sessions containing both targets: %d of %d.\n\n', sum(has_both), n_sess);

% =========================================================================
% 4.  Warm-start values and optimiser setup
% =========================================================================
perf = load(fullfile(project_dir, 'results', 'params_perf.mat'));
pooled_fit = perf.fit_perf(strcmp({perf.fit_perf.label}, 'pooled'));
fix_drift  = perf.fix_drift;

if fix_drift
    lb_d = [0.1,  0.01];
    ub_d = [1.0,  1.00];
else
    lb_d = [0.25, 0.01, 0.01];
    ub_d = [0.5,  2.00, 1.00];
end
x0_d = pooled_fit.x_opt;
fprintf('Warm start for the diffusion refits: [%s] (full-data pooled optimum)\n\n', ...
    strjoin(arrayfun(@(v) sprintf('%.4f', v), x0_d, 'UniformOutput', false), '  '));

opts = optimset('Display', 'off', 'MaxFunEvals', max_fun_evals, ...
    'MaxIter', max_fun_evals, 'TolX', 1e-3, 'TolFun', 1e-3);

% =========================================================================
% 5.  Fold assignment — one fold per session
%
%     Fold k IS session k.  There is nothing to shuffle and nothing to
%     average over, so the whole procedure is deterministic and does not
%     depend on the seed.
% =========================================================================
n_folds = n_sess;
fold_of = 1 : n_sess;             % [n_reps(=1) x n_sess]

cv_n = NaN(n_reps, n_folds);      % held-out trials per fold
for k = 1 : n_folds
    held      = sess_counts(fold_of(1, :) == k, :, :);
    cv_n(1,k) = sum(held(:));
end

fprintf(['Leave-one-session-out: %d folds, one per session ' ...
    '(%d - %d trials held out, median %d).\n\n'], n_folds, ...
    min(cv_n(:)), max(cv_n(:)), round(median(cv_n(:))));

% =========================================================================
% 6.  Reuse of the diffusion refits
%
%     The diffusion refits dominate the runtime and do not depend on which
%     Gaussian variants are scored alongside them, so a fold whose sessions
%     were split exactly the same way in an earlier run can reuse its
%     diffusion fit verbatim.
%
%     Matching is done PER REPETITION, not on the whole fold_of array.  The
%     repetitions are drawn one after another from a stream seeded with
%     rng(42), so raising n_reps leaves the earlier repetitions byte
%     identical and only appends new ones: going from 5 to 10 repetitions
%     reuses all 50 existing refits and fits only the 50 new ones.  A
%     repetition is reused only if its session split AND its held-out trial
%     counts both match; anything else is refitted.
% =========================================================================
ddm_cache_ll     = NaN(n_reps, n_folds);
ddm_cache_params = cell(n_reps, n_folds);

cache_n_eval = NaN;    % n_eval the cached scores were produced at, if any

cache_file = fullfile(project_dir, 'results', 'cv_results.mat');
if ~quick_test && isfile(cache_file)
    C = load(cache_file);
    fields_ok = all(isfield(C, {'cv_ll', 'cv_n', 'cv_params', 'fold_of', ...
        'model_names', 'n_starts_cv', 'n_sim', 'n_eval', 'RT_cutoff', 'targets_monk'}));
    % n_eval is deliberately NOT part of this test.  What the cache supplies
    % is the fitted PARAMETERS, which depend only on the fitting settings;
    % the held-out prediction is re-simulated from them below at the current
    % n_eval regardless.  Raising n_eval therefore costs one simulation per
    % fold rather than a full refit.
    settings_ok = fields_ok && ...
        isequal(size(C.cv_ll, 3), n_folds)   && ...
        isequal(C.n_starts_cv, n_starts_cv)  && isequal(C.n_sim, n_sim)      && ...
        isequal(C.RT_cutoff, RT_cutoff)      && ...
        isequal(C.targets_monk, targets_monk) && ...
        strcmp(C.model_names{1}, model_names{1});

    if settings_ok
        cache_n_eval = C.n_eval;
        n_reused = 0;
        for rep = 1 : n_reps
            % Find a stored repetition with an identical session split.
            src = find(arrayfun(@(r) isequal(C.fold_of(r,:), fold_of(rep,:)), ...
                1 : size(C.fold_of, 1)), 1);
            if isempty(src) || ~isequal(C.cv_n(src,:), cv_n(rep,:))
                continue
            end
            ddm_cache_ll(rep, :)     = reshape(C.cv_ll(1, src, :), 1, n_folds);
            ddm_cache_params(rep, :) = reshape(C.cv_params(1, src, :), 1, n_folds);
            n_reused = n_reused + sum(~isnan(ddm_cache_ll(rep, :)));
        end
        fprintf(['Reusing %d of %d cached diffusion refits from cv_results.mat\n' ...
            '(matched per repetition on session split and held-out trials).\n'], ...
            n_reused, n_reps * n_folds);
        if cache_n_eval ~= n_eval
            fprintf(['The cache was scored at n_eval = %d and is being re-scored ' ...
                'at %d;\nthe parameters are reused, the held-out predictions are ' ...
                'regenerated.\n'], cache_n_eval, n_eval);
        end
        fprintf('\n');
    else
        fprintf(['cv_results.mat does not match the current settings — ' ...
            'the diffusion model\nwill be refitted from scratch.\n\n']);
    end
end

% =========================================================================
% 7.  Cross-validation
% =========================================================================
cv_ll     = NaN(n_models, n_reps, n_folds);   % held-out log-likelihood
cv_params = cell(n_models, n_reps, n_folds);

% The same held-out predictions, scored at two finer resolutions.  Neither
% costs an extra fit — they are the same numbers summed differently.
%   cv_ll_tg    splits a fold's score by target numerosity, which separates
%               models that predict the two targets alike (the diffusion
%               model, the shared linear number line) from those that do not.
%   cv_ll_sess  gives one score per held-out session.  A session is held out
%               exactly once per repetition, so this is the natural unit for
%               a paired test — unlike folds, whose training sets overlap.
cv_ll_tg   = NaN(n_models, n_reps, n_folds, n_targets);
cv_ll_sess = NaN(n_models, n_reps, n_sess);

% Checkpointing: every completed fold is written out, and a re-run skips
% folds already present.  Delete the partial file to force a clean run.
partial_file = fullfile(project_dir, 'results', 'cv_partial.mat');
if quick_test
    partial_file = fullfile(project_dir, 'results', 'cv_partial_QUICKTEST.mat');
end
if isfile(partial_file)
    P = load(partial_file);
    if isequal(size(P.cv_ll), size(cv_ll)) && isequal(P.fold_of, fold_of) && ...
            isfield(P, 'cv_ll_sess') && isequal(size(P.cv_ll_sess), size(cv_ll_sess))
        cv_ll      = P.cv_ll;
        cv_params  = P.cv_params;
        cv_ll_tg   = P.cv_ll_tg;
        cv_ll_sess = P.cv_ll_sess;
        fprintf('Resuming: %d of %d folds already done.\n', ...
            sum(all(~isnan(cv_ll(:, :)), 1)), n_reps * n_folds);
    else
        fprintf('Ignoring %s — it does not match the current settings.\n', partial_file);
    end
end

fprintf('Leave-one-session-out CV over %d sessions\n\n', n_folds);

t_start = tic;
for rep = 1 : n_reps
    for k = 1 : min(n_folds, max_folds)
        if all(~isnan(cv_ll(:, rep, k)))
            continue    % already computed in an earlier run
        end
        test_sess  = fold_of(rep, :) == k;
        train_sess = ~test_sess;

        train_tg = squeeze(sum(sess_counts(train_sess, :, :), 1));
        test_tg  = squeeze(sum(sess_counts(test_sess,  :, :), 1));
        train_po = sum(train_tg, 1);
        test_po  = sum(test_tg,  1);

        % Every model's held-out prediction, one row per target.
        pred_fold = NaN(n_models, n_targets, n_bins + 1);

        % ---- Accumulation to bound: one fit, shared across targets -------
        if ~isnan(ddm_cache_ll(rep, k))
            best_x  = ddm_cache_params{rep, k};
            ddm_tag = 'cached';
        else
            obj_ddm = @(p) ddm_neg_log_lik(p, fix_drift, bound, ...
                train_po, n_sim, n_bins, bins, t_vec, n_t, dt, noise, period);

            starts_d = x0_d;
            for s = 2 : n_starts_cv   % jittered restarts around the warm start
                starts_d(s, :) = min(max(x0_d .* (1 + 0.15 * randn(size(x0_d))), ...
                    lb_d + 1e-3), ub_d - 1e-3);
            end

            best_nll = Inf;  best_x = x0_d;
            for s = 1 : n_starts_cv
                [x_s, nll_s] = bounded_fminsearch(obj_ddm, starts_d(s,:), lb_d, ub_d, opts);
                if nll_s < best_nll
                    best_nll = nll_s;
                    best_x   = x_s;
                end
            end
            ddm_tag = 'fitted';
        end

        % The prediction is regenerated from the fitted parameters under the
        % fold's fixed seed, so a cached fold reproduces its original
        % prediction exactly.  That costs one simulation per fold; what the
        % cache saves is the hundreds of simulations inside the fit itself.
        rng(5000 + k)
        pred_ddm = ddm_simulate(best_x, fix_drift, bound, n_eval, ...
            n_bins, bins, t_vec, n_t, dt, noise, period);
        cv_params{1, rep, k} = best_x;
        for ti = 1 : n_targets
            pred_fold(1, ti, :) = pred_ddm;      % target-invariant
        end

        % Only meaningful when the cache was scored at the same n_eval; with a
        % different n_eval the prediction is a different draw by construction.
        if strcmp(ddm_tag, 'cached') && cache_n_eval == n_eval
            ll_check = sum(test_po .* log(pred_ddm));
            if abs(ll_check - ddm_cache_ll(rep, k)) > 1e-6
                error(['Cached diffusion fit for rep %d fold %d does not ' ...
                    'reproduce (%.6f vs %.6f). Delete results/cv_results.mat ' ...
                    'and refit.'], rep, k, ll_check, ddm_cache_ll(rep, k));
            end
        end

        % ---- Gaussians ---------------------------------------------------
        for mi = 2 : n_models
            G = gauss_spec(mi - 1);

            if G.shared
                % One number line for both targets: (bias, sigma)
                switch G.scale
                    case 'linear'
                        x0_g = [0,     0.70];
                        lb_g = [-2.0,  0.05];
                        ub_g = [ 2.0,  3.00];
                    case 'log'
                        x0_g = [0,     0.25];
                        lb_g = [-0.7,  0.02];
                        ub_g = [ 0.7,  1.50];
                end

                obj_g = @(p) gauss_nll_shared(p, G.scale, targets_monk, ...
                    train_tg, bins, n_bins);
                x_g = bounded_fminsearch(obj_g, x0_g, lb_g, ub_g, opts);

                for ti = 1 : n_targets
                    mu = mu_shared(x_g(1), G.scale, targets_monk(ti));
                    pred_fold(mi, ti, :) = gauss_predict(mu, x_g(2), G.scale, ...
                        targets_monk(ti), bins, n_bins);
                end
                x_store = x_g;

            else
                % One number line per target: (mu, sigma) each
                x_store = NaN(n_targets, 2);
                for ti = 1 : n_targets
                    tg = targets_monk(ti);
                    switch G.scale
                        case 'linear'
                            x0_g = [tg,      0.70];
                            lb_g = [tg - 2,  0.05];
                            ub_g = [tg + 2,  3.00];
                        case 'log'
                            x0_g = [log(tg),       0.25];
                            lb_g = [log(tg) - 0.7, 0.02];
                            ub_g = [log(tg) + 0.7, 1.50];
                    end

                    obj_g = @(p) gauss_nll_single(p, G.scale, tg, ...
                        train_tg(ti,:), bins, n_bins);
                    x_g = bounded_fminsearch(obj_g, x0_g, lb_g, ub_g, opts);
                    pred_fold(mi, ti, :) = gauss_predict(x_g(1), x_g(2), ...
                        G.scale, tg, bins, n_bins);
                    x_store(ti,:) = x_g;
                end
            end

            cv_params{mi, rep, k} = x_store;
        end

        % ---- Score every model on the held-out sessions -------------------
        % One set of predictions, summed three ways: over the whole fold, per
        % target, and per held-out session.
        test_idx = find(test_sess);
        for mi = 1 : n_models
            pt = reshape(pred_fold(mi, :, :), n_targets, n_bins + 1);
            for ti = 1 : n_targets
                cv_ll_tg(mi, rep, k, ti) = sum(test_tg(ti,:) .* log(pt(ti,:)));
            end
            cv_ll(mi, rep, k) = sum(cv_ll_tg(mi, rep, k, :));
            for s = test_idx
                sc = reshape(sess_counts(s, :, :), n_targets, n_bins + 1);
                cv_ll_sess(mi, rep, s) = sum(sum(sc .* log(pt)));
            end
        end

        save(partial_file, 'cv_ll', 'cv_ll_tg', 'cv_ll_sess', 'cv_n', ...
            'cv_params', 'fold_of');

        fprintf('  session %2d/%d (%s, n_test = %4d, DDM %s, %.0f min):', ...
            k, n_folds, monkeys{sess_animal(k)}, cv_n(rep, k), ddm_tag, ...
            toc(t_start) / 60);
        fprintf('  %s = %9.2f', ...
            'DDM',    cv_ll(1, rep, k), 'lin-sh', cv_ll(2, rep, k), ...
            'log-sh', cv_ll(3, rep, k), 'lin-tg', cv_ll(4, rep, k), ...
            'log-tg', cv_ll(5, rep, k));
        fprintf('\n');
    end
end
t_cv = toc(t_start);
fprintf('\nCross-validation finished in %.1f min.\n\n', t_cv / 60);

% =========================================================================
% 8.  Summary
% =========================================================================
% Two currencies.  The POOLED value sums every held-out trial and divides by
% their total, so it answers "how well is a trial predicted"; it is the
% number the ceiling and floor benchmarks live on.  The SESSION value takes
% each session's own per-trial mean first, so every session counts once; it
% is the one the statistics are reported over, and the only one that has a
% variance across sessions to quote.
% A run stopped early (or a quick test) leaves later folds unscored; the
% summary is restricted to the sessions that were actually held out.
done     = reshape(~isnan(cv_ll(1, 1, :)), 1, []);
n_done   = sum(done);
ll_tot   = reshape(sum(cv_ll(:, 1, done), 3), 1, []);   % total held-out LL
ll_pool  = ll_tot / sum(cv_n(1, done));                 % per trial, trial-weighted
ll_sess  = squeeze(cv_ll_sess(:, 1, :)) ./ n_per_sess'; % [model x session]
ll_sess  = ll_sess(:, done);
sess_an  = sess_animal(done);

% Signed-rank over the paired session differences, guarded for short runs.
paired_p = @(d) fun_paired_p(d);

if n_done < n_folds
    fprintf('*** only %d of %d sessions scored — summary is partial ***\n\n', ...
        n_done, n_folds);
end

fprintf('Held-out log-likelihood per trial (%d sessions, each held out once)\n', n_done);
fprintf('%-30s %12s %12s %12s %14s\n', ...
    'Model', 'pooled', 'sess. mean', 'sess. SD', 'geom. mean p');
fprintf('%s\n', repmat('-', 1, 84));
for mi = 1 : n_models
    fprintf('%-30s %12.4f %12.4f %12.4f %14.4f\n', model_names{mi}, ...
        ll_pool(mi), mean(ll_sess(mi, :)), std(ll_sess(mi, :)), exp(ll_pool(mi)));
end
fprintf(['\nThe SD column is variability BETWEEN sessions, not uncertainty in ' ...
    'the mean:\nsessions differ in how predictable they are, and all models ' ...
    'follow that\nvariation together, which is why they are compared within ' ...
    'session below.\n\n']);

% Paired per-session comparison against the diffusion model.  Each session is
% held out exactly once, so these are n_sess genuinely paired observations —
% no repetition dimension to inflate them.
fprintf('Paired comparison against the accumulation-to-bound model\n');
fprintf('(one held-out session per pair, %d sessions):\n', n_done);
for mi = 2 : n_models
    d = ll_sess(1, :) - ll_sess(mi, :);   % > 0 : diffusion model predicts better
    fprintf(['  vs %-30s DDM better in %2d/%2d sessions, ' ...
        'mean dLL/trial = %+.5f (SD %.5f), p = %.3g\n'], ...
        model_names{mi}, sum(d > 0), numel(d), mean(d), std(d), paired_p(d));
end
fprintf('\n');

% ---- Finer resolution 1: where does the advantage come from? -------------
% Folds pool the two target numerosities.  Splitting them apart shows whether
% a model's deficit is specific to one target, which is exactly the
% distinction between a shared linear and a shared logarithmic number line.
fprintf('Held-out log-likelihood per trial, split by target numerosity\n');
n_tg = squeeze(sum(sum(sess_counts(:, :, :), 3), 1))';   % trials per target
fprintf('%-30s', 'Model');
for ti = 1 : n_targets, fprintf('%14s', sprintf('target %d', targets_monk(ti))); end
fprintf('%14s\n', 'difference');
fprintf('%s\n', repmat('-', 1, 30 + 14 * (n_targets + 1)));
for mi = 1 : n_models
    per_tg = NaN(1, n_targets);
    for ti = 1 : n_targets
        per_tg(ti) = sum(cv_ll_tg(mi, 1, done, ti)) / ...
            sum(sum(sess_counts(done, ti, :), 3), 1);
    end
    fprintf('%-30s', model_names{mi});
    fprintf('%14.4f', per_tg);
    fprintf('%14.4f\n', per_tg(2) - per_tg(1));
end
fprintf(['\nA model that treats both targets alike shows the same pattern in ' ...
    'both columns;\nthe difference column isolates any target-specific ' ...
    'deficit.\n\n']);

% ---- Finer resolution 2: does the same ranking hold in each animal? ------
% The models are fitted once on both animals together.  If that single fit
% serves them equally, the paired session-level difference should be positive
% in both, and the between-session SD should be of the same order as the
% between-animal difference in level.
fprintf('Per-animal breakdown of the same held-out sessions\n');
for mk = 1 : numel(monkeys)
    sel = sess_an(:)' == mk;
    if ~any(sel), continue, end
    fprintf('  %s (%d sessions)  stopping model %.4f +/- %.4f per trial\n', ...
        monkeys{mk}, sum(sel), mean(ll_sess(1, sel)), std(ll_sess(1, sel)));
    for mi = 2 : n_models
        d = ll_sess(1, sel) - ll_sess(mi, sel);
        fprintf('      vs %-30s %2d/%2d, mean dLL/trial = %+.5f, p = %.3g\n', ...
            model_names{mi}, sum(d > 0), numel(d), mean(d), paired_p(d));
    end
end
if all(ismember([1 2], sess_an))
    fprintf(['\nBetween-animal difference in level: %.4f, against a between-session ' ...
        'SD of %.4f.\n\n'], ...
        abs(mean(ll_sess(1, sess_an == 1)) - mean(ll_sess(1, sess_an == 2))), ...
        std(ll_sess(1, :)));
end

% =========================================================================
% 9.  Save
% =========================================================================
results_dir = fullfile(project_dir, 'results');
if ~isfolder(results_dir), mkdir(results_dir); end

% A quick test must never overwrite the real results
out_name = 'cv_results.mat';
if quick_test, out_name = 'cv_results_QUICKTEST.mat'; end

save(fullfile(results_dir, out_name), ...
    'cv_ll', 'cv_ll_tg', 'cv_ll_sess', 'cv_n', 'cv_params', 'fold_of', ...
    'sess_animal', 'sess_counts', 'n_per_sess', ...
    'model_names', 'gauss_spec', 'n_folds', 'n_reps', 'targets_monk', 'bins', ...
    'n_starts_cv', 'n_sim', 'n_eval', 'RT_cutoff');
fprintf('CV results saved to %s\n', fullfile(results_dir, out_name));

% =========================================================================
% Local functions — identical model code to the two fitter scripts
% =========================================================================

function [x_opt, nll_opt] = bounded_fminsearch(obj, x0, lb, ub, opts)
    % fminsearch with a sigmoid transform enforcing the bounds:
    %   raw -> bounded:  p = lb + (ub-lb) ./ (1+exp(-r))
    %   bounded -> raw:  r = log((p-lb) ./ (ub-p))
    to_raw    = @(p) log((p - lb) ./ (ub - p));
    to_params = @(r) lb + (ub - lb) ./ (1 + exp(-r));
    [r_opt, nll_opt] = fminsearch(@(r) obj(to_params(r)), to_raw(x0), opts);
    x_opt = to_params(r_opt);
end


function nll = ddm_neg_log_lik(params, fix_drift, bound, obs_counts, ...
        n_sim, n_bins, bins, t_vec, n_t, dt, noise, period)
    pred_p = ddm_simulate(params, fix_drift, bound, n_sim, ...
        n_bins, bins, t_vec, n_t, dt, noise, period);
    nll = -sum(obs_counts .* log(pred_p));
end


function pred_p = ddm_simulate(params, fix_drift, bound, n_trials, ...
        n_bins, bins, t_vec, n_t, dt, noise, period)
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
        k_i              = ceil(t_vec(i) / period);
        crossed          = active & (dv_curr >= bound);
        stop_k(crossed)  = k_i;
        stopped(crossed) = true;
    end

    offsets = stop_k - 3;

    % Sixth category = everything that is not a response in -2:+2, i.e. never
    % stopped OR stopped later than +2.  Must match simulate_model in
    % fit_diffusion_mle_MS.m — see the comment there.
    pred_p = zeros(1, n_bins + 1);
    for b = 1 : n_bins
        pred_p(b) = sum(offsets == bins(b));
    end
    pred_p(end) = n_trials - sum(pred_p(1 : n_bins));

    pred_p = max(pred_p, 1e-10);
    pred_p = pred_p / sum(pred_p);
end


function mu = mu_shared(b, scale, target)
    % Location of `target` on the shared number line.  b is the bias in the
    % units of that number line: additive in counts on the linear scale,
    % multiplicative (a gain of exp(b)) on the logarithmic scale.
    switch scale
        case 'linear', mu = target + b;
        case 'log',    mu = log(target) + b;
        otherwise,     error('Unknown scale "%s".', scale);
    end
end


function nll = gauss_nll_shared(params, scale, targets, obs_counts_tg, bins, n_bins)
    % One (b, sigma) pair scored against BOTH targets' counts.
    nll = 0;
    for ti = 1 : numel(targets)
        mu     = mu_shared(params(1), scale, targets(ti));
        pred_p = gauss_predict(mu, params(2), scale, targets(ti), bins, n_bins);
        nll    = nll - sum(obs_counts_tg(ti, :) .* log(pred_p));
    end
end


function nll = gauss_nll_single(params, scale, target, obs_counts, bins, n_bins)
    pred_p = gauss_predict(params(1), params(2), scale, target, bins, n_bins);
    nll = -sum(obs_counts .* log(pred_p));
end


function pred_p = gauss_predict(mu, sigma, scale, target, bins, n_bins)
    % Height of the Gaussian at each candidate response, normalised over the
    % candidate set.  Must stay identical to predict_gauss in
    % fit_gaussian_mle_MS.m — see the comment there for why the height and
    % not the mass between half-integer edges.
    R_MAX = 40;

    sigma = max(sigma, 1e-6);
    r     = 1 : R_MAX;

    switch scale
        case 'linear', x = r;
        case 'log',    x = log(r);
        otherwise,     error('Unknown scale "%s".', scale);
    end

    dens = exp(-0.5 * ((x - mu) / sigma) .^ 2);
    if sum(dens) > 0
        dens = dens / sum(dens);
    else
        [~, i_near] = min(abs(x - mu));
        dens = zeros(1, R_MAX);  dens(i_near) = 1;
    end

    resp   = target + bins;
    in_set = resp >= 1 & resp <= R_MAX;

    pred_p           = zeros(1, n_bins + 1);
    pred_p(in_set)   = dens(resp(in_set));
    pred_p(end)      = max(1 - sum(pred_p(1:n_bins)), 0);
    pred_p = max(pred_p, 1e-10);
    pred_p = pred_p / sum(pred_p);
end

% -------------------------------------------------------------------------
function p = fun_paired_p(d)
% Signed-rank p for a vector of paired differences, guarded for the cases a
% partial run produces: too few values, or all differences exactly zero.
d = d(~isnan(d));
if numel(d) < 2 || ~any(d ~= 0) || ~exist('signrank', 'file')
    p = NaN;
else
    p = signrank(d);
end
end
