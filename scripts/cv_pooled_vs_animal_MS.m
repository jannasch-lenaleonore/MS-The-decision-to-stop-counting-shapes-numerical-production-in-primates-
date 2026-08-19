% cv_pooled_vs_animal_MS.m
%
% Does fitting the two animals separately predict held-out data better than
% fitting them together?  This is the out-of-sample counterpart to the
% question "is one parameter set for both monkeys justified?", and it is a
% stronger argument than a non-significant difference test: it asks whether
% the extra parameters BUY anything, rather than whether they are detectable.
%
% Two arms, both scored on exactly the same held-out trials:
%
%   POOLED      one parameter set estimated from all training sessions of
%               both animals, used to predict both animals' held-out trials
%   PER-ANIMAL  one parameter set per animal, each estimated from that
%               animal's training sessions only and used to predict that
%               animal's held-out trials
%
% The per-animal arm has twice as many free parameters.  If it does not
% improve held-out log-likelihood, the pooled fit is justified on predictive
% grounds and the extra parameters are not earning their keep.
%
% Leave-one-session-out, exactly as in cv_models_MS.m: fold k is session k.
% The pooled arm's diffusion fit for that fold is therefore already in
% results/cv_results.mat and is read straight out of it; only the per-animal
% fits are computed here.  Because a held-out session belongs to one animal,
% only that animal's per-animal fit is needed per fold — one diffusion fit
% per session, 76 in total.
%
% All five models are run.  Only the diffusion model is expensive; the
% Gaussians are analytic and cost nothing.
%
% Run fit_diffusion_mle_MS.m and cv_models_MS.m first.
% Saves results/cv_pooled_vs_animal.mat.

clear; clc;

% =========================================================================
% 1.  Settings — the fitting protocol must match cv_models_MS.m exactly, so
%     that the cached pooled fits and the new per-animal fits are produced
%     under identical conditions.  Anything else would bias the comparison.
% =========================================================================
quick_test = false;

n_reps = 1;        % leave-one-session-out: nothing to repeat
rng(42)

n_starts_cv   = 2;
max_fun_evals = 300;
n_sim         = 20000;
n_eval        = 500000;   % matches fit_diffusion_mle_MS.m and cv_models_MS.m

max_folds = Inf;   % quick test only

if quick_test
    max_folds = 3;  n_starts_cv = 1;
    max_fun_evals = 60;  n_sim = 4000;  n_eval = 20000;
    fprintf('*** QUICK TEST — results are not usable ***\n\n');
end

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

targets_monk = [3, 4];
RT_cutoff    = 1100;
bins         = -2 : 2;
n_bins       = numel(bins);
n_targets    = numel(targets_monk);

model_names = {'Accumulation to bound', 'Gaussian linear (shared)', ...
    'Gaussian log (shared)', 'Gaussian linear (per target)', ...
    'Gaussian log (per target)', 'Ceiling (empirical frequencies)'};
gauss_spec  = struct( ...
    'scale',  {'linear', 'log', 'linear', 'log'}, ...
    'shared', {true,     true,  false,    false});
n_models    = numel(model_names);

% The last model has nothing to fit: it is the held-out session predicted by
% the raw response frequencies of the training sessions, one free probability
% per offset category per target.  No model of any form can do better on the
% training counts, so it is the empirical ceiling.  It is computed in one
% analytic pass after the fitting loop (section 7b) rather than inside it, so
% that adding it never invalidates a partial run of the expensive models.
n_models_fit = n_models - 1;

arm_names = {'pooled', 'per-animal'};
n_arms    = numel(arm_names);

project_dir = fileparts(fileparts(mfilename('fullpath')));
monkeys     = {'m1', 'm2'};
n_animals   = numel(monkeys);

% =========================================================================
% 3.  Observed counts per session and target (identical selection)
% =========================================================================
sess_counts = [];
sess_animal = [];
n_sess      = 0;

for mk = 1 : n_animals
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

fprintf('Loaded %d sessions (m1: %d, m2: %d), %d trials.\n\n', n_sess, ...
    sum(sess_animal == 1), sum(sess_animal == 2), sum(sess_counts(:)));

% =========================================================================
% 4.  Warm starts — the full-data optimum of the matching fit set
% =========================================================================
perf      = load(fullfile(project_dir, 'results', 'params_perf.mat'));
fix_drift = perf.fix_drift;

if fix_drift
    lb_d = [0.1,  0.01];
    ub_d = [1.0,  1.00];
else
    lb_d = [0.25, 0.01, 0.01];
    ub_d = [0.5,  2.00, 1.00];
end

x0_pool = perf.fit_perf(strcmp({perf.fit_perf.label}, 'pooled')).x_opt;
x0_anim = NaN(n_animals, numel(x0_pool));
for mk = 1 : n_animals
    x0_anim(mk, :) = perf.fit_perf(strcmp({perf.fit_perf.label}, monkeys{mk})).x_opt;
end
fprintf('Warm starts:  pooled [%s]', ...
    strjoin(arrayfun(@(v) sprintf('%.4f', v), x0_pool, 'UniformOutput', false), ' '));
for mk = 1 : n_animals
    fprintf('   %s [%s]', monkeys{mk}, strjoin(arrayfun(@(v) sprintf('%.4f', v), ...
        x0_anim(mk,:), 'UniformOutput', false), ' '));
end
fprintf('\n\n');

opts = optimset('Display', 'off', 'MaxFunEvals', max_fun_evals, ...
    'MaxIter', max_fun_evals, 'TolX', 1e-3, 'TolFun', 1e-3);

% =========================================================================
% 5.  Fold assignment — one fold per session, as in cv_models_MS.m
% =========================================================================
n_folds = n_sess;
fold_of = 1 : n_sess;

cv_n = NaN(n_reps, n_folds);
for k = 1 : n_folds
    held       = sess_counts(fold_of(1, :) == k, :, :);
    cv_n(1, k) = sum(held(:));
end

% =========================================================================
% 6.  Reuse of the pooled diffusion refits from cv_models_MS.m
% =========================================================================
ddm_cache_ll     = NaN(n_reps, n_folds);
ddm_cache_params = cell(n_reps, n_folds);

cache_file = fullfile(project_dir, 'results', 'cv_results.mat');
if ~quick_test && isfile(cache_file)
    C = load(cache_file);
    settings_ok = all(isfield(C, {'cv_ll','cv_n','cv_params','fold_of', ...
        'model_names','n_starts_cv','n_sim','n_eval','RT_cutoff','targets_monk'})) && ...
        isequal(size(C.cv_ll, 3), n_folds)  && isequal(C.n_starts_cv, n_starts_cv) && ...
        isequal(C.n_sim, n_sim)             && isequal(C.n_eval, n_eval) && ...
        isequal(C.RT_cutoff, RT_cutoff)     && isequal(C.targets_monk, targets_monk) && ...
        strcmp(C.model_names{1}, model_names{1});
    if settings_ok
        n_reused = 0;
        for rep = 1 : n_reps
            src = find(arrayfun(@(r) isequal(C.fold_of(r,:), fold_of(rep,:)), ...
                1 : size(C.fold_of, 1)), 1);
            if isempty(src) || ~isequal(C.cv_n(src,:), cv_n(rep,:)), continue; end
            ddm_cache_ll(rep, :)     = reshape(C.cv_ll(1, src, :), 1, n_folds);
            ddm_cache_params(rep, :) = reshape(C.cv_params(1, src, :), 1, n_folds);
            n_reused = n_reused + sum(~isnan(ddm_cache_ll(rep, :)));
        end
        fprintf('Reusing %d of %d pooled diffusion refits from cv_results.mat.\n\n', ...
            n_reused, n_reps * n_folds);
    else
        fprintf('cv_results.mat does not match these settings — refitting the pooled arm too.\n\n');
    end
end

% =========================================================================
% 7.  Cross-validation over both arms
% =========================================================================
% Held-out log-likelihood, split by animal so each arm can be inspected
% where it differs: [arm x model x rep x fold x animal]
cv2_ll     = NaN(n_arms, n_models, n_reps, n_folds, n_animals);
cv2_params = cell(n_arms, n_models, n_reps, n_folds, n_animals);

partial_file = fullfile(project_dir, 'results', 'cv_pooled_vs_animal_partial.mat');
if quick_test
    partial_file = fullfile(project_dir, 'results', 'cv_pva_partial_QUICKTEST.mat');
end
% A fold's entries for the animal that does NOT own the held-out session stay
% NaN, so completion is judged on the tested animal's slice only.  Only the
% fitted models count: the ceiling is filled in later and is never a reason to
% redo a fold.
fold_done = @(A, k, mk) all(~isnan(reshape(A(:, 1:n_models_fit, 1, k, mk), 1, [])));

% Seed from the partial run if there is one, otherwise from a completed run of
% this script.  The second case is what makes appending a cheap model to the
% list free: the expensive fits are already on disk in the final .mat.
seed_files = {partial_file, fullfile(project_dir, 'results', out_name_of(quick_test))};
for f = 1 : numel(seed_files)
    if ~isfile(seed_files{f}), continue; end
    P = load(seed_files{f});
    if ~all(isfield(P, {'cv2_ll', 'cv2_params', 'fold_of'})), continue; end

    % A file written before a model was appended has a shorter model
    % dimension; its entries still belong in the same slots, so pad it.
    sz_p = size(P.cv2_ll);
    sz_w = size(cv2_ll);
    if numel(sz_p) == numel(sz_w) && sz_p(2) <= n_models && ...
            isequal(sz_p([1 3 4 5]), sz_w([1 3 4 5]))
        if sz_p(2) < n_models
            pad_ll = NaN(n_arms, n_models, n_reps, n_folds, n_animals);
            pad_pr = cell(n_arms, n_models, n_reps, n_folds, n_animals);
            pad_ll(:, 1:sz_p(2), :, :, :) = P.cv2_ll;
            pad_pr(:, 1:sz_p(2), :, :, :) = P.cv2_params;
            P.cv2_ll = pad_ll;  P.cv2_params = pad_pr;
            fprintf('%s has %d models, this run has %d — padding.\n', ...
                seed_files{f}, sz_p(2), n_models);
        end
    else
        fprintf('Ignoring %s — array shape %s does not fit %s.\n\n', ...
            seed_files{f}, mat2str(sz_p), mat2str(sz_w));
        continue
    end

    if isequal(P.fold_of, fold_of)
        cv2_ll     = P.cv2_ll;
        cv2_params = P.cv2_params;
        n_done     = sum(arrayfun(@(k) fold_done(cv2_ll, k, sess_animal(k)), 1 : n_folds));
        fprintf('Seeding from %s: %d of %d sessions already done.\n\n', ...
            seed_files{f}, n_done, n_folds);
        break
    else
        fprintf('Ignoring %s — different fold assignment.\n\n', seed_files{f});
    end
end

fprintf('Pooled vs per-animal, leave-one-session-out over %d sessions\n', n_folds);
fprintf(['(the per-animal arm needs one diffusion fit per session — the ' ...
    'held-out session\nbelongs to a single animal, so only that animal is ' ...
    'refitted)\n\n']);

t_start = tic;
for rep = 1 : n_reps
    for k = 1 : min(n_folds, max_folds)
        mk_test = sess_animal(k);           % the held-out session's animal
        if fold_done(cv2_ll, k, mk_test)
            continue
        end
        test_sess  = fold_of(rep, :) == k;
        train_sess = ~test_sess;

        % Held-out counts per animal — the scoring target for both arms.  Only
        % mk_test contributes any; the other animal's row is all zeros.
        test_tg_a = NaN(n_animals, n_targets, n_bins + 1);
        for mk = 1 : n_animals
            test_tg_a(mk, :, :) = sum(sess_counts(test_sess & sess_animal == mk, :, :), 1);
        end

        % ---- Arm 1: one parameter set from all training sessions ---------
        train_tg = reshape(sum(sess_counts(train_sess, :, :), 1), n_targets, n_bins + 1);
        pred_arm = NaN(n_arms, n_models, n_animals, n_targets, n_bins + 1);

        if ~isnan(ddm_cache_ll(rep, k))
            x_ddm   = ddm_cache_params{rep, k};
            ddm_tag = 'cached';
        else
            x_ddm   = fit_ddm(train_tg, x0_pool, lb_d, ub_d, n_starts_cv, ...
                fix_drift, bound, n_sim, n_bins, bins, t_vec, n_t, dt, noise, period, opts);
            ddm_tag = 'fitted';
        end
        rng(5000 + k)
        pp = ddm_simulate(x_ddm, fix_drift, bound, n_eval, ...
            n_bins, bins, t_vec, n_t, dt, noise, period);
        for mk = 1 : n_animals
            for ti = 1 : n_targets
                pred_arm(1, 1, mk, ti, :) = pp;
            end
            cv2_params{1, 1, rep, k, mk} = x_ddm;
        end

        for mi = 2 : n_models_fit
            [pg, xg] = fit_gauss_model(gauss_spec(mi-1), train_tg, targets_monk, ...
                bins, n_bins, opts);
            for mk = 1 : n_animals
                pred_arm(1, mi, mk, :, :) = pg;
                cv2_params{1, mi, rep, k, mk} = xg;
            end
        end

        % ---- Arm 2: one parameter set per animal -------------------------
        % Only the held-out session's own animal is refitted: the other
        % animal has no held-out trials this fold, so its fit would never be
        % scored.  That is what makes this arm 76 fits rather than 152.
        for mk = mk_test
            tr_mk = train_sess & sess_animal == mk;
            train_tg_mk = reshape(sum(sess_counts(tr_mk, :, :), 1), n_targets, n_bins + 1);

            x_ddm_mk = fit_ddm(train_tg_mk, x0_anim(mk,:), lb_d, ub_d, n_starts_cv, ...
                fix_drift, bound, n_sim, n_bins, bins, t_vec, n_t, dt, noise, period, opts);
            rng(6000 + 100 * k + mk)
            pp_mk = ddm_simulate(x_ddm_mk, fix_drift, bound, n_eval, ...
                n_bins, bins, t_vec, n_t, dt, noise, period);
            for ti = 1 : n_targets
                pred_arm(2, 1, mk, ti, :) = pp_mk;
            end
            cv2_params{2, 1, rep, k, mk} = x_ddm_mk;

            for mi = 2 : n_models_fit
                [pg, xg] = fit_gauss_model(gauss_spec(mi-1), train_tg_mk, ...
                    targets_monk, bins, n_bins, opts);
                pred_arm(2, mi, mk, :, :) = pg;
                cv2_params{2, mi, rep, k, mk} = xg;
            end
        end

        % ---- Score both arms on the same held-out trials ------------------
        for a = 1 : n_arms
            for mi = 1 : n_models_fit
                pt = reshape(pred_arm(a, mi, mk_test, :, :), n_targets, n_bins + 1);
                oc = reshape(test_tg_a(mk_test, :, :), n_targets, n_bins + 1);
                cv2_ll(a, mi, rep, k, mk_test) = sum(sum(oc .* log(pt)));
            end
        end

        save(partial_file, 'cv2_ll', 'cv2_params', 'cv_n', 'fold_of');

        d_ddm = cv2_ll(2,1,rep,k,mk_test) - cv2_ll(1,1,rep,k,mk_test);
        fprintf(['  session %2d/%d (%s, n=%4d, pooled DDM %s):  pooled %9.2f   ' ...
            'per-animal %9.2f   gain %+7.2f  [%.0f min]\n'], ...
            k, n_folds, monkeys{mk_test}, cv_n(rep, k), ddm_tag, ...
            cv2_ll(1,1,rep,k,mk_test), cv2_ll(2,1,rep,k,mk_test), ...
            d_ddm, toc(t_start)/60);
    end
end
fprintf('\nFinished in %.1f min.\n\n', toc(t_start) / 60);

% =========================================================================
% 7b.  Ceiling model — the same two arms, no fitting
% =========================================================================
% Predict the held-out session from the training sessions' own response
% frequencies, separately per target.  Pooled arm: frequencies of both
% animals' training sessions.  Per-animal arm: frequencies of the held-out
% animal's training sessions only.  Nothing is optimised, so this is exact and
% costs no simulation; it is recomputed on every run.
%
% This is the strongest thing the per-animal arm could possibly buy: if two
% animals differ in their response distribution at all, a model made of raw
% frequencies will find it.  It therefore calibrates the DDM comparison — a
% null result for the DDM is only informative if this one is positive.
for k = 1 : min(n_folds, max_folds)
    mk_test    = sess_animal(k);
    test_sess  = fold_of(1, :) == k;
    train_sess = ~test_sess;

    oc = reshape(sum(sess_counts(test_sess, :, :), 1), n_targets, n_bins + 1);

    src = {train_sess, train_sess & sess_animal == mk_test};
    for a = 1 : n_arms
        tr = reshape(sum(sess_counts(src{a}, :, :), 1), n_targets, n_bins + 1);
        pt = NaN(n_targets, n_bins + 1);
        for ti = 1 : n_targets
            p = tr(ti, :) / sum(tr(ti, :));
            p = max(p, 1e-10);          % same guard as the fitted predictors
            pt(ti, :) = p / sum(p);
        end
        cv2_ll(a, n_models, 1, k, mk_test) = sum(sum(oc .* log(pt)));
    end
end

% =========================================================================
% 8.  Summary
% =========================================================================
% Collapse the animal dimension: each session was scored under exactly one
% animal, so this just picks out the non-NaN entry.  [arm x model x session]
ll2      = NaN(n_arms, n_models, n_folds);
for k = 1 : n_folds
    ll2(:, :, k) = cv2_ll(:, :, 1, k, sess_animal(k));
end
done     = reshape(~isnan(ll2(1, 1, :)), 1, []);
n_done   = sum(done);
n_tot    = sum(cv_n(1, done));

% Per trial within each session, so long and short sessions weigh equally —
% the same currency as cv_models_MS.m.
gain_ps  = @(mi) reshape(ll2(2, mi, done) - ll2(1, mi, done), 1, []) ./ ...
    reshape(cv_n(1, done), 1, []);

if n_done < n_folds
    fprintf('*** only %d of %d sessions scored — summary is partial ***\n\n', ...
        n_done, n_folds);
end

fprintf('Held-out log-likelihood per trial, over the %d held-out sessions\n', n_done);
fprintf('%-32s %12s %12s %12s\n', ...
    'Model', 'pooled', 'per-animal', 'gain/trial');
fprintf('%s\n', repmat('-', 1, 72));
for mi = 1 : n_models
    a1 = sum(ll2(1, mi, done));
    a2 = sum(ll2(2, mi, done));
    fprintf('%-32s %12.4f %12.4f %12.5f\n', model_names{mi}, ...
        a1 / n_tot, a2 / n_tot, (a2 - a1) / n_tot);
end

fprintf('\nThe same, summed over sessions (nats)\n');
fprintf('%-32s %12s %12s %12s\n', 'Model', 'pooled', 'per-animal', 'gain');
fprintf('%s\n', repmat('-', 1, 72));
for mi = 1 : n_models
    a1 = sum(ll2(1, mi, done));
    a2 = sum(ll2(2, mi, done));
    fprintf('%-32s %12.1f %12.1f %12.2f\n', model_names{mi}, a1, a2, a2 - a1);
end
fprintf(['\nA positive gain means the extra per-animal parameters improved ' ...
    'prediction of\nheld-out sessions; a negative gain means they cost ' ...
    'accuracy (overfitting).\n\n']);

% Paired per-session test, per model
fprintf('Paired per-session comparison (%d held-out sessions):\n', n_done);
for mi = 1 : n_models
    d = gain_ps(mi);
    fprintf(['  %-30s per-animal better in %2d/%2d sessions, ' ...
        'mean %+.5f/trial (SD %.5f), p = %.3g\n'], ...
        model_names{mi}, sum(d > 0), numel(d), mean(d), std(d), fun_paired_p(d));
end
fprintf('\n');

% The same, per animal — a shared fit may suit one animal and not the other
fprintf('Diffusion model, gain from per-animal parameters, split by animal:\n');
an_done = sess_animal(done);
for mk = 1 : n_animals
    d = gain_ps(1);
    d = d(an_done == mk);
    fprintf('  %-6s better in %2d/%2d sessions, mean %+.5f/trial, p = %.3g\n', ...
        monkeys{mk}, sum(d > 0), numel(d), mean(d), fun_paired_p(d));
end
fprintf('\n');

% =========================================================================
% 9.  Save
% =========================================================================
results_dir = fullfile(project_dir, 'results');
out_name = out_name_of(quick_test);
save(fullfile(results_dir, out_name), ...
    'cv2_ll', 'cv2_params', 'cv_n', 'fold_of', 'sess_animal', 'sess_counts', ...
    'arm_names', 'model_names', 'gauss_spec', 'monkeys', 'n_folds', 'n_reps', ...
    'targets_monk', 'bins', 'n_starts_cv', 'n_sim', 'n_eval', 'RT_cutoff');
fprintf('Saved to %s\n', fullfile(results_dir, out_name));

% =========================================================================
% Local functions
% =========================================================================

function x_opt = fit_ddm(train_tg, x0, lb, ub, n_starts, fix_drift, bound, ...
        n_sim, n_bins, bins, t_vec, n_t, dt, noise, period, opts)
    % The diffusion model is target-invariant, so it is fitted to the counts
    % pooled over targets, exactly as in cv_models_MS.m.
    train_po = sum(train_tg, 1);
    obj = @(p) -sum(train_po .* log(ddm_simulate(p, fix_drift, bound, n_sim, ...
        n_bins, bins, t_vec, n_t, dt, noise, period)));

    starts = x0;
    for s = 2 : n_starts
        starts(s, :) = min(max(x0 .* (1 + 0.15 * randn(size(x0))), lb + 1e-3), ub - 1e-3);
    end

    best_nll = Inf;  x_opt = x0;
    for s = 1 : n_starts
        [x_s, nll_s] = bounded_fminsearch(obj, starts(s,:), lb, ub, opts);
        if nll_s < best_nll
            best_nll = nll_s;
            x_opt    = x_s;
        end
    end
end


function [pred_tg, x_store] = fit_gauss_model(G, train_tg, targets, bins, n_bins, opts)
    n_targets = numel(targets);
    pred_tg   = NaN(n_targets, n_bins + 1);

    if G.shared
        switch G.scale
            case 'linear', x0 = [0, 0.70];  lb = [-2.0, 0.05];  ub = [2.0, 3.00];
            case 'log',    x0 = [0, 0.25];  lb = [-0.7, 0.02];  ub = [0.7, 1.50];
        end
        obj = @(p) gauss_nll_shared(p, G.scale, targets, train_tg, bins, n_bins);
        x_store = bounded_fminsearch(obj, x0, lb, ub, opts);
        for ti = 1 : n_targets
            mu = mu_shared(x_store(1), G.scale, targets(ti));
            pred_tg(ti, :) = gauss_predict(mu, x_store(2), G.scale, targets(ti), bins, n_bins);
        end
    else
        x_store = NaN(n_targets, 2);
        for ti = 1 : n_targets
            tg = targets(ti);
            switch G.scale
                case 'linear', x0 = [tg, 0.70];       lb = [tg-2, 0.05];       ub = [tg+2, 3.00];
                case 'log',    x0 = [log(tg), 0.25];  lb = [log(tg)-0.7, 0.02]; ub = [log(tg)+0.7, 1.50];
            end
            obj = @(p) gauss_nll_single(p, G.scale, tg, train_tg(ti,:), bins, n_bins);
            x_ti = bounded_fminsearch(obj, x0, lb, ub, opts);
            pred_tg(ti, :) = gauss_predict(x_ti(1), x_ti(2), G.scale, tg, bins, n_bins);
            x_store(ti, :) = x_ti;
        end
    end
end


function [x_opt, nll_opt] = bounded_fminsearch(obj, x0, lb, ub, opts)
    to_raw    = @(p) log((p - lb) ./ (ub - p));
    to_params = @(r) lb + (ub - lb) ./ (1 + exp(-r));
    [r_opt, nll_opt] = fminsearch(@(r) obj(to_params(r)), to_raw(x0), opts);
    x_opt = to_params(r_opt);
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
    switch scale
        case 'linear', mu = target + b;
        case 'log',    mu = log(target) + b;
        otherwise,     error('Unknown scale "%s".', scale);
    end
end


function nll = gauss_nll_shared(params, scale, targets, obs_counts_tg, bins, n_bins)
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
function nm = out_name_of(quick_test)
% Named once so that section 7's seeding and section 9's save cannot drift.
if quick_test
    nm = 'cv_pooled_vs_animal_QUICKTEST.mat';
else
    nm = 'cv_pooled_vs_animal.mat';
end
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
