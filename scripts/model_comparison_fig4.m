% model_comparison_fig4.m
%
% Compares three accounts of the monkeys' response distributions on the
% offset bins -2:+2:
%
%   Stopping                     accumulation to a stopping bound; 2
%                                parameters, SHARED across both target
%                                numerosities (the model is target-invariant
%                                by construction)     -> fit_diffusion_mle_MS
%   Linear number line           2 parameters, one shared linear number line
%   Log number line              2 parameters, one shared log number line
%                                                     -> fit_gaussian_mle_MS
%
% Those are the DISPLAY labels (model_disp).  The strings stored in the .mat
% files are the older internal keys ('Accumulation to bound', 'Gaussian
% linear (shared)', 'Gaussian log (shared)'); they are kept in model_names
% and used for every by-name lookup, so no refit is needed.
%
% All three are matched in flexibility: two free parameters that have to
% account for both targets at once.  In each, location and width are the free
% parameters and the proportion of correct trials is a PREDICTION (the
% offset-0 probability), never a fitted quantity.
%
% The numerosity-specific (per-target) Gaussian fits that fit_gaussian_mle_MS
% also produces are NOT compared here.  The saturated ceiling drawn on MS Fig. 4E
% already bounds what any numerosity-specific account could reach, and it
% does so without introducing a fitted competitor that has to be penalised
% for its extra parameters.
%
% Each comparison is run on three fit sets — the combined data of both
% animals (the version used to illustrate the mechanism) and each animal
% separately.
%
% Run fit_diffusion_mle_MS.m and fit_gaussian_mle_MS.m first; run
% cv_models_MS.m and cv_pooled_vs_animal_MS.m as well for the
% cross-validation panel and the per-animal table.
%
% Figures written to figures/:
%   fig4C_model_fits.pdf       MS Fig. 4C  predictions of all three models
%                                          fitted on the COMBINED data, both
%                                          targets on one shared release axis
%   fig4D_loglikelihood.pdf    MS Fig. 4D  per-trial log-likelihood of the
%                                          combined fit, scored per animal,
%                                          against that animal's ceiling
%   fig4E_crossvalidation.pdf  MS Fig. 4E  held-out log-likelihood per
%                                          session (left panel is the one
%                                          reproduced in the manuscript; the
%                                          other panels carry the paired
%                                          within-session statistics quoted
%                                          in Results)
%
% Printed to the console: the in-sample log-likelihood table (Results,
% "Model predictions were compared with 16537 monkey trials" and Methods,
% "Comparison models"), the cross-validation table (Methods, "Cross
% validation across sessions") and the per-animal held-out comparison
% (Results, "Individual parameter fits per animal ...").

clear; clc;

% =========================================================================
% 1.  Load fitted models
% =========================================================================
project_dir = fileparts(fileparts(mfilename('fullpath')));
results_dir = fullfile(project_dir, 'results');

perf  = load(fullfile(results_dir, 'params_perf.mat'));
gauss = load(fullfile(results_dir, 'params_gauss.mat'));

if ~isfield(perf, 'fit_perf')
    error(['params_perf.mat has no per-monkey fits. ' ...
        'Re-run fit_diffusion_mle_MS.m with the updated script.']);
end
if ~isfield(gauss, 'gauss_models')
    error(['params_gauss.mat predates the shared-number-line models. ' ...
        'Re-run fit_gaussian_mle_MS.m with the updated script.']);
end

targets_monk = gauss.targets_monk;
bins         = gauss.bins;
n_bins       = numel(bins);
n_targets    = numel(targets_monk);

set_labels = {'pooled', 'm1', 'm2'};
set_names  = {'Combined', 'Monkey 1', 'Monkey 2'};
n_sets     = numel(set_labels);

% Only the SHARED number lines are compared here, so every model in this
% script carries exactly 2 free parameters.  The numerosity-specific
% (per-target) Gaussian fits are deliberately left out: the saturated
% ceiling drawn on MS Fig. 4E already bounds what any numerosity-specific
% account could achieve, and it does so without adding a fitted model.
% fit_gaussian_mle_MS.m still produces the per-target fits, so they remain
% in params_gauss.mat if they are ever wanted.
gauss_keep  = find([gauss.gauss_models.shared]);
n_gauss     = numel(gauss_keep);

% model_names are LOOKUP KEYS, not labels: they must keep matching the
% strings written into params_gauss.mat, cv_results.mat and
% cv_pooled_vs_animal.mat, which the by-name lookups below rely on.  Renaming
% them here would silently break those lookups (or force a refit).
model_names = [{'Accumulation to bound'}, {gauss.gauss_models(gauss_keep).name}];
n_models    = numel(model_names);

% model_disp are the LABELS shown to a reader — in every figure and every
% printed table.  The contrast the comparison is about is stopping versus a
% number line, so each label names its commitment.
model_disp  = {'Stopping', 'Linear number line', 'Log number line'};
model_short = {'stop', 'lin NL', 'log NL'};

% Two-line labels for the axis ticks of MS Fig. 4D and 4E.  The line break is
% \newline rather than a literal newline: tick labels are rendered by the
% TeX interpreter, which splits a char containing \n into separate labels.
model_tick = {'Stopping', 'Linear\newlinenumber line', ...
              'Log\newlinenumber line'};

% =========================================================================
% 2.  Collect predictions and observed counts per fit set and target
% =========================================================================
pred      = NaN(n_models, n_sets, n_targets, n_bins + 1);
obs       = NaN(n_sets, n_targets, n_bins + 1);
k_per_set = zeros(n_models, n_sets);
ddm_sd    = NaN(1, n_sets);

for si = 1 : n_sets
    ddm = perf.fit_perf(strcmp({perf.fit_perf.label}, set_labels{si}));
    if isempty(ddm)
        error('No diffusion fit "%s" in params_perf.mat.', set_labels{si});
    end
    k_per_set(1, si) = ddm.k;               % shared across targets
    if isfield(ddm, 'nll_eval_sd'), ddm_sd(si) = ddm.nll_eval_sd; end

    gi = find(strcmp(gauss.set_labels, set_labels{si}), 1);
    if isempty(gi)
        error('No Gaussian fit "%s" in params_gauss.mat.', set_labels{si});
    end

    for ti = 1 : n_targets
        obs(si, ti, :)     = ddm.obs_counts_tg(ti, :);
        pred(1, si, ti, :) = ddm.pred_p;    % target-invariant
    end

    for mi = 1 : n_gauss
        g = gauss.fit_gauss(gauss_keep(mi), gi);
        k_per_set(mi + 1, si) = g.k;        % already totalled over targets

        for ti = 1 : n_targets
            pred(mi + 1, si, ti, :) = g.pred_p(ti, :);

            if ~isequal(g.obs_counts(ti, :), ddm.obs_counts_tg(ti, :))
                error(['Observed counts differ between params_perf.mat and ' ...
                    'params_gauss.mat for set %s target %d — the two fitters ' ...
                    'are not using the same trials.'], set_labels{si}, targets_monk(ti));
            end
        end
    end
end

% =========================================================================
% 3.  Log-likelihood, AIC and BIC
%
%     AIC = 2k + 2*nll        BIC = k*log(n) + 2*nll
% =========================================================================
nll_tg    = NaN(n_models, n_sets, n_targets);
nll_total = NaN(n_models, n_sets);
n_trials  = NaN(1, n_sets);
nll_sat   = NaN(1, n_sets);

for si = 1 : n_sets
    n_trials(si) = sum(sum(obs(si, :, :)));
    sat = 0;
    for ti = 1 : n_targets
        oc  = reshape(obs(si, ti, :), 1, []);
        nz  = oc > 0;
        sat = sat - sum(oc(nz) .* log(oc(nz) / sum(oc)));
        for mi = 1 : n_models
            nll_tg(mi, si, ti) = -sum(oc .* reshape(log(pred(mi, si, ti, :)), 1, []));
        end
    end
    nll_sat(si)      = sat;
    nll_total(:, si) = sum(nll_tg(:, si, :), 3);
end

aic = 2 * k_per_set + 2 * nll_total;
bic = k_per_set .* log(repmat(n_trials, n_models, 1)) + 2 * nll_total;

d_aic = aic - min(aic, [], 1);
d_bic = bic - min(bic, [], 1);

% =========================================================================
% 4.  Print comparison table
% =========================================================================
fprintf('Model comparison on response frequencies (offsets -2:+2)\n');
fprintf(['Targets %s; the stopping model and the two shared number lines ' ...
    'use one\nparameter set for both targets.\n\n'], mat2str(targets_monk));

% "% of range" places a model on the interval between the two reference
% points that bracket any possible model.  The FLOOR is a model that knows
% only which categories exist and spreads its mass evenly over them; the
% CEILING is the saturated model, handed each target's exact response
% frequencies.  0% means the fit is worth no more than guessing uniformly,
% 100% that it is as good as reading the answer off the data.  It is the same
% scale the cross-validation table uses, so the two can be compared directly,
% with the caveat that the ceiling is computed in sample.
ll_floor  = log(1 / n_bins);
pct_range = @(ll, si) 100 * (ll - ll_floor) / (-nll_sat(si) / n_trials(si) - ll_floor);

for si = 1 : n_sets
    fprintf('--- %s  (N = %d trials) ---\n', set_names{si}, n_trials(si));
    fprintf('%-30s %4s %10s %10s %10s %10s %10s %10s %10s\n', ...
        'Model', 'k', 'neg-LL', 'LL/trial', '% of range', 'AIC', 'BIC', ...
        'dAIC', 'dBIC');
    fprintf('%s\n', repmat('-', 1, 110));
    for mi = 1 : n_models
        ll_mi = -nll_total(mi, si) / n_trials(si);
        fprintf('%-30s %4d %10.2f %10.4f %10.1f %10.1f %10.1f %10.1f %10.1f', ...
            model_disp{mi}, k_per_set(mi, si), nll_total(mi, si), ...
            ll_mi, pct_range(ll_mi, si), ...
            aic(mi, si), bic(mi, si), d_aic(mi, si), d_bic(mi, si));
        if mi == 1 && ~isnan(ddm_sd(si))
            % Only the diffusion model is simulation-based, so only its
            % neg-LL carries Monte-Carlo noise.
            fprintf('   (MC SD %.1f)', ddm_sd(si));
        end
        fprintf('\n');
    end
    fprintf('%-30s %4s %10.2f %10.4f %10.1f %10s %10s %10s %10s\n', ...
        'saturated (ceiling)', '-', nll_sat(si), -nll_sat(si) / n_trials(si), ...
        100, '-', '-', '-', '-');
    fprintf('%-30s %4s %10s %10.4f %10.1f %10s %10s %10s %10s\n', ...
        'uniform over 5 categories', '-', '-', ll_floor, 0, ...
        '-', '-', '-', '-');
    fprintf('\n');
end
fprintf(['dAIC / dBIC are differences to the best model within a fit set ' ...
    '(0 = best).\nThe saturated row is the lowest neg-LL any model could ' ...
    'reach on these counts.\n"%% of range" places each model between the ' ...
    'uniform floor (0%%) and that\nsaturated ceiling (100%%); it is in ' ...
    'sample, so a held-out model sits slightly lower.\n\n']);

% LL per trial is the same quantity the cross-validation reports, so the two
% tables can be read against each other directly: the in-sample value is the
% model scored on the trials it was fitted to, the held-out value on trials it
% was not.  Its exponential is the geometric-mean probability the model gave
% to the response that actually occurred.
fprintf('Log-likelihood per trial, per target:\n');
fprintf('%-12s %-8s', 'Set', 'Target');
fprintf('%-14s', model_short{:});  fprintf('\n');
for si = 1 : n_sets
    for ti = 1 : n_targets
        n_ti = sum(obs(si, ti, :));
        fprintf('%-12s %-8d', set_names{si}, targets_monk(ti));
        fprintf('%-14.4f', -nll_tg(:, si, ti) / n_ti);
        fprintf('\n');
    end
end
fprintf('\n');

% Proportion correct is a prediction of every model, not a free parameter.
% Its target-to-target change is the sharpest discriminator between a
% linear and a logarithmic number line, so report it explicitly.
i0 = find(bins == 0);
fprintf('Proportion of correct trials (offset 0) — predicted vs observed:\n');
fprintf('%-12s %-8s %-10s', 'Set', 'Target', 'observed');
fprintf('%-14s', model_short{:});  fprintf('\n');
for si = 1 : n_sets
    for ti = 1 : n_targets
        oc = reshape(obs(si, ti, :), 1, []);
        fprintf('%-12s %-8d %-10.3f', set_names{si}, targets_monk(ti), oc(i0) / sum(oc));
        fprintf('%-14.3f', pred(:, si, ti, i0));
        fprintf('\n');
    end
end
fprintf('\n');

% =========================================================================
% 5.  Plot settings — colour carries the model, lightness carries the animal
%
%     Encoding: COLOUR carries the model (stopping / linear number line / log
%     number line) and LIGHTNESS carries the animal, dark for monkey 1 and
%     light for monkey 2.  Marker and line style still repeat the model, so
%     the figures survive being printed in greyscale.
%
%     The three hues are from the Okabe-Ito colourblind-safe palette.
% =========================================================================
lw = 1.6;
fs = 14;          % axis labels, tick labels and legends all sit at this size
fs_title = fs - 2;   % titles are sentences, so they stay one notch down
fs_note  = fs - 3;   % in-plot annotations (counts, "ceiling", target labels)

% Every axis is styled through this one call, so tick size and colour cannot
% drift between figures.  Ticks are set to the label size and everything that
% carries text is forced to black — MATLAB's default is a grey that prints
% washed out.
style_ax = @(ax) apply_axis_style(ax, fs, lw);

clr_model_ind = {[0.00 0.45 0.70], ...   % stopping           — blue
                 [0.84 0.37 0.00], ...   % linear number line — vermillion
                 [0.00 0.62 0.45]};      % log number line    — bluish green
sty_model     = {'--', ':', '-.'};
mrk_model     = {'s', '^', 'd'};

% Lightness ramp on a model colour: 0 leaves it alone, 1 washes it to white.
lighten      = @(c, f) c + f * (1 - c);
shade_animal = [0, 0.55];          % monkey 1 dark, monkey 2 light

monk_lbl = {'Monkey 1', 'Monkey 2'};

fit_paper = @(h) set(h, 'PaperUnits', 'centimeters', ...
    'PaperSize', h.Position(3:4), 'PaperPosition', [0 0 h.Position(3:4)]);
fig_dir = fullfile(project_dir, 'figures');
if ~isfolder(fig_dir), mkdir(fig_dir); end

% =========================================================================
% 6.  MS Fig. 4D — per-trial log-likelihood per animal
%
%     This used to show Delta AIC and Delta BIC side by side.  Both were
%     dropped: all three models carry k = 2 free parameters and are scored on
%     the same trials, so BIC's k*log(n) penalty is identical across models
%     and cancels in the difference.  That makes dBIC = dAIC = 2*d(neg-LL)
%     EXACTLY — the two old panels held the same numbers, and both were a
%     rescaling of the log-likelihood reported everywhere else.  The printed
%     table still carries AIC and BIC for anyone who wants them.
%
%     What is shown instead is the per-trial log-likelihood of the COMBINED
%     fit — the one parameter set per model that the paper reports — scored
%     separately on each animal's trials.  Deliberately not the per-animal
%     refits: the paper argues those extra parameters buy nothing, so a
%     figure built from them would undercut its own claim.
%
%     Each animal gets its OWN ceiling line, because the animals differ in
%     how predictable they are (m1 -1.159, m2 -1.139).  Without those lines
%     the panel reads as "the models fit monkey 2 better", when in fact the
%     stopping model sits the same distance from each animal's ceiling.  Bars
%     grow from the uniform floor, which is exactly ln(1/5) for every animal
%     and every session, so bar height IS the explainable-deviance fraction.
% =========================================================================
n_animals_f8 = n_sets - 1;                 % fit sets 2..3 are the animals
ll_pm   = NaN(n_models, n_animals_f8);     % combined fit, scored per animal
ceil_pa = NaN(1, n_animals_f8);            % that animal's saturated ceiling

for si = 2 : n_sets
    a = si - 1;
    ceil_pa(a) = -nll_sat(si) / n_trials(si);
    for mi = 1 : n_models
        s = 0;  N = 0;
        for ti = 1 : n_targets
            oc = reshape(obs(si, ti, :), 1, []);
            pr = reshape(pred(mi, 1, ti, :), 1, []);   % fit set 1 = combined
            s  = s + sum(oc .* log(pr));
            N  = N + sum(oc);
        end
        ll_pm(mi, a) = s / N;
    end
end

% Near-square canvas: the panel carries two bar groups and a short y range,
% so the extra width of a landscape frame was empty space.  Fonts are set a
% few points above the shared sizes here, as on figure 9, so the labels stay
% legible when the figure is placed at a column width.
fs8       = fs + 4;
fs8_title = fs8 - 2;
fs8_note  = fs8 - 2;

fig8 = figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 16 14]);
fit_paper(fig8);  hold on

% One group per animal, one bar per model.  The animal is already the group,
% so the bars keep the pure model colour rather than the usual lightness
% ramp — that keeps the legend unambiguous.
hb = bar(1 : n_animals_f8, ll_pm', 0.78, 'EdgeColor', 'k', 'LineWidth', 0.8, ...
    'BaseValue', ll_floor);
for mi = 1 : n_models
    hb(mi).FaceColor   = clr_model_ind{mi};
    hb(mi).DisplayName = model_disp{mi};
end

y_top = max([ll_pm(:); ceil_pa(:)]);
y_pad = y_top - ll_floor;
for a = 1 : n_animals_f8
    % The ceiling spans only its own animal's group: it is that animal's
    % number, not a figure-wide reference.
    plot(a + [-0.45 0.45], ceil_pa(a) * [1 1], '--', ...
        'Color', [0.30 0.30 0.30], 'LineWidth', 1.4, 'HandleVisibility', 'off')
    % Centred above its own dashed segment rather than hung off the right
    % end: on the near-square canvas the right-hand group's label ran past
    % the axis.
    text(a, ceil_pa(a), 'ceiling', 'FontSize', fs8_note, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', 'k')
end
% The exact values are printed in the table below rather than written on the
% bars: the axis is fine enough to read them off, and printed numbers were
% competing with the ceiling line, which sits only just above the bars.

xticks(1 : n_animals_f8);  xticklabels(set_names(2:end));  xtickangle(0)
xlim([0.4, n_animals_f8 + 0.75])
% Only enough headroom for the word "ceiling" above the highest dashed
% segment: the ceiling is the reference the bars are read against, so it
% belongs at the top of the axis rather than half way up it.  The space that
% used to sit above it now goes to the bars.
ylim([ll_floor, y_top + 0.14 * y_pad])
yline(ll_floor, '-', 'Color', 'k', 'LineWidth', 0.8, 'HandleVisibility', 'off')
% Far right, the only strip of the baseline that neither the legend nor a bar
% covers.  Shortened to the one word, as on figure 10.
text(n_animals_f8 + 0.72, ll_floor, 'floor', 'FontSize', fs8_note, ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Color', 'k')
ylabel('Log-likelihood per trial', 'FontSize', fs8)
% Two lines: at this font size the sentence is wider than the square canvas.
title({'One parameter set per model,', 'scored on each animal'}, ...
    'FontSize', fs8_title)
% Bottom left, over the bars.  The bars are flat-coloured blocks down there
% and the reader takes their value off the top edge, so the key costs no
% information and frees the top of the panel for the ceiling.
% It sits on top of the bars, so it needs an opaque backdrop — but a drawn
% frame would read as a second box inside the panel.  'Box' on supplies the
% white fill, and the edge is then turned off again.
lg8 = legend('Location', 'southwest', 'FontSize', fs8, 'TextColor', 'k', ...
    'NumColumns', 1);
set(lg8, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'none')
apply_axis_style(gca, fs8, lw)

exportgraphics(fig8, fullfile(fig_dir, 'fig4D_loglikelihood.pdf'), ...
    'ContentType', 'vector')

fprintf('\nPer-trial log-likelihood of the combined fit, scored per animal\n');
fprintf('%-22s %10s %10s %10s %10s\n', 'Model', ...
    'Monkey 1', 'to ceiling', 'Monkey 2', 'to ceiling');
fprintf('%s\n', repmat('-', 1, 66));
for mi = 1 : n_models
    fprintf('%-22s %10.4f %10.4f %10.4f %10.4f\n', model_disp{mi}, ...
        ll_pm(mi, 1), ll_pm(mi, 1) - ceil_pa(1), ...
        ll_pm(mi, 2), ll_pm(mi, 2) - ceil_pa(2));
end
fprintf('%-22s %10.4f %10s %10.4f %10s\n', 'ceiling', ceil_pa(1), '—', ...
    ceil_pa(2), '—');
fprintf('%-22s %10.4f %10s %10.4f %10s\n', 'floor (uniform)', ll_floor, '—', ...
    ll_floor, '—');
fprintf(['\nThe animals differ in how predictable they are, so the raw values ' ...
    'are not\ncomparable between them; the distance to each animal''s own ' ...
    'ceiling is.\n']);

% =========================================================================
% 8.  MS Fig. 4C — the three models' predictions, both targets on one release
%     axis.  The observed data are NOT plotted here (they are still printed
%     in the table below); this figure is about what the accounts predict.
%
%     The models shown are the ones the evaluation actually rests on: fitted
%     on the COMBINED data of both animals, one parameter set per model.  The
%     per-animal fits are compared in figures 8, 10 and 12.
%
%     The two target numerosities are drawn over a single shared x-axis: the
%     NUMBER OF RELEASES the monkey actually produced, i.e. target + response
%     offset.  The target-4 family therefore sits one release to the right of
%     the target-3 family, which puts the +1 error at target 3 on the same x
%     as the correct response at target 4.
%
%     Showing the targets separately but aligned is what makes the accounts
%     visibly different.  The stopping model is target-invariant, so its
%     curve is one shape translated by one release; the logarithmic number
%     line predicts a genuinely different shape for each target (wider and
%     later at 4).  Pooling the targets averages that difference away.
%
%     The "outside -2:+2" category is left out — it is 0 in the data and
%     under 1% in every model.  Percentages are of all trials at that target,
%     so the plotted model values are not renormalised over the five bins.
% =========================================================================
si_pool = find(strcmp(set_labels, 'pooled'), 1);

% One curve set per target: row 1 the data, rows 2:end the models.
curves_tg = cell(1, n_targets);
releases  = cell(1, n_targets);
n_tg_tr   = NaN(1, n_targets);
for ti = 1 : n_targets
    oc            = reshape(obs(si_pool, ti, :), 1, []);
    n_tg_tr(ti)   = sum(oc);
    releases{ti}  = targets_monk(ti) + bins;
    pm            = reshape(pred(:, si_pool, ti, :), n_models, n_bins + 1);
    curves_tg{ti} = [100 * oc(1:n_bins) / n_tg_tr(ti); 100 * pm(:, 1:n_bins)];
end

for ti = 1 : n_targets
    fprintf('Both monkeys, target %d (N = %d trials)\n', ...
        targets_monk(ti), n_tg_tr(ti));
    fprintf('%-30s', 'Number of releases:');
    fprintf('%9d', releases{ti});  fprintf('\n');
    fprintf('%-30s', 'Data %:');
    fprintf('%9.1f', curves_tg{ti}(1, :));  fprintf('\n');
    for mi = 1 : n_models
        fprintf('%-30s', sprintf('%s %%:', model_disp{mi}));
        fprintf('%9.1f', curves_tg{ti}(mi + 1, :));  fprintf('\n');
    end
    fprintf('\n');
end

% Both targets share ONE release axis, so the target-4 distribution sits one
% release to the right of the target-3 one: the +1 error at target 3 and the
% correct response at target 4 fall on the same x.  That alignment is the
% point of the figure — on this axis the response distribution is carried
% along bodily by the target rather than re-shaped by it, which is what the
% stopping model predicts and the logarithmic number line does not.
x_all = unique([releases{:}]);

% ---- The two number lines drawn as the distributions they are ------------
%
% Each is a Gaussian HEIGHT over the candidate responses r = 1:R_MAX,
% normalised over that set — evaluated at r on the linear number line, at
% log(r) on the logarithmic one — exactly as predict_gauss in
% fit_gaussian_mle_MS.m does it.  Reusing that normaliser is what makes each
% curve pass exactly through its own markers.
%
% So the logarithmic curve is a Gaussian in log(r) drawn against r: lognormal
% in shape, but without the 1/r Jacobian a properly integrated lognormal
% density would carry.  That height reading is deliberate in this project
% (see the comment in predict_gauss), and the curve has to match it.
%
% Each curve is drawn only over its own target's release range, and never
% below r = 1, which is where the model's candidate set starts.
R_MAX  = 40;
n_fine = 400;

% Drawn on the POOLED fit — the one model per account that the evaluation
% uses.  Colour therefore carries the model and nothing else here.
pred_pool = cell(1, n_targets);             % n_models x n_bins, in percent
for ti = 1 : n_targets
    pred_pool{ti} = 100 * reshape(pred(:, si_pool, ti, 1:n_bins), ...
        n_models, n_bins);
end

r_fine = cell(1, n_targets);
for ti = 1 : n_targets
    r_fine{ti} = linspace(max(releases{ti}(1) - 0.5, 1), ...
                          releases{ti}(end) + 0.5, n_fine);
end

gi_pool = find(strcmp(gauss.set_labels, set_labels{si_pool}), 1);
if isempty(gi_pool)
    error('No Gaussian fit "%s" in params_gauss.mat.', set_labels{si_pool});
end

dens_pool = cell(n_gauss, n_targets);

% Where each number line is CENTRED, expressed on the release axis, and the
% height of its own curve there.
%
% The centre is mu converted back to the release scale: mu itself on the
% linear line, exp(mu) on the logarithmic one.  On both, that is exactly
% where the drawn curve peaks — the density is a Gaussian in x, maximal at
% x = mu, and x is log(r) for the log line — so the marker lands on the top
% of its own curve rather than beside it.
%
% This is deliberately NOT the arithmetic mean of the predicted response
% distribution.  On the logarithmic line the two come apart: a Gaussian that
% is symmetric in log units is right-skewed once read back onto the release
% axis, so its arithmetic mean sits about 0.27 releases above its peak at
% target 3 and 0.38 above it at target 4.  Marking the arithmetic mean would
% put the line on the falling flank of the curve, which reads as an error.
% The arithmetic mean is still printed below, where the number can be
% compared with the centre instead of being mistaken for it.
centre_pool = NaN(n_gauss, n_targets);
centre_dens = NaN(n_gauss, n_targets);
mean_pool   = NaN(n_gauss, n_targets);    % printed, not drawn

for mi_g = 1 : n_gauss
    g  = gauss.fit_gauss(gauss_keep(mi_g), gi_pool);
    b  = g.x_opt(1);
    sg = max(g.x_opt(2), 1e-6);
    for ti = 1 : n_targets
        switch g.scale
            case 'linear'
                mu = targets_monk(ti) + b;
                x_int = 1 : R_MAX;        x_fine = r_fine{ti};
                centre_pool(mi_g, ti) = mu;
            case 'log'
                mu = log(targets_monk(ti)) + b;
                x_int = log(1 : R_MAX);   x_fine = log(r_fine{ti});
                centre_pool(mi_g, ti) = exp(mu);
            otherwise
                error('Unknown number-line scale "%s".', g.scale);
        end
        Z = sum(exp(-0.5 * ((x_int - mu) / sg) .^ 2));
        dens_pool{mi_g, ti} = 100 * exp(-0.5 * ((x_fine - mu) / sg) .^ 2) / Z;

        % At the centre the exponent is zero, so the curve is at 100/Z —
        % its peak.
        centre_dens(mi_g, ti) = 100 / Z;

        p_int               = exp(-0.5 * ((x_int - mu) / sg) .^ 2) / Z;
        mean_pool(mi_g, ti) = sum((1 : R_MAX) .* p_int);
    end
end

% The vertical lines on MS Fig. 4C are the centres, so print them — and print the
% arithmetic mean beside them, because on the logarithmic line the gap
% between the two IS the skew the figure is claiming, and it grows with the
% target in the same way the curve widens.
fprintf('Number line centre on the release axis (the vertical lines on fig9)\n');
fprintf('%-32s', 'Target:');
fprintf('%12d', targets_monk);  fprintf('\n');
for mi_g = 1 : n_gauss
    fprintf('%-32s', sprintf('%s:', model_disp{mi_g + 1}));
    fprintf('%12.3f', centre_pool(mi_g, :));  fprintf('\n');
end
fprintf('\nArithmetic mean of the same distribution (not drawn)\n');
for mi_g = 1 : n_gauss
    fprintf('%-32s', sprintf('%s:', model_disp{mi_g + 1}));
    fprintf('%12.3f', mean_pool(mi_g, :));  fprintf('\n');
end
fprintf('%-32s', 'mean - centre (skew on the');
fprintf('\n%-32s', '  release axis):');
fprintf('\n');
for mi_g = 1 : n_gauss
    fprintf('%-32s', sprintf('  %s:', model_disp{mi_g + 1}));
    fprintf('%12.3f', mean_pool(mi_g, :) - centre_pool(mi_g, :));  fprintf('\n');
end
fprintf('\n');

% The y-scale comes from the markers and from the smooth densities, which
% rise above their own markers wherever a peak falls between two whole
% numbers of releases.
y_max = 1.45 * max([cellfun(@(c) max(c(:)), pred_pool), ...
                    cellfun(@max, dens_pool(:)')]);

% This figure carries the qualitative argument of the comparison and is the
% one most often shown on its own, so its text and markers are set a few
% points above the shared sizes rather than inheriting them.  The other
% figures in this script are untouched.
fs9       = fs + 4;
fs9_title = fs9 - 2;
fs9_note  = fs9 - 2;
msz9      = 9.5;

fig9 = figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 21 17]);
fit_paper(fig9)
hold on

% The legend carries the three models and nothing else.  The vertical centre
% markers below are drawn in their own model's colour and rise to that
% model's own curve, so they read as part of the curve; giving them a fourth
% legend row made the key look like a list of four things being compared.
h_leg = gobjects(1, n_models);
for mi = 1 : n_models
    h_leg(mi) = plot(NaN, NaN, [mrk_model{mi} sty_model{mi}], ...
        'Color', clr_model_ind{mi}, 'MarkerFaceColor', 'w', 'MarkerSize', msz9, ...
        'LineWidth', lw, 'DisplayName', model_disp{mi});
end

for ti = 1 : n_targets
    % Centre of each number line, drawn from the axis up to the peak of that
    % model's own curve, so it reads as belonging to the curve rather than
    % floating over the whole panel.  Drawn first, so the densities and
    % markers sit on top of it.
    %
    % Only the number lines get one.  The stopping model has no number line
    % to be centred on, so marking it would suggest a comparison the figure
    % is not making.
    for mi_g = 1 : n_gauss
        mi = mi_g + 1;
        % Solid, which is the one style none of the three curves uses, so it
        % reads as an annotation rather than as a fourth model.  Thin,
        % because the two number lines sit close together (2.93 against 2.68
        % at target 3) and at full weight they compete with the curves they
        % are annotating.
        plot([centre_pool(mi_g, ti), centre_pool(mi_g, ti)], ...
            [0, centre_dens(mi_g, ti)], '-', 'Color', clr_model_ind{mi}, ...
            'LineWidth', lw * 0.7, 'HandleVisibility', 'off')
    end

    % Number lines: the fitted density itself, with markers only at the
    % whole numbers of releases the animals could actually produce.
    for mi_g = 1 : n_gauss
        mi  = mi_g + 1;
        clr = clr_model_ind{mi};
        plot(r_fine{ti}, dens_pool{mi_g, ti}, sty_model{mi}, ...
            'Color', clr, 'LineWidth', lw)
        plot(releases{ti}, pred_pool{ti}(mi, :), mrk_model{mi}, ...
            'LineStyle', 'none', 'Color', clr, 'MarkerFaceColor', 'w', ...
            'MarkerSize', msz9, 'LineWidth', lw)
    end

    % The stopping model is simulated and has no closed form on the release
    % axis, so it keeps joined markers.
    plot(releases{ti}, pred_pool{ti}(1, :), [mrk_model{1} sty_model{1}], ...
        'Color', clr_model_ind{1}, 'MarkerFaceColor', 'w', 'MarkerSize', msz9, ...
        'LineWidth', lw)
end

% Which family belongs to which target is read off its position, so label the
% families directly rather than doubling the length of the legend.
for ti = 1 : n_targets
    fam_max = max([max(pred_pool{ti}(:)), ...
                   cellfun(@max, reshape(dens_pool(:, ti), 1, []))]);
    text(targets_monk(ti), fam_max + 0.03 * y_max, ...
        sprintf('target %d', targets_monk(ti)), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', fs9_note, 'FontWeight', 'bold', 'Color', 'k')
end

xticks(x_all);  xlim([x_all(1) - 0.6, x_all(end) + 0.6]);  ylim([0, y_max])
xlabel('Number of releases', 'FontSize', fs9)
ylabel('Predicted response frequency [%]', 'FontSize', fs9)
% Two lines, as on figure 8: at the enlarged font the sentence is as wide as
% the whole canvas and the first and last words sat on the frame.
title({'Predicted response distributions,', 'both animals fitted together'}, ...
    'FontSize', fs9_title)
legend(h_leg, 'Location', 'northeast', 'Box', 'off', 'FontSize', fs9, ...
    'TextColor', 'k', 'NumColumns', 1)
apply_axis_style(gca, fs9, lw)

exportgraphics(fig9, fullfile(fig_dir, 'fig4C_model_fits.pdf'), ...
    'ContentType', 'vector')

% =========================================================================
% 10. MS Fig. 4E — cross-validation (only if cv_models_MS has been run)
% =========================================================================
cv_file = fullfile(results_dir, 'cv_results.mat');
if ~isfile(cv_file)
    fprintf(['No cv_results.mat found — run cv_models_MS.m for the ' ...
        'cross-validation panel.\n']);
    cv = struct([]);
else
    cv = load(cv_file);
end

if isempty(cv)
    % nothing to plot
elseif ~all(ismember(model_names, cv.model_names))
    missing = model_names(~ismember(model_names, cv.model_names));
    fprintf(['cv_results.mat is missing %s — re-run cv_models_MS.m\n' ...
        'for the cross-validation panel.\n'], strjoin(missing, ', '));
else
    % cv_models_MS.m scores more models than are compared here, so pick the
    % rows by NAME rather than assuming the two model lists line up.
    cv_rows = cellfun(@(nm) find(strcmp(cv.model_names, nm), 1), model_names);
    cv_ll   = cv.cv_ll(cv_rows, :, :);

    % Leave-one-session-out: fold k is session k, so each column below is one
    % held-out session, scored per trial so that long and short sessions
    % carry equal weight.
    ll_per_trial = cv_ll ./ reshape(cv.cv_n, 1, cv.n_reps, cv.n_folds);
    ll_flat = reshape(ll_per_trial, n_models, []);     % models x sessions
    cv_animal = repmat(cv.sess_animal(:)', 1, cv.n_reps);

    % Two reference points put the per-trial log-likelihood on an
    % interpretable scale.  The FLOOR is a model that spreads its mass evenly
    % over the five observed categories.  The CEILING is a model handed each
    % target's exact response frequencies — the saturated fit, 4 free
    % probabilities per target — so it is scored in sample and is therefore a
    % slightly optimistic bound on what any model could reach out of sample.
    % The cross-validation is run on the combined data, so the pooled
    % saturated fit is the relevant one.
    ll_floor   = log(1 / n_bins);
    ll_ceiling = -nll_sat(1) / n_trials(1);
    pct_range  = @(ll) 100 * (ll - ll_floor) / (ll_ceiling - ll_floor);

    % A CROSS-VALIDATED ceiling, one value per held-out session: the
    % empirical-frequency model of cv_pooled_vs_animal_MS.m (pooled arm),
    % which predicts each held-out session from the training sessions' raw
    % response frequencies.  Unlike ll_ceiling above it is scored exactly the
    % way the models are — held out, same trials, same per-trial units — so
    % model minus ceiling is a fair paired difference and can be tested.
    ll_ceil_sess = [];
    pva_pre = fullfile(results_dir, 'cv_pooled_vs_animal.mat');
    if isfile(pva_pre)
        Q  = load(pva_pre, 'cv2_ll', 'cv_n', 'model_names', 'sess_animal', 'n_folds');
        ci = find(strcmp(Q.model_names, 'Ceiling (empirical frequencies)'), 1);
        % Same folds in the same order is not assumed — it is checked against
        % the held-out trial counts before the two files are combined.
        if ~isempty(ci) && Q.n_folds == cv.n_folds && ...
                isequal(Q.cv_n(1, :), cv.cv_n(1, :))
            ll_ceil_sess = NaN(1, Q.n_folds);
            for k = 1 : Q.n_folds
                ll_ceil_sess(k) = Q.cv2_ll(1, ci, 1, k, Q.sess_animal(k)) / Q.cv_n(1, k);
            end
        else
            fprintf(['cv_pooled_vs_animal.mat does not line up with ' ...
                'cv_results.mat — skipping the vs-ceiling panel.\n']);
        end
    end
    have_ceil_sess = ~isempty(ll_ceil_sess) && all(isfinite(ll_ceil_sess));
    n_panels       = 2 + have_ceil_sess;

    fprintf('Cross-validation (leave-one-session-out, %d sessions)\n', cv.n_folds);
    fprintf('%-30s %14s %12s %14s %12s\n', ...
        'Model', 'LL per trial', 'SD (sess.)', 'geom. mean p', '% of range');
    fprintf('%s\n', repmat('-', 1, 86));
    fprintf('%-30s %14.4f %12s %14.4f %12.1f\n', ...
        'uniform over 5 categories', ll_floor, '—', exp(ll_floor), 0);
    for mi = 1 : n_models
        ll_mi = mean(ll_flat(mi, :));
        fprintf('%-30s %14.4f %12.4f %14.4f %12.1f\n', model_disp{mi}, ...
            ll_mi, std(ll_flat(mi, :)), exp(ll_mi), pct_range(ll_mi));
    end
    fprintf('%-30s %14.4f %12s %14.4f %12.1f\n', ...
        'ceiling (exact frequencies)', ll_ceiling, '—', exp(ll_ceiling), 100);
    fprintf(['\nThe SD column is the spread BETWEEN held-out sessions, not the ' ...
        'precision of\nthe mean.  It is an order of magnitude larger than the ' ...
        'gaps between models,\nwhich is why the models are compared within ' ...
        'session (middle panel of fig4E_crossvalidation.pdf).\n']);

    % Per-session correlation between models: Methods, "Per session scores
    % were highly correlated across models".  It is what justifies the
    % within-session paired test that follows.
    fprintf('\nPearson r between models across the %d held-out sessions:\n', ...
        size(ll_flat, 2));
    for mi = 1 : n_models - 1
        for mj = mi + 1 : n_models
            fprintf('  %-22s vs %-22s r = %.3f\n', model_disp{mi}, ...
                model_disp{mj}, corr(ll_flat(mi, :)', ll_flat(mj, :)'));
        end
    end

    fig10 = figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 15 * n_panels 11]);
    fit_paper(fig10)

    % ---- Left: held-out log-likelihood per trial, one point per session --
    %
    % The three models carry 2 parameters each, so they are drawn alike.  The
    % heavy bar is the mean over sessions and the whisker its SD ACROSS
    % sessions — the spread of the cloud, not the precision of the mean,
    % which is far smaller.
    %
    % The empirical CEILING leads the columns.  It is a per-session quantity
    % exactly like the models — the same held-out sessions, scored the same
    % way — so drawing it as a cloud rather than a line shows two things a
    % line cannot: that the achievable maximum itself varies from session to
    % session, and that the models' clouds overlap it.
    %
    % The FLOOR stays a line because it genuinely is one.  A model that puts
    % 1/5 on every category scores ln(1/5) on every trial whatever the animal
    % did, so its per-session value is identical in all 76 sessions (verified:
    % SD 4e-16).  It is the only exactly constant quantity in the panel.
    subplot(1, n_panels, 1);  hold on

    if have_ceil_sess
        col_y    = [ll_ceil_sess; ll_flat];
        col_clr  = [{[0.30 0.30 0.30]}, clr_model_ind];
        % Four columns in one panel leave less width per tick than the
        % two-line labels need, so "number line" breaks over two lines here.
        col_tick = {'Ceiling', 'Stopping', 'Linear\newlinenumber\newlineline', ...
                    'Log\newlinenumber\newlineline'};
    else
        % No cross-validated ceiling available: fall back to the in-sample
        % saturated value as a line, as before.
        col_y    = ll_flat;
        col_clr  = clr_model_ind;
        col_tick = model_tick;
        yline(ll_ceiling, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4);
        text(0.34, ll_ceiling, 'ceiling', 'FontSize', fs, ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
            'Color', 'k')
    end
    n_col = size(col_y, 1);

    yline(ll_floor, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4);
    text(0.34, ll_floor, 'floor', 'FontSize', fs, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'Color', 'k')

    % The summary is a filled dot with capped error bars, drawn the same way
    % as the median RT points on fig4 and fig5 so the two figures read alike.
    % It sits to the RIGHT of its cloud rather than in the middle of it: on
    % fig4 the dot IS the data, here there is a cloud behind it, and a dot
    % centred on the cloud would be buried in it.
    %
    % The whisker remains the SD ACROSS sessions, not the standard error of
    % the mean.  The two differ by a factor of sqrt(76) here, and it is the
    % between-session spread the panel exists to show — that the models'
    % clouds overlap each other and the ceiling.  The precision of the mean
    % is not the question on this panel; the paired within-session test in
    % the right panel is what carries the comparison.
    for ci = 1 : n_col
        m_ci = mean(col_y(ci, :));
        s_ci = std(col_y(ci, :));
        x_j  = ci + 0.20 * (rand(1, size(col_y, 2)) - 0.5);
        plot(x_j, col_y(ci, :), 'o', 'Color', col_clr{ci}, ...
            'MarkerFaceColor', 'w', 'MarkerSize', 3.5, 'LineWidth', 0.6)
        errorbar(ci + 0.32, m_ci, s_ci, s_ci, 'o', 'Color', col_clr{ci}, ...
            'MarkerFaceColor', col_clr{ci}, 'MarkerSize', 6, ...
            'LineWidth', lw, 'CapSize', 5)
    end
    xticks(1 : n_col);  xticklabels(col_tick);  xtickangle(0)
    xlim([0.30, n_col + 0.6])
    ylim([min([col_y(:); ll_floor]) - 0.03, max(col_y(:)) + 0.02])
    ylabel('Held-out log-lik. per trial', 'FontSize', fs)
    % Shortened from "one held-out session": at this panel width the longer
    % title ran into the middle panel's y label.
    title('One point = one session (dot = mean \pm SD)', 'FontSize', fs_title)
    style_ax(gca)

    % ---- Right: paired difference within session -------------------------
    %
    % The between-session variation on the left is common to all models, so
    % differencing within session removes it.  This is the panel the reported
    % statistic belongs to: n = one per held-out session, no repetitions.
    subplot(1, n_panels, 2);  hold on
    mk_mrk = {'o', 's'};
    d_all  = ll_flat(1, :) - ll_flat(2 : n_models, :);
    y_lab  = max(d_all(:)) + 0.06 * range(d_all(:));

    % The same numbers the panel labels carry, printed so they can be quoted
    % without reading them off the figure.  These are the values reported in
    % Results ("the stopping model predicted most held-out sessions better
    % than the linear number line ...").
    fprintf('\nPaired within-session comparison (%d held-out sessions):\n', ...
        size(ll_flat, 2));
    fprintf('%-40s %12s %14s %12s\n', ...
        'Comparison', 'Stop wins', 'mean dLL/tr', 'p');
    fprintf('%s\n', repmat('-', 1, 82));

    for mi = 2 : n_models
        d = ll_flat(1, :) - ll_flat(mi, :);
        for mk = 1 : max(cv_animal)
            sel = cv_animal == mk;
            x_j = (mi - 1) + 0.30 * (mk - 1.5) + 0.10 * (rand(1, sum(sel)) - 0.5);
            plot(x_j, d(sel), mk_mrk{mk}, ...
                'Color', lighten(clr_model_ind{mi}, shade_animal(mk)), ...
                'MarkerFaceColor', 'w', 'MarkerSize', 4, 'LineWidth', 0.8, ...
                'HandleVisibility', 'off')
        end
        plot((mi - 1) + [-0.42 0.42], mean(d) * [1 1], '-', ...
            'Color', clr_model_ind{mi}, 'LineWidth', 2.5, ...
            'HandleVisibility', 'off')
        if exist('signrank', 'file'), pv = signrank(d); else, pv = NaN; end
        text(mi - 1, y_lab, sprintf('%d/%d, p = %.0e', ...
            sum(d > 0), numel(d), pv), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontSize', fs_note, 'Color', 'k')
        fprintf('%-40s %5d/%-6d %14.5f %12.3g\n', ...
            sprintf('Stopping vs %s', model_disp{mi}), ...
            sum(d > 0), numel(d), mean(d), pv);
    end
    fprintf('\n');
    % The two animals sit at their own x offset within each model's column.
    % Lightness names them, as everywhere else, with marker shape repeating it
    % so the panel still reads in greyscale.  The legend keys are neutral, so
    % that hue keeps meaning model and nothing else.
    for mk = 1 : numel(mk_mrk)
        plot(NaN, NaN, mk_mrk{mk}, 'MarkerFaceColor', 'w', ...
            'Color', lighten([0.15 0.15 0.15], shade_animal(mk)), ...
            'MarkerSize', 4, 'LineWidth', 0.8, 'DisplayName', monk_lbl{mk})
    end
    yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off')
    % Headroom for the counts and p-values, plus a single legend row above them.
    % Headroom above y_lab holds the single-column legend; it has to clear the
    % per-comparison count labels sitting at y_lab.
    ylim([min(d_all(:)) - 0.02 * range(d_all(:)), y_lab + 0.45 * range(d_all(:))])
    xticks(1 : n_models - 1);  xticklabels(model_tick(2:end));  xtickangle(0)
    xlim([0.5, n_models - 0.5])
    ylabel('\DeltaLL per trial  (Stopping - number line)', 'FontSize', fs)
    title('Above 0 = stopping model better', 'FontSize', fs_title)
    legend('Location', 'northeast', 'Box', 'off', 'FontSize', fs, 'TextColor', 'k', ...
        'NumColumns', 1)
    style_ax(gca)

    % ---- Third: each model against the cross-validated ceiling -----------
    %
    % The left panel shows where the models sit; this one shows how far each
    % of them is from the best any model could do on the SAME held-out
    % session.  Because the ceiling is cross-validated it is a paired
    % per-session difference, so it can be tested and its spread is
    % meaningful.  Zero means a model predicted the held-out session as well
    % as its own training frequencies did; everything sits below.
    if have_ceil_sess
        subplot(1, n_panels, 3);  hold on

        d_ceil = ll_flat - ll_ceil_sess;          % models x sessions, <= 0
        y_lab2 = max(d_ceil(:)) + 0.06 * range(d_ceil(:));

        yline(0, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4, ...
            'HandleVisibility', 'off')
        text(0.44, 0, 'ceiling', 'FontSize', fs, 'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'bottom', 'Color', 'k')

        % The two animals are POOLED here: one cloud per model over all 76
        % held-out sessions, no per-animal offset, shape or key.
        %
        % That is the claim this project makes everywhere else — the animals
        % are fitted together, cv_pooled_vs_animal_MS.m shows that splitting
        % them does not predict held-out sessions any better, and the
        % per-animal breakdown is presented on purpose in the printed tables
        % (MS Fig. 4D and section 11 below).
        % Splitting them again on the panel that goes into the manuscript
        % would invite the reader to look for a difference the analysis has
        % already argued is not there.
        %
        % Summary drawn as a dot with capped error bars, matching the
        % left-hand panel and the median-RT points on MS Fig. 3E/4F: mean over
        % all 76 sessions, whisker the SD across them.  It sits to the right
        % of its own cloud so it is not buried in it.
        for mi = 1 : n_models
            d   = d_ceil(mi, :);
            x_j = mi + 0.26 * (rand(1, numel(d)) - 0.5);
            plot(x_j, d, 'o', 'Color', clr_model_ind{mi}, ...
                'MarkerFaceColor', 'w', 'MarkerSize', 4, 'LineWidth', 0.8, ...
                'HandleVisibility', 'off')
            errorbar(mi + 0.36, mean(d), std(d), std(d), 'o', ...
                'Color', clr_model_ind{mi}, 'MarkerFaceColor', clr_model_ind{mi}, ...
                'MarkerSize', 6, 'LineWidth', lw, 'CapSize', 5, ...
                'HandleVisibility', 'off')
            text(mi, y_lab2, sprintf('%.3f', mean(d)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'FontSize', fs_note, 'Color', 'k')
        end

        xticks(1 : n_models);  xticklabels(model_tick);  xtickangle(0)
        xlim([0.4, n_models + 0.7])
        ylim([min(d_ceil(:)) - 0.08 * range(d_ceil(:)), ...
              y_lab2 + 0.10 * range(d_ceil(:))])
        ylabel('\DeltaLL per trial  (model - ceiling)', 'FontSize', fs)
        title('Distance from the held-out ceiling', 'FontSize', fs_title)
        style_ax(gca)

        fprintf(['\nDistance from the cross-validated ceiling ' ...
            '(fig4E_crossvalidation.pdf, third panel)\n']);
        fprintf('The ceiling predicts each held-out session from the training\n');
        fprintf('sessions'' response frequencies; it averages %.4f per trial.\n', ...
            mean(ll_ceil_sess));
        fprintf('%-22s %8s %10s %10s %12s\n', ...
            'Model', 'animal', 'mean dLL', 'SD (sess.)', 'sess. above');
        fprintf('%s\n', repmat('-', 1, 66));
        for mi = 1 : n_models
            d = d_ceil(mi, :);
            fprintf('%-22s %8s %10.4f %10.4f %8d/%d\n', model_disp{mi}, ...
                'both', mean(d), std(d), sum(d > 0), numel(d));
            for mk = 1 : max(cv_animal)
                dm = d(cv_animal == mk);
                fprintf('%-22s %8s %10.4f %10.4f %8d/%d\n', '', ...
                    monk_lbl{mk}, mean(dm), std(dm), sum(dm > 0), numel(dm));
            end
        end
    end

    exportgraphics(fig10, fullfile(fig_dir, 'fig4E_crossvalidation.pdf'), ...
        'ContentType', 'vector')
end

% =========================================================================
% 11. Stopping model vs the two shared number lines, under per-animal fits
%
%     Held-out dLL = LL(stopping) - LL(number line), one value per held-out
%     session.  Above zero means the stopping model predicted that session
%     better.  Only the three 2-parameter models appear, so the comparison is
%     at matched complexity throughout.
%
%     Both arms of cv_pooled_vs_animal_MS.m are reported, because "per monkey"
%     can mean two different things and they answer different questions:
%       fitted on combined data     models fitted on the COMBINED training
%                                   sessions, then scored separately on each
%                                   animal's held-out trials — does the shared
%                                   fit serve both animals equally?
%       fitted on that animal only  models fitted on that animal's OWN
%                                   training sessions — does the ranking hold
%                                   within each animal independently?
%
%     The "both" row of the second arm is the statistic Results quotes
%     ("Separate fits did not change the model ranking ..."): every session
%     scored under its own animal's fit, pooled back into one paired test
%     over all sessions.
%
%     Prints only — no manuscript panel is built from this.  Needs
%     results/cv_pooled_vs_animal.mat (run cv_pooled_vs_animal_MS.m).
% =========================================================================
pva_file = fullfile(results_dir, 'cv_pooled_vs_animal.mat');
if ~isfile(pva_file)
    fprintf(['No cv_pooled_vs_animal.mat found — run cv_pooled_vs_animal_MS.m ' ...
        'for the\nper-animal comparison.\n']);
else
    pva = load(pva_file);

    % Models to contrast against the stopping model: the two shared number
    % lines, located by name so the indices cannot silently drift.  cmp_names
    % are the stored keys; cmp_disp are the labels shown to a reader.
    cmp_names = {'Gaussian linear (shared)', 'Gaussian log (shared)'};
    cmp_disp  = {'Linear number line', 'Log number line'};
    cmp_idx   = cellfun(@(nm) find(strcmp(pva.model_names, nm), 1), cmp_names);
    n_cmp     = numel(cmp_idx);
    n_animals = numel(pva.monkeys);
    n_arms    = numel(pva.arm_names);
    arm_title = {'fitted on combined data', 'fitted on that animal only'};

    % ---- Printed table ---------------------------------------------------
    fprintf(['Held-out dLL, stopping model minus shared number line, per animal\n' ...
        '(positive = the stopping model predicts that animal better)\n\n']);
    fprintf('%-26s %-8s %-26s %10s %14s %10s\n', ...
        'Parameters', 'Animal', 'Comparison', 'Stop wins', 'mean dLL/tr', 'p');
    fprintf('%s\n', repmat('-', 1, 100));
    % Held-out sessions belong to one animal each, so the other animal's
    % entries are NaN and drop out here.  Each session's difference is divided
    % by its own trial count, the same currency as fig4E_crossvalidation.pdf.
    n_held = reshape(pva.cv_n, 1, []);
    d_all  = cell(n_arms, n_animals, n_cmp);
    for a = 1 : n_arms
        for mk = 1 : n_animals
            for ci = 1 : n_cmp
                d  = reshape(pva.cv2_ll(a, 1, :, :, mk) - ...
                             pva.cv2_ll(a, cmp_idx(ci), :, :, mk), 1, []);
                ok = ~isnan(d);
                d  = d(ok) ./ n_held(ok);
                d_all{a, mk, ci} = d;
                if exist('signrank', 'file') && ~isempty(d)
                    pv = signrank(d);
                else
                    pv = NaN;
                end
                fprintf('%-26s %-8s %-26s %4d/%-5d %14.5f %10.3g\n', ...
                    arm_title{a}, pva.monkeys{mk}, cmp_disp{ci}, ...
                    sum(d > 0), numel(d), mean(d), pv);
            end
        end

        % Both animals pooled back together: each session still scored under
        % the arm's own fit, but tested as one set of paired differences over
        % all sessions.  This is the row Results quotes.
        for ci = 1 : n_cmp
            d = [d_all{a, :, ci}];
            if exist('signrank', 'file') && ~isempty(d)
                pv = signrank(d);
            else
                pv = NaN;
            end
            fprintf('%-26s %-8s %-26s %4d/%-5d %14.5f %10.3g\n', ...
                arm_title{a}, 'both', cmp_disp{ci}, ...
                sum(d > 0), numel(d), mean(d), pv);
        end
    end
    fprintf('\n');

end

fprintf('Figures saved to %s\n', fig_dir);


% =========================================================================
% Local functions
% =========================================================================
function apply_axis_style(ax, fs, lw)
% Tick labels at the same size as the axis labels, and every piece of text on
% the axes in true black.  MATLAB defaults the axis colour to [0.15 0.15 0.15]
% and the title to the same grey, which prints noticeably lighter than the
% data; journals generally want black.
set(ax, 'TickDir', 'out', 'LineWidth', lw, 'FontSize', fs, 'Box', 'off', ...
    'XColor', 'k', 'YColor', 'k', 'ZColor', 'k')

% Labels and title inherit size from their own set() calls, so only the colour
% is forced here.  Legends are children of the parent figure, not the axes,
% and are handled where they are created.
lbl = [ax.Title, ax.XLabel, ax.YLabel];
set(lbl(isgraphics(lbl)), 'Color', 'k')
end
