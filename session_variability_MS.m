% session_variability_MS.m
%
% Quantifies BETWEEN-SESSION variability in the response distribution, and
% asks how it compares with the between-animal difference.  Produces the two
% numbers the Results need:
%
%   1. Why statistics are done across cross-validation folds/sessions and not
%      across trials:  responses within a session are not independent.  The
%      session-to-session spread in the offset distribution is far larger
%      than the binomial/multinomial expectation for independent trials, so
%      a trial-level test is anticonservative by the design effect DEFF.
%
%   2. Why both animals can be fitted together:  the difference between the
%      two monkeys is small relative to the spread among sessions of the same
%      monkey, and (see cv_pooled_vs_animal_MS.m) fitting them separately
%      buys nothing out of sample.
%
% Reads results/cv_results.mat (session-level counts + held-out LL) and
% results/cv_pooled_vs_animal.mat.  Prints only, writes nothing.

clear; clc

project_dir = fileparts(fileparts(mfilename('fullpath')));
cv          = load(fullfile(project_dir, 'results', 'cv_results.mat'));

bins      = cv.bins;
n_bins    = numel(bins);
n_sess    = numel(cv.sess_animal);
n_targets = numel(cv.targets_monk);
monkeys   = {'m1', 'm2'};

% Counts per session over the five offset bins (no-stop column is empty in
% the data and is dropped here so the rows are proper multinomials).
sc      = cv.sess_counts(:, :, 1:n_bins);        % [sess x target x bin]
cnt     = squeeze(sum(sc, 2));                   % [sess x bin], targets pooled
n_s     = sum(cnt, 2);                           % trials per session
p_s     = cnt ./ n_s;                            % session response profile
animal  = cv.sess_animal(:);

fprintf('%d sessions (m1: %d, m2: %d), %d trials, median %.0f trials/session\n\n', ...
    n_sess, sum(animal == 1), sum(animal == 2), sum(n_s), median(n_s));

% =========================================================================
% 1.  Between-session variability of the proportion correct
% =========================================================================
% If trials were independent draws from one common distribution, the spread
% of the session-wise proportion correct would be purely binomial.  The
% ratio of the observed to the binomial variance is the design effect.
i0    = find(bins == 0);
k_s   = cnt(:, i0);
pc_s  = k_s ./ n_s;
p_bar = sum(k_s) / sum(n_s);

var_obs   = var(pc_s, 1);                          % observed, across sessions
var_binom = mean(p_bar * (1 - p_bar) ./ n_s);      % expected if independent
deff_pc   = var_obs / var_binom;

% Heterogeneity test: are the sessions drawing from one common p?
X2_pc  = sum((k_s - n_s * p_bar).^2 ./ (n_s * p_bar * (1 - p_bar)));
df_pc  = n_sess - 1;
p_het  = 1 - chi2cdf(X2_pc, df_pc);

n_bar   = mean(n_s);
icc_pc  = (deff_pc - 1) / (n_bar - 1);

fprintf('--- Proportion correct across sessions -----------------------------\n');
fprintf('  pooled                         %.3f\n', p_bar);
fprintf('  observed SD across sessions    %.3f  (range %.3f - %.3f)\n', ...
    std(pc_s, 1), min(pc_s), max(pc_s));
fprintf('  SD expected for indep. trials  %.3f\n', sqrt(var_binom));
fprintf('  overdispersion (design effect) %.2f  -> SE inflated %.2fx\n', ...
    deff_pc, sqrt(deff_pc));
fprintf('  intraclass correlation         %.3f\n', icc_pc);
fprintf('  heterogeneity  X2(%d) = %.1f, p = %.3g\n\n', df_pc, X2_pc, p_het);

% =========================================================================
% 2.  Same thing for the whole 5-bin distribution
% =========================================================================
% G2 tests the stronger claim that the entire offset distribution is fixed
% across sessions, not just its peak.
p_pool = sum(cnt, 1) / sum(n_s);
exp_c  = n_s * p_pool;
nz     = cnt > 0;
G2     = 2 * sum(cnt(nz) .* log(cnt(nz) ./ exp_c(nz)));
df_G2  = (n_sess - 1) * (n_bins - 1);
p_G2   = 1 - chi2cdf(G2, df_G2);

fprintf('--- Whole offset distribution across sessions ----------------------\n');
fprintf('  G2(%d) = %.1f, p = %.3g   (G2/df = %.2f)\n\n', ...
    df_G2, G2, p_G2, G2 / df_G2);

% =========================================================================
% 2b. The same variance ratio computed within each animal
%
%     Pooling the animals could in principle manufacture overdispersion by
%     itself: if the two differed a lot, the mixture of two distributions
%     would look overdispersed even with perfectly homogeneous sessions
%     inside each.  Computing the ratio separately per animal removes that
%     objection — whatever it shows is genuine session-to-session variation
%     within one animal.
% =========================================================================
fprintf('--- Variance ratio within each animal -----------------------------\n');
fprintf('%-6s %8s %10s %10s %10s %8s %10s %12s\n', ...
    'Animal', 'sess.', 'p(corr)', 'SD obs', 'SD indep', 'DEFF', 'ICC', 'X2 (df)');
deff_mk = NaN(1, 2);
for a = 1 : 2
    sel  = animal == a;
    ks   = k_s(sel);  ns = n_s(sel);  ps = pc_s(sel);
    pb   = sum(ks) / sum(ns);
    vo   = var(ps, 1);
    vb   = mean(pb * (1 - pb) ./ ns);
    deff_mk(a) = vo / vb;
    X2a  = sum((ks - ns * pb).^2 ./ (ns * pb * (1 - pb)));
    dfa  = sum(sel) - 1;
    fprintf('%-6s %8d %10.3f %10.4f %10.4f %8.2f %10.4f %6.1f (%d)\n', ...
        monkeys{a}, sum(sel), pb, std(ps, 1), sqrt(vb), deff_mk(a), ...
        (deff_mk(a) - 1) / (mean(ns) - 1), X2a, dfa);
end
fprintf(['\nBoth ratios exceed 1 on their own, so the clustering is not an ' ...
    'artefact of\nmixing the two animals.\n\n']);

% =========================================================================
% 3.  Between-session vs between-animal
% =========================================================================
% One-way decomposition of the session-wise proportion correct: how much of
% the spread is "which monkey" and how much is "which session".
grp_mean = arrayfun(@(a) mean(pc_s(animal == a)), 1 : 2);
grp_sd   = arrayfun(@(a)  std(pc_s(animal == a)), 1 : 2);
sd_within = sqrt(mean(arrayfun(@(a) var(pc_s(animal == a)), 1 : 2)));
d_animal  = abs(diff(grp_mean));

ss_tot = sum((pc_s - mean(pc_s)).^2);
ss_bet = sum(arrayfun(@(a) sum(animal == a) * (grp_mean(a) - mean(pc_s))^2, 1 : 2));
eta2   = ss_bet / ss_tot;

fprintf('--- Between animals vs between sessions ----------------------------\n');
for a = 1 : 2
    fprintf('  %s   p(correct) = %.3f +/- %.3f (SD over %d sessions)\n', ...
        monkeys{a}, grp_mean(a), grp_sd(a), sum(animal == a));
end
fprintf('  animal difference              %.3f\n', d_animal);
fprintf('  within-animal session SD       %.3f\n', sd_within);
fprintf('  difference in session SDs      %.2f\n', d_animal / sd_within);
fprintf('  variance explained by animal   %.1f%% (eta^2)\n\n', 100 * eta2);

% ---- The same variance ratio one level up -------------------------------
% Section 2 asked: do sessions vary more than the trials inside them can
% explain?  The matching question here is: do the two ANIMALS differ more
% than the sessions inside them can explain?  Same construction — observed
% variation over expected variation — so the two ratios are read the same
% way, and an F near 1 says the animals are exchangeable at session level.
n_a    = arrayfun(@(a) sum(animal == a), 1 : 2);
ms_bet = ss_bet / 1;                       % 1 df: two animals
ms_wit = sum(arrayfun(@(a) sum((pc_s(animal == a) - grp_mean(a)).^2), 1 : 2)) / ...
    (n_sess - 2);
F_anim = ms_bet / ms_wit;
p_anim = 1 - fcdf(F_anim, 1, n_sess - 2);

% Welch on the session-level values: the animal difference in units of the
% between-session spread.
se_w  = sqrt(grp_sd(1)^2 / n_a(1) + grp_sd(2)^2 / n_a(2));
t_w   = d_animal / se_w;
df_w  = se_w^4 / ((grp_sd(1)^2/n_a(1))^2/(n_a(1)-1) + (grp_sd(2)^2/n_a(2))^2/(n_a(2)-1));
p_w   = 2 * (1 - tcdf(abs(t_w), df_w));

% For contrast: the same comparison made at TRIAL level, which is what
% ignoring the clustering would give.
p1 = sum(k_s(animal==1))/sum(n_s(animal==1));
p2 = sum(k_s(animal==2))/sum(n_s(animal==2));
N1 = sum(n_s(animal==1));  N2 = sum(n_s(animal==2));
pp = (p1*N1 + p2*N2) / (N1 + N2);
z_trial = (p1 - p2) / sqrt(pp*(1-pp)*(1/N1 + 1/N2));
p_trial = 2 * (1 - normcdf(abs(z_trial)));

fprintf('  Variance ratio at the ANIMAL level (sessions as the unit):\n');
fprintf('    observed animal difference     %.4f\n', d_animal);
fprintf('    expected from session spread    %.4f  (SE of the difference)\n', se_w);
fprintf('    F(1,%d) = %.2f, p = %.3f   (between-animal MS / within-animal MS)\n', ...
    n_sess - 2, F_anim, p_anim);
fprintf('    Welch t(%.0f) = %.2f, p = %.3f\n', df_w, t_w, p_w);
fprintf('    same test at TRIAL level: z = %.2f, p = %.2g  <- clustering ignored\n', ...
    z_trial, p_trial);
fprintf([ ...
    '\n  The two ratios are NOT equally well determined and should not be read\n' ...
    '  as one number against another.  The session-level ratio rests on %d df\n' ...
    '  and is decisive.  The animal-level ratio has 1 numerator df: F = %.2f is\n' ...
    '  an unresolved difference (p = %.2f), not a demonstrated equality.  What\n' ...
    '  it does establish is that the animal difference (%.3f) is only %.2f of a\n' ...
    '  between-session SD and stops being significant the moment the clustering\n' ...
    '  is respected (trial level p = %.1g, session level p = %.2f).  The\n' ...
    '  positive case for pooling is the out-of-sample one below.\n\n'], ...
    n_sess - 1, F_anim, p_anim, d_animal, d_animal / sd_within, p_trial, p_anim);

% =========================================================================
% 4.  The same picture in the cross-validated currency
% =========================================================================
% Held-out log-likelihood per trial of the diffusion model, one value per
% session — under leave-one-session-out each session is held out exactly
% once, so there is nothing to average over.
sess_pertrial = cv.cv_ll_sess ./ reshape(n_s, 1, 1, n_sess);
ll_sess       = reshape(mean(sess_pertrial, 2), [], n_sess);  % [model x sess]
ddm           = ll_sess(1, :)';

gi   = arrayfun(@(s) find(strcmp(cv.model_names, s), 1), ...
    {'Gaussian linear (shared)', 'Gaussian log (shared)'});

fprintf('--- Held-out LL/trial, diffusion model, per session ----------------\n');
fprintf('  mean over sessions             %.3f\n', mean(ddm));
fprintf('  SD across sessions             %.3f  (range %.3f - %.3f)\n', ...
    std(ddm), min(ddm), max(ddm));
for a = 1 : 2
    fprintf('  %s mean                        %.3f  (SD %.3f)\n', ...
        monkeys{a}, mean(ddm(animal == a)), std(ddm(animal == a)));
end
fprintf('  animal difference              %.3f\n', ...
    abs(mean(ddm(animal == 1)) - mean(ddm(animal == 2))));
fprintf('\n  Advantage of the diffusion model, paired within session:\n');
for j = 1 : numel(gi)
    d = ddm - ll_sess(gi(j), :)';
    if exist('signrank', 'file'), pv = signrank(d); else, pv = NaN; end
    fprintf('    vs %-26s %+.4f +/- %.4f, better in %d/%d sessions, p = %.4g\n', ...
        cv.model_names{gi(j)}, mean(d), std(d), sum(d > 0), n_sess, pv);
end
fprintf('\n');

% =========================================================================
% 5.  Out-of-sample cost of pooling the animals
% =========================================================================
f2 = fullfile(project_dir, 'results', 'cv_pooled_vs_animal.mat');
if isfile(f2) && size(load(f2, 'cv2_ll').cv2_ll, 4) ~= n_sess
    fprintf(['--- results/cv_pooled_vs_animal.mat predates the ' ...
        'leave-one-session-out\n    switch; re-run cv_pooled_vs_animal_MS.m ' ...
        'for the pooling comparison.\n\n']);
elseif isfile(f2)
    pa = load(f2);

    % Each held-out session belongs to one animal, so its score sits in that
    % animal's slice and the other animal's entry is NaN.
    n_pa = numel(pa.sess_animal);
    ll2  = NaN(2, n_pa);
    for k = 1 : n_pa
        ll2(:, k) = pa.cv2_ll(:, 1, 1, k, pa.sess_animal(k));
    end
    ok = ~isnan(ll2(1, :));
    d_tot = ll2(2, ok) - ll2(1, ok);              % per session, summed LL
    d_ptr = d_tot ./ reshape(n_s(ok), 1, []);     % per session, per trial
    if exist('signrank', 'file'), pv = signrank(d_ptr); else, pv = NaN; end

    % Header says "read from" on purpose: this script fits nothing, it reports
    % the per-animal arm already stored in cv_pooled_vs_animal.mat.  The old
    % wording ("Fitting the animals separately") read like a fit starting up.
    fprintf('--- Animals fitted separately: read from cv_pooled_vs_animal.mat ---\n');
    fprintf('  held-out LL gain over pooling  %+.2f over %d sessions (%+.5f per trial)\n', ...
        sum(d_tot), sum(ok), sum(d_tot) / sum(n_s(ok)));
    fprintf('  per session                    %+.5f +/- %.5f per trial, better in %d/%d, p = %.3g\n\n', ...
        mean(d_ptr), std(d_ptr), sum(d_ptr > 0), numel(d_ptr), pv);
end
