% compare_rt_fit_MS.m
%
% Compares RT goodness-of-fit (G² on quantile bins) between two parameter
% sets fitted with different objectives:
%   params_perf.mat — fitted to response frequencies (fit_diffusion_mle_MS)
%   params_rt.mat   — fitted to RT distributions     (fit_diffusion_rt_MS)
%
% Both sets are evaluated with the same G² metric on the same monkey RT data.
% Run the two fitter scripts first to generate the .mat files in results/.
%
% Prints MS Table 1 (G² per offset category for both fits, ΔG² and its
% stratified-bootstrap 95% CI) and the G² totals quoted in Results,
% "we refitted parameters of the stopping model directly to the monkeys'
% within step RT distributions".  Its three figures are diagnostics, not
% manuscript panels; they go to figures/diagnostics/.

clear; clc;
rng(42);

% helpers/ holds run_diffusion_sim.m and compute_g2.m.  Added here relative
% to this file so the script runs from any working directory without the
% reviewer having to set up a path.
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'helpers'));

% =========================================================================
% 1.  Fixed model parameters — must match both fitters exactly
% =========================================================================
noise             = 0.1;
dt                = 0.01;
step_dur          = 1;
bound             = 1;
RT_cutoff         = 1100;          % RT-fitter cutoff used as shared reference
resp_time_scaling = [200, 500, 800, 1100, 200, 200, 800, 400];

n_steps_rel = 7;
period      = step_dur;
t_vec       = (0 : dt : n_steps_rel * period)';
n_t         = numel(t_vec);

% =========================================================================
% 2.  Load monkey RT data (identical procedure to fit_diffusion_rt_MS)
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
                monk_off_rt = [monk_off_rt; zeros(sum(valid),1)];  
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
% 3.  Observed RT quantile boundaries (fixed reference for G²)
% =========================================================================
bins               = -2 : 2;
n_bins             = numel(bins);
q_probs            = [.10, .30, .50, .70, .90];
min_trials_per_cat = 20;

obs_quantiles = NaN(n_bins, numel(q_probs));
obs_n         = zeros(n_bins, 1);
for b = 1 : n_bins
    cat_rt   = monk_rt(monk_off_rt == bins(b) & ~isnan(monk_rt));
    obs_n(b) = numel(cat_rt);
    if obs_n(b) >= min_trials_per_cat
        obs_quantiles(b,:) = quantile(cat_rt, q_probs);
    end
end

% =========================================================================
% 4.  Load fitted parameter sets
% =========================================================================
results_dir = fullfile(project_dir, 'results');
perf = load(fullfile(results_dir, 'params_perf.mat'));
rt   = load(fullfile(results_dir, 'params_rt.mat'));

fprintf('Performance-fit parameters (%s):\n', strjoin(perf.param_names, ', '));
for i = 1 : numel(perf.param_names)
    fprintf('  %-14s = %.4f\n', perf.param_names{i}, perf.x_opt(i));
end
fprintf('\nRT-fit parameters (%s):\n', strjoin(rt.param_names, ', '));
for i = 1 : numel(rt.param_names)
    fprintf('  %-14s = %.4f\n', rt.param_names{i}, rt.x_opt(i));
end
fprintf('\n');

% =========================================================================
% 5.  Simulate both models (large n, same seed — fair comparison)
% =========================================================================
n_eval = 60000;

[sim_rt_perf, sim_off_perf] = run_diffusion_sim(perf.x_opt, perf.fix_drift, bound, ...
    n_eval, t_vec, n_t, dt, noise, period, resp_time_scaling);

rng(99)
[sim_rt_rt, sim_off_rt] = run_diffusion_sim(rt.x_opt, rt.fix_drift, bound, ...
    n_eval, t_vec, n_t, dt, noise, period, resp_time_scaling);

% =========================================================================
% 6.  G² per offset category and total
% =========================================================================
obs_bin_p   = diff([0, q_probs, 1]);   % [.10 .20 .20 .20 .20 .10]
g2_perf_cat = NaN(1, n_bins);
g2_rt_cat   = NaN(1, n_bins);

for b = 1 : n_bins
    if obs_n(b) < min_trials_per_cat || any(isnan(obs_quantiles(b,:))), continue; end
    edges = [-inf, obs_quantiles(b,:), inf];

    sim_cat = sim_rt_perf(sim_off_perf == bins(b) & ~isnan(sim_rt_perf));
    if numel(sim_cat) >= 10
        pred_p = histcounts(sim_cat, edges) / numel(sim_cat);
        pred_p = max(pred_p, 1e-10);
        g2_perf_cat(b) = 2 * obs_n(b) * sum(obs_bin_p .* log(obs_bin_p ./ pred_p));
    end

    sim_cat = sim_rt_rt(sim_off_rt == bins(b) & ~isnan(sim_rt_rt));
    if numel(sim_cat) >= 10
        pred_p = histcounts(sim_cat, edges) / numel(sim_cat);
        pred_p = max(pred_p, 1e-10);
        g2_rt_cat(b) = 2 * obs_n(b) * sum(obs_bin_p .* log(obs_bin_p ./ pred_p));
    end
end

g2_perf_total = sum(g2_perf_cat, 'omitnan');
g2_rt_total   = sum(g2_rt_cat,   'omitnan');

% =========================================================================
% 6b. Stratified bootstrap: significance of ΔG² = G²(perf) − G²(RT)
%
%     Monkey RT data are resampled with replacement within each offset
%     category (stratified), shifting the quantile boundaries used to
%     compute G².  The pre-computed model simulations (sim_rt_perf /
%     sim_rt_rt) are reused in every replicate — no re-simulation needed.
%     This preserves the positive correlation between the two G² values
%     (same boundary shifts affect both), making the paired test sensitive.
%
%     p-value: fraction of replicates in which ΔG²_total ≤ 0, i.e. the
%     perf-fit is at least as good as the RT-fit on RTs.
% =========================================================================
B   = 1000;
rng(7)
dg2_boot = NaN(B, n_bins + 1);   % columns: per-category ΔG², then total

for boot = 1 : B
    g2_p_b = NaN(1, n_bins);
    g2_r_b = NaN(1, n_bins);

    for b = 1 : n_bins
        cat_rt = monk_rt(monk_off_rt == bins(b) & ~isnan(monk_rt));
        if numel(cat_rt) < min_trials_per_cat, continue; end

        boot_rt = cat_rt(randi(numel(cat_rt), numel(cat_rt), 1));
        edges   = [-inf, quantile(boot_rt, q_probs), inf];

        sim_cat = sim_rt_perf(sim_off_perf == bins(b) & ~isnan(sim_rt_perf));
        if numel(sim_cat) >= 10
            pred_p = histcounts(sim_cat, edges) / numel(sim_cat);
            pred_p = max(pred_p, 1e-10);
            g2_p_b(b) = 2 * obs_n(b) * sum(obs_bin_p .* log(obs_bin_p ./ pred_p));
        end

        sim_cat = sim_rt_rt(sim_off_rt == bins(b) & ~isnan(sim_rt_rt));
        if numel(sim_cat) >= 10
            pred_p = histcounts(sim_cat, edges) / numel(sim_cat);
            pred_p = max(pred_p, 1e-10);
            g2_r_b(b) = 2 * obs_n(b) * sum(obs_bin_p .* log(obs_bin_p ./ pred_p));
        end
    end

    dg2_boot(boot, 1:n_bins) = g2_p_b - g2_r_b;
    dg2_boot(boot, end)      = sum(g2_p_b - g2_r_b, 'omitnan');
end

dg2_obs = [g2_perf_cat - g2_rt_cat, g2_perf_total - g2_rt_total];
ci_lo   = quantile(dg2_boot, 0.025);
ci_hi   = quantile(dg2_boot, 0.975);
p_vals  = mean(dg2_boot(:, 1:n_bins) <= 0);   % per-bin one-sided p-values
p_total = mean(dg2_boot(:, end) <= 0);

% =========================================================================
% 7.  Print summary table with per-bin bootstrap p-values
% =========================================================================
cat_labels = {'-2', '-1', '0 (corr)', '+1', '+2'};
fprintf('G² comparison  (n_eval=%d, B=%d bootstrap replicates, seed=99/7)\n', n_eval, B);
fprintf('%-12s  %8s  %8s  %8s  %16s  %8s\n', 'Offset', 'G²(perf)', 'G²(RT)', 'ΔG²', '95% CI [lo, hi]', 'p (one-sided)');
fprintf('%s\n', repmat('-', 1, 72));
for b = 1 : n_bins
    fprintf('%-12s  %8.1f  %8.1f  %8.1f  [%6.1f, %6.1f]  %8.3f\n', ...
        cat_labels{b}, g2_perf_cat(b), g2_rt_cat(b), dg2_obs(b), ci_lo(b), ci_hi(b), p_vals(b));
end
fprintf('%s\n', repmat('-', 1, 72));
fprintf('%-12s  %8.1f  %8.1f  %8.1f  [%6.1f, %6.1f]  %8.3f\n', 'Total', ...
    g2_perf_total, g2_rt_total, dg2_obs(end), ci_lo(end), ci_hi(end), p_total);
fprintf('\nΔG² = G²(perf-fit) − G²(RT-fit);  positive = perf-fit explains RTs worse.\n');
fprintf('p = fraction of bootstrap replicates where ΔG² ≤ 0 (per bin and total).\n\n');

% =========================================================================
% 8.  Figure 1 — RT quantile curves per offset: monkey / perf-fit / RT-fit
% =========================================================================
lw       = 1.5;
fs       = 11;
clr_monk = [0.20 0.20 0.20];
clr_perf = [0.15 0.45 0.75];   % blue   — performance-fit model
clr_rt   = [0.85 0.33 0.10];   % orange — RT-fit model
be_rt    = linspace(0, RT_cutoff, 21);

fig1 = figure('Color','w','Units','centimeter','Position',[3 3 32 12]);

for b = 1 : n_bins
    monk_cat  = monk_rt(monk_off_rt == bins(b) & ~isnan(monk_rt));
    sim_cat_p = sim_rt_perf(sim_off_perf == bins(b) & ~isnan(sim_rt_perf));
    sim_cat_r = sim_rt_rt(sim_off_rt   == bins(b) & ~isnan(sim_rt_rt));
    if bins(b) == 0, lbl = 'correct'; else, lbl = sprintf('%+d', bins(b)); end

    % ── Top row: quantile curves ──────────────────────────────────────────
    subplot(2, n_bins, b);  hold on
    if ~isempty(monk_cat)
        plot(q_probs, quantile(monk_cat,  q_probs), 'o-', ...
            'Color', clr_monk, 'MarkerFaceColor', clr_monk, ...
            'MarkerSize', 5, 'LineWidth', lw, 'DisplayName', 'Monkey')
    end
    if ~isempty(sim_cat_p)
        plot(q_probs, quantile(sim_cat_p, q_probs), 's--', ...
            'Color', clr_perf, 'MarkerFaceColor', 'w', ...
            'MarkerSize', 5, 'LineWidth', lw, ...
            'DisplayName', sprintf('Perf-fit  G²=%.0f', g2_perf_cat(b)))
    end
    if ~isempty(sim_cat_r)
        plot(q_probs, quantile(sim_cat_r, q_probs), 'd:', ...
            'Color', clr_rt, 'MarkerFaceColor', 'w', ...
            'MarkerSize', 5, 'LineWidth', lw, ...
            'DisplayName', sprintf('RT-fit    G²=%.0f', g2_rt_cat(b)))
    end
    xlim([0 1]);  ylim([0 RT_cutoff])
    xticks(q_probs);  xticklabels({'.1','.3','.5','.7','.9'})
    xlabel('Quantile', 'FontSize', fs-1)
    if b == 1, ylabel('RT (ms)', 'FontSize', fs-1); end
    title(sprintf('Offset %s  (N=%d)', lbl, numel(monk_cat)), 'FontSize', fs-1)
    legend('Location', 'northwest', 'Box', 'off', 'FontSize', fs-2)
    set(gca, 'TickDir','out', 'LineWidth', 1, 'FontSize', fs-1, 'Box','off')

    % ── Bottom row: RT histograms ─────────────────────────────────────────
    subplot(2, n_bins, n_bins + b);  hold on
    if ~isempty(monk_cat)
        stairs(be_rt(1:end-1), histcounts(monk_cat,  be_rt, 'Normalization','probability'), ...
            '-',  'Color', clr_monk, 'LineWidth', lw)
    end
    if ~isempty(sim_cat_p)
        stairs(be_rt(1:end-1), histcounts(sim_cat_p, be_rt, 'Normalization','probability'), ...
            '--', 'Color', clr_perf, 'LineWidth', lw)
    end
    if ~isempty(sim_cat_r)
        stairs(be_rt(1:end-1), histcounts(sim_cat_r, be_rt, 'Normalization','probability'), ...
            ':',  'Color', clr_rt,   'LineWidth', lw)
    end
    xlabel('RT (ms)', 'FontSize', fs-1)
    if b == 1, ylabel('Proportion', 'FontSize', fs-1); end
    set(gca, 'TickDir','out', 'LineWidth', 1, 'FontSize', fs-1, 'Box','off')
end

sgtitle(sprintf('RT fit comparison — G²(perf-fit) = %.1f   vs   G²(RT-fit) = %.1f   (bootstrap p = %.3f)', ...
    g2_perf_total, g2_rt_total, p_total), 'FontSize', fs+1)

% =========================================================================
% 9.  Figure 2 — Grouped bar chart: G² per offset
% =========================================================================
fig2 = figure('Color','w','Units','centimeter','Position',[3 18 16 8]);
hold on
bar_w = 0.35;
x     = 1 : n_bins;
bar(x - bar_w/2, g2_perf_cat, bar_w, 'FaceColor', clr_perf, 'EdgeColor','k','LineWidth',0.8)
bar(x + bar_w/2, g2_rt_cat,   bar_w, 'FaceColor', clr_rt,   'EdgeColor','k','LineWidth',0.8)
xticks(x);  xticklabels({'-2','-1','0 (corr)','+1','+2'})
xlabel('Response offset', 'FontSize', fs)
ylabel('G²', 'FontSize', fs)
legend('Perf-fit', 'RT-fit', 'Location','best', 'Box','off', 'FontSize', fs)
title(sprintf('G² per offset   |   Total:  perf-fit = %.1f   RT-fit = %.1f', ...
    g2_perf_total, g2_rt_total), 'FontSize', fs)
set(gca, 'TickDir','out', 'LineWidth', 1, 'FontSize', fs, 'Box','off')

% =========================================================================
% 10.  Figure 3 — Bootstrap distribution of ΔG²
% =========================================================================
fig3 = figure('Color','w','Units','centimeter','Position',[3 30 20 8]);

subplot(1, 2, 1);  hold on
boxplot(dg2_boot(:, 1:n_bins), 'Labels', cat_labels, 'Notch','off', 'Symbol','')
yline(0, '--', 'Color','k', 'LineWidth', 1.5)
for b = 1 : n_bins
    plot(b, dg2_obs(b), 'o', 'MarkerFaceColor','k', 'MarkerEdgeColor','k', 'MarkerSize', 5)
end
xlabel('Offset', 'FontSize', fs)
ylabel('\DeltaG²  (perf - RT)', 'FontSize', fs)
title('Per-offset \DeltaG²  (bootstrap)', 'FontSize', fs)
set(gca, 'TickDir','out', 'LineWidth', 1, 'FontSize', fs, 'Box','off')

subplot(1, 2, 2);  hold on
histogram(dg2_boot(:, end), 40, 'FaceColor',[0.75 0.75 0.75], 'EdgeColor','none')
xline(dg2_obs(end), '-',  'Color','k',              'LineWidth', 2,   'DisplayName','Observed \DeltaG²')
xline(ci_lo(end),   '--', 'Color',[0.4 0.4 0.4],   'LineWidth', 1.5, 'DisplayName','95% CI')
xline(ci_hi(end),   '--', 'Color',[0.4 0.4 0.4],   'LineWidth', 1.5, 'HandleVisibility','off')
xline(0,            ':',  'Color',[0.6 0.6 0.6],   'LineWidth', 1.5, 'DisplayName','No difference')
xlabel('\DeltaG²_{total}  (perf - RT)', 'FontSize', fs)
ylabel('Bootstrap count', 'FontSize', fs)
title(sprintf('Total \\DeltaG²  —  p = %.3f', p_total), 'FontSize', fs)
legend('Location','best', 'Box','off', 'FontSize', fs-1)
set(gca, 'TickDir','out', 'LineWidth', 1, 'FontSize', fs, 'Box','off')

% =========================================================================
% 11.  Save figures
%
%      None of these three is a manuscript panel — the manuscript reports
%      this comparison as Table 1 and the G² totals in the text, both printed
%      above.  They are kept as the visual form of that table and are written
%      to figures/diagnostics/ so that figures/ holds manuscript panels only.
% =========================================================================
fig_dir = fullfile(project_dir, 'figures', 'diagnostics');
if ~isfolder(fig_dir), mkdir(fig_dir); end
exportgraphics(fig1, fullfile(fig_dir, 'compare_rt_quantiles.pdf'), 'ContentType','vector')
exportgraphics(fig2, fullfile(fig_dir, 'compare_rt_g2_bar.pdf'),    'ContentType','vector')
exportgraphics(fig3, fullfile(fig_dir, 'compare_rt_bootstrap.pdf'), 'ContentType','vector')
fprintf('Figures saved to %s\n', fig_dir);
