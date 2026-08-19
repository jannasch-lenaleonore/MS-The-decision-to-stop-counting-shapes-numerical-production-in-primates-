% diffusion_model_fig4.m
%
% Forward simulation of the stopping model at the fitted parameters, drawn
% against the monkeys' data.  Reads results/params_perf.mat, so it always
% shows the parameter set the manuscript reports.
%
% Model: drift rate drawn fresh per trial from N(drift_mean, drift_sigma)
% (drift_sigma = Ratcliff's eta, zero in the manuscript's variant), starting
% point folded-normal |N(0, sigma_z)|, reflecting lower border at 0.
%
% Figures written to figures/:
%   fig4A_model_traces.pdf        MS Fig. 4A  three example DV trajectories
%   fig4F_rt_model_vs_monkey.pdf  MS Fig. 4F  median RT +/- IQR per offset,
%                                             both monkeys and the model
%   fig4G_rt_quantiles.pdf        MS Fig. 4G  RT quantiles per offset
%   fig3E_rt_per_offset.pdf       MS Fig. 3E  median RT +/- IQR per offset,
%                                             monkeys only (no RT cutoff)
%
% Printed to the console:
%   the session-level RT trend and asymmetry tests (Wilcoxon signed-rank on
%   per-session Spearman rho and on median RT at negative vs positive
%   offsets) reported in Results, "Asymmetry in confirmation reaction times".

clear
rng(42);

% --- Parameters ---
targets_monk = [3, 4];  % monkey target numerosities to combine
RT_cutoff    = 1100;    % exclude monkey RTs above this value (ms) --> invalid
resp_time_scaling = [200, 500, 800, 1100, 200, 200, 800, 400];


% Parameters of the stopping model.  These are READ FROM THE FIT rather than
% typed in, so this figure can never drift away from the numbers reported in
% the results text and used in the cross-validation.  Hardcoding them meant
% every refit of fit_diffusion_mle_MS.m silently left the figures behind.
%
% The pooled fit is the one the paper reports (one parameter set for both
% animals); params_perf.mat also holds the per-animal fits under labels
% 'm1' and 'm2' if a per-animal version of a panel is ever wanted.
perf_fit = load(fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'results', 'params_perf.mat'));
fit_pooled = perf_fit.fit_perf(strcmp({perf_fit.fit_perf.label}, 'pooled'));

fix_drift  = perf_fit.fix_drift;
drift_mean = fit_pooled.x_opt(1);
if fix_drift
    sigma_z = fit_pooled.x_opt(2);
else
    sigma_z = fit_pooled.x_opt(3);
end
fprintf(['Stopping model from params_perf.mat (%s): drift %.4f, ' ...
    'sigma_z %.4f, neg-LL %.2f\n'], fit_pooled.label, drift_mean, sigma_z, ...
    fit_pooled.nll_eval);
% The per-animal fits and the fitted-vs-observed category tables used to be
% transcribed here as comments.  They are deliberately gone: they went stale
% at every refit and there is no way to tell a stale comment from a current
% one by reading it.  Run model_comparison_fig4.m for the current tables, or
% inspect perf_fit.fit_perf directly.



noise           = 0.1;   % within-trial diffusion noise
dt              = 0.01;
step_dur        = 1;
lower_border    = 'reflecting';   % 'reflecting' or 'open'
execution_noise = 0;

% Derived
bound       = 1;
n_steps_rel = 7;
period      = step_dur;
t           = 0 : dt : n_steps_rel * period;
n_t         = numel(t);

if fix_drift, drift_sigma = 0; end
draw_drift = @() drift_mean + drift_sigma * randn();
draw_start = @() sigma_z * abs(randn());

% --- Run n_trials trials ---
% Taken from the fit rather than chosen here, so the curves drawn in these
% figures are the same evaluation of the model that produced the numbers in
% the text.  At 20,000 the figures carried a few tenths of a percentage point
% of Monte Carlo noise relative to the reported predictions; at the fit's
% n_eval they agree to the printed precision.
n_trials = fit_pooled.n_eval;
offsets  = NaN(1, n_trials);
rts      = NaN(1, n_trials);
rts_raw  = NaN(1, n_trials);

for tr = 1 : n_trials
    drift_tr = draw_drift();

    dv     = zeros(1, n_t);
    dv(1)  = draw_start();
    stop_k = NaN;
    stop_t = NaN;

    for i = 2 : n_t
        step_raw = drift_tr * dt + noise * sqrt(dt) * randn;
        if strcmp(lower_border, 'reflecting')
            dv(i) = max(0, dv(i-1) + step_raw);
        else
            dv(i) = dv(i-1) + step_raw;
        end
        if dv(i) >= bound
            stop_k = ceil(t(i) / period);
            stop_t = t(i);
            break
        end
    end

    if ~isnan(stop_k)
        rt_val = stop_t - (stop_k - 1) * period + execution_noise * randn();
        if rt_val >= 0
            offsets(tr) = stop_k - 3;
            rts_raw(tr) = rt_val;
            rts(tr)     = rt_val * resp_time_scaling(randi(numel(resp_time_scaling)));
        end
    end
end

% --- Tally categories (-2:2 only) ---
bins   = -2 : 2;
n_bins = numel(bins);
pct    = zeros(1, n_bins);
for b = 1 : n_bins
    pct(b) = 100 * sum(offsets == bins(b), 'omitnan') / n_trials;
end
pct_nostop = 100 * sum(isnan(offsets)) / n_trials;

% These figures re-simulate the model at n_trials, while the fit evaluated it
% at perf_fit's much larger n_eval.  Same parameters, so the two agree up to
% Monte Carlo noise — printed here so any real disagreement (a stale fit, a
% changed RT_cutoff, different targets) is visible instead of silent.
fprintf('\nStopping model, predicted response frequencies [%%]\n');
fprintf('  %-22s%s\n', 'offset', sprintf('%+8d', bins));
fprintf('  %-22s%s\n', sprintf('this figure (n=%d)', n_trials), ...
    sprintf('%8.2f', pct));
fprintf('  %-22s%s\n', sprintf('the fit (n=%d)', fit_pooled.n_eval), ...
    sprintf('%8.2f', 100 * fit_pooled.pred_p(1:n_bins)));
fprintf('  %-22s%s\n', 'difference', ...
    sprintf('%8.2f', pct - 100 * fit_pooled.pred_p(1:n_bins)));
fprintf('  %-22s%8.2f\n', 'never stopped in range', pct_nostop);

% --- Load monkey data ---
project_dir  = fileparts(fileparts(mfilename('fullpath')));
monkeys      = {'m1', 'm2'};
monk_offsets = [];
monk_off_id  = [];   % monkey index for each entry of monk_offsets
monk_rt      = [];
monk_off_rt  = [];
monk_rt_id   = [];   
monk_sess_id = [];   % unique session index across both monkeys
monk_rt_nc      = [];
monk_off_rt_nc  = [];
monk_rt_id_nc   = [];
monk_sess_id_nc = [];
global_sess  = 0;

for mk = 1 : numel(monkeys)
    load(fullfile(project_dir, 'data', sprintf('bhv_data_tbl_%s', monkeys{mk})));
    for sess = 1 : height(dataTable)
        global_sess = global_sess + 1;
        RM = dataTable.RespMat{sess};
        RM(all(RM == 9, 2), :) = [];

        corr_mask = RM(:,5) == 0;
        err_mask  = RM(:,5) == 6;

        rt_conf      = dataTable.RT_conf{sess};
        rt_err_conf  = dataTable.RT_err_conf{sess};
        corr_counter = cumsum(corr_mask);
        err_counter  = cumsum(err_mask);

        for tg = targets_monk
            tg_mask = RM(:,2) == tg;
            off_tg  = RM(tg_mask, 3) - tg;
            monk_offsets = [monk_offsets; off_tg];
            monk_off_id  = [monk_off_id;  mk * ones(numel(off_tg), 1)];

            tg_corr_rows = find(corr_mask & tg_mask);
            if ~isempty(tg_corr_rows) && ~isempty(rt_conf)
                rt_c     = rt_conf(corr_counter(tg_corr_rows));
                valid    = ~isnan(rt_c) & rt_c <= RT_cutoff;
                valid_nc = ~isnan(rt_c);
                monk_rt      = [monk_rt;      rt_c(valid)];                          
                monk_off_rt  = [monk_off_rt;  zeros(sum(valid), 1)];                
                monk_rt_id   = [monk_rt_id;   mk          * ones(sum(valid), 1)];  
                monk_sess_id = [monk_sess_id; global_sess * ones(sum(valid), 1)];   
                monk_rt_nc      = [monk_rt_nc;      rt_c(valid_nc)];                         
                monk_off_rt_nc  = [monk_off_rt_nc;  zeros(sum(valid_nc), 1)];               
                monk_rt_id_nc   = [monk_rt_id_nc;   mk          * ones(sum(valid_nc), 1)];    
                monk_sess_id_nc = [monk_sess_id_nc; global_sess * ones(sum(valid_nc), 1)];    
            end

            tg_err_rows = find(err_mask & tg_mask);
            if ~isempty(tg_err_rows) && ~isempty(rt_err_conf)
                rt_e     = rt_err_conf(err_counter(tg_err_rows));
                off_e    = RM(tg_err_rows, 3) - tg;
                valid    = ~isnan(rt_e) & rt_e <= RT_cutoff;
                valid_nc = ~isnan(rt_e);
                monk_rt      = [monk_rt;      rt_e(valid)];                         
                monk_off_rt  = [monk_off_rt;  off_e(valid)];                         
                monk_rt_id   = [monk_rt_id;   mk          * ones(sum(valid), 1)];   
                monk_sess_id = [monk_sess_id; global_sess * ones(sum(valid), 1)];    
                monk_rt_nc      = [monk_rt_nc;      rt_e(valid_nc)];                          
                monk_off_rt_nc  = [monk_off_rt_nc;  off_e(valid_nc)];                        
                monk_rt_id_nc   = [monk_rt_id_nc;   mk          * ones(sum(valid_nc), 1)];    
                monk_sess_id_nc = [monk_sess_id_nc; global_sess * ones(sum(valid_nc), 1)];    
            end
        end
    end
end

% Observed response frequencies per animal, on the same offset bins as the
% model prediction above.  Printed rather than plotted: the manuscript shows
% the model against the two number lines (MS Fig. 4C, model_comparison_fig4.m),
% so this is the numerical check that the simulation drawn in MS Fig. 4F/4G
% sits on the same distribution the monkeys produced.
monk_pct_mk = zeros(2, n_bins);
for mk = 1 : 2
    n_monk = sum(monk_off_id == mk);
    for b = 1 : n_bins
        monk_pct_mk(mk, b) = 100 * sum(monk_offsets == bins(b) & monk_off_id == mk) / n_monk;
    end
end

fprintf('\nObserved response frequencies [%%], targets %s pooled\n', ...
    mat2str(targets_monk));
fprintf('  %-22s%s\n', 'offset', sprintf('%+8d', bins));
for mk = 1 : 2
    fprintf('  %-22s%s\n', sprintf('Monkey %d (n=%d)', mk, sum(monk_off_id == mk)), ...
        sprintf('%8.2f', monk_pct_mk(mk, :)));
end

% --- Demo trials ---
% Own seed, so the illustrated trials do not change when anything upstream
% draws a different number of random values — raising n_trials used to
% silently reshuffle this panel.  This seed gives one trial stopping early,
% one correct and one late, in that order.
rng(5)

n_demo        = 3;
dv_demo       = cell(1, n_demo);
si_demo       = NaN(1, n_demo);
sk_demo       = NaN(1, n_demo);
drift_demo    = NaN(1, n_demo);
dv_start_demo = NaN(1, n_demo);

for d = 1 : n_demo
    require_reflect = (d == 3 && strcmp(lower_border, 'reflecting'));

    while true
        drift_demo(d)    = draw_drift();
        dv_start_demo(d) = draw_start();
        si_demo(d) = NaN;
        sk_demo(d) = NaN;

        dv    = zeros(1, n_t);
        dv(1) = dv_start_demo(d);
        hit_reflect = false;
        for i = 2 : n_t
            step_raw = drift_demo(d) * dt + noise * sqrt(dt) * randn;
            if strcmp(lower_border, 'reflecting')
                if dv(i-1) + step_raw < 0
                    hit_reflect = true;
                end
                dv(i) = max(0, dv(i-1) + step_raw);
            else
                dv(i) = dv(i-1) + step_raw;
            end
            if dv(i) >= bound
                si_demo(d) = i;
                sk_demo(d) = ceil(t(i) / period);
                break
            end
        end

        if ~require_reflect || hit_reflect
            break
        end
    end
    dv_demo{d} = dv;
end

% Display up to end of +2 step
t_demo_end = 5 * period;
plot_end   = find(t >= t_demo_end, 1);
if isempty(plot_end), plot_end = n_t; end
t_demo_x = t(1 : plot_end);

% --- Plot ---
lw = 1.6;
fs = 16;

% One call styles every axis, as in model_comparison_fig4.m: tick labels at
% the same size as the axis labels, and all text in true black rather than
% MATLAB's default grey, which prints washed out.
style_ax = @(ax) apply_axis_style(ax, fs, lw);
% The stopping model carries ONE colour across every figure of the paper: the
% Okabe-Ito blue used for it in model_comparison_fig4.m.  Keep the two in
% step — a reader who has learned "blue = the stopping model" in one figure
% must not meet it in a different colour in the next.
clr_stop = [0.00 0.45 0.70];

% The three demo trials are three draws from that same model, so they are
% three lightness steps of its colour rather than three unrelated hues.
clr_dv    = {[0.00 0.28 0.45], clr_stop, [0.45 0.70 0.84]};
clr_bound = [0 0 0];
clr_gap   = [0.80 0.80 0.80];  % light grey — frozen-time boundaries
clr_rt    = [0.4 0.7 0.3];
gap_vis   = 0.15;
t2d       = @(tt) tt + floor(tt / period) * gap_vis;

% Monkeys are always shown separately: black = monkey 1, grey = monkey 2.
% The model keeps its blue and stays dashed, so data and model separate by
% both colour and line style and the figures survive greyscale printing.
clr_monk_ind = {[0 0 0], [0.62 0.62 0.62]};
monk_lbl     = {'Monkey 1', 'Monkey 2'};
clr_model    = clr_stop;
x_jitter     = [-0.05, 0.05];
fit_paper = @(h) set(h, 'PaperUnits', 'centimeters', ...
    'PaperSize', h.Position(3:4), 'PaperPosition', [0 0 h.Position(3:4)]);
fig_dir = fullfile(project_dir, 'figures');
if ~isfolder(fig_dir), mkdir(fig_dir); end

rt_cats = bins;
n_cats  = numel(rt_cats);

% Pre-compute RT statistics (model)
rt_med_mod = NaN(1, n_cats);
rt_q25_mod = NaN(1, n_cats);
rt_q75_mod = NaN(1, n_cats);
for ci = 1 : n_cats
    cat_rt = rts(offsets == rt_cats(ci) & ~isnan(rts));
    if ~isempty(cat_rt)
        rt_med_mod(ci) = median(cat_rt);
        rt_q25_mod(ci) = quantile(cat_rt, 0.25);
        rt_q75_mod(ci) = quantile(cat_rt, 0.75);
    end
end

% Pre-compute RT statistics per monkey (RT cutoff applied)
rt_med_monk = NaN(2, n_cats);
rt_q25_monk = NaN(2, n_cats);
rt_q75_monk = NaN(2, n_cats);
for mk = 1 : 2
    for ci = 1 : n_cats
        cat_rt = monk_rt(monk_rt_id == mk & monk_off_rt == rt_cats(ci));
        cat_rt = cat_rt(~isnan(cat_rt));
        if ~isempty(cat_rt)
            rt_med_monk(mk, ci) = median(cat_rt);
            rt_q25_monk(mk, ci) = quantile(cat_rt, 0.25);
            rt_q75_monk(mk, ci) = quantile(cat_rt, 0.75);
        end
    end
end

y_lim_demo   = [0, bound * 1.4];
n_steps_demo = 5;

% ═══════════════════════════════════════════════════════════════════════════
% MS Fig. 4A — example decision-variable trajectories
% ═══════════════════════════════════════════════════════════════════════════
figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 28 10])
fit_paper(gcf)
ax_sz   = axes('Position', [0.05,  0.15, 0.07, 0.6]);
ax_demo = axes('Position', [0.12,  0.15, 0.85,  0.6]);

% ── Demo traces ──────────────────────────────────────────────────────────
axes(ax_demo);  hold on

% Pre-step gap (frozen period before step -2)
xline(-gap_vis, '-', 'Color', clr_gap, 'LineWidth', 0.8, 'HandleVisibility', 'off')
xline(0,        '-', 'Color', clr_gap, 'LineWidth', 0.8, 'HandleVisibility', 'off')

% Gap boundary xlines (start + end of each frozen period) and offset labels
for k = 1 : n_steps_demo
    x_gap_s = t2d(k * period) - gap_vis;
    x_gap_e = t2d(k * period);
    xline(x_gap_s, '-', 'Color', clr_gap, 'LineWidth', 0.8, 'HandleVisibility', 'off')
    xline(x_gap_e, '-', 'Color', clr_gap, 'LineWidth', 0.8, 'HandleVisibility', 'off')
    off_lbl = k - 3;
    if off_lbl == 0, lbl_str = 'correct'; else, lbl_str = sprintf('%+d', off_lbl); end
    lbl_x = t2d((k - 0.5) * period);
    text(lbl_x, y_lim_demo(2) - 0.05, lbl_str, ...
        'HorizontalAlignment', 'center', 'FontSize', fs, 'Color', 'k')
end

% Decision boundary
for k = 1 : n_steps_demo
    t_kstart = (k - 1) * period;
    t_seg_d  = [t2d(t_kstart), t2d(k * period) - gap_vis];
    plot(t_seg_d, [bound bound], '--', 'Color', clr_bound, 'LineWidth', lw, ...
        'HandleVisibility', 'off')
end

% DV traces, RT arrows, and frozen-time dotted lines
legend_rt_added = false;
rt_arrow_yoff   = [0.05, 0.075, 0.1];

for d = 1 : n_demo
    dv_d = dv_demo{d}(1 : plot_end);
    trace_end = plot_end;
    if ~isnan(si_demo(d)), trace_end = min(si_demo(d), plot_end); end
    first_seg = true;
    % Pre-step dotted line at starting value during pre-step gap
    plot([-gap_vis, 0], [dv_start_demo(d), dv_start_demo(d)], ':', ...
        'Color', clr_dv{d}, 'LineWidth', lw, 'HandleVisibility', 'off')
    for k = 1 : n_steps_demo
        t_lo = (k-1) * period;
        idx  = find(t(1:trace_end) >= t_lo & t(1:trace_end) < k * period);
        if isempty(idx), break; end
        if first_seg
            plot(t2d(t(idx)), dv_d(idx), 'Color', clr_dv{d}, 'LineWidth', lw, ...
                'DisplayName', sprintf('DV trial %d', d))
            first_seg = false;
        else
            plot(t2d(t(idx)), dv_d(idx), 'Color', clr_dv{d}, 'LineWidth', lw, 'HandleVisibility', 'off')
        end
        % Frozen-time dotted line: DV held constant during inter-step gap
        trial_stopped_here = ~isnan(si_demo(d)) && max(idx) >= si_demo(d);
        if ~trial_stopped_here
            dv_end  = dv_d(max(idx));
            plot([t2d(k * period) - gap_vis, t2d(k * period)], [dv_end, dv_end], ':', ...
                'Color', clr_dv{d}, 'LineWidth', lw, 'HandleVisibility', 'off')
        end
        if trial_stopped_here, break; end
    end
    if ~isnan(si_demo(d))
        stop_t_eff  = t(si_demo(d));
        stop_disp_x = t2d(stop_t_eff);
        plot(stop_disp_x, bound, 'o', 'MarkerSize', 8, ...
            'MarkerFaceColor', clr_dv{d}, 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
            'HandleVisibility', 'off')
        nrw_end_t    = (sk_demo(d) - 1) * period;
        nrw_end_disp = t2d(nrw_end_t);
        rt_d         = stop_t_eff - nrw_end_t;
        arrow_y      = bound + rt_arrow_yoff(d);
        if d < n_demo
            quiver(nrw_end_disp, arrow_y, rt_d, 0, 0, ...
                'Color', clr_rt, 'LineWidth', 2, 'MaxHeadSize', min(1, 0.3/max(rt_d,dt)), ...
                'HandleVisibility', 'off')
        else
            quiver(nrw_end_disp, arrow_y, rt_d, 0, 0, ...
            'Color', clr_rt, 'LineWidth', 2, 'MaxHeadSize', min(1, 0.3/max(rt_d,dt)), ...
            'Displayname', 'RT')
        end
        xline(stop_disp_x, ':', 'Color', clr_dv{d}, 'LineWidth', 1, 'HandleVisibility', 'off')
    end
end

xlim([-gap_vis  t2d(t_demo_end)- gap_vis]);  ylim(y_lim_demo)
legend('Location', 'southeast', 'FontSize', fs)
for k = 1 : n_steps_demo
    seg_x0 = t2d((k-1) * period);
    seg_x1 = t2d(k * period) - gap_vis;
    plot([seg_x0, seg_x1], [0, 0], '-', 'Color', 'k', 'LineWidth', 0.8, 'HandleVisibility', 'off')
end
set(ax_demo, 'TickDir', 'out', 'LineWidth', lw, 'FontSize', fs, ...
    'YTick', [], 'XTick', [], 'Box', 'off', 'XColor', 'k', 'YColor', 'k')
try, ax_demo.XAxis.Axle.Visible = 'off'; catch, end

% ── Starting-point distribution ───────────────────────────────────────────
axes(ax_sz);  hold on
sz_grid  = linspace(y_lim_demo(1), y_lim_demo(2), 300);
sz_pdf   = 1/(sigma_z * sqrt(2*pi)) * exp(-sz_grid.^2 / (2*sigma_z^2));
sz_pdf_n = sz_pdf / max(sz_pdf);
fill([0, sz_pdf_n, 0], [sz_grid(1), sz_grid, sz_grid(end)], ...
    [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.4)
plot(sz_pdf_n, sz_grid, 'Color', [0.5 0.5 0.5], 'LineWidth', lw)
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 1)
for d = 1 : n_demo
    pdf_n_at_d = exp(-dv_start_demo(d)^2 / (2*sigma_z^2));
    plot(pdf_n_at_d, dv_start_demo(d), 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', clr_dv{d}, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8)
end
xlim([0, 1.16]);  ylim(y_lim_demo)
xlabel('P(z_0)', 'FontSize', fs)
set(ax_sz, 'TickDir', 'out', 'LineWidth', lw, 'FontSize', fs, 'XTick', [], ...
    'XDir', 'reverse', 'YTick', [], 'YColor', 'none', 'XColor', 'k', 'Box', 'off')
linkaxes([ax_sz, ax_demo], 'y')
ax_demo.YAxis.Color = [0 0 0];
ax_sz.XAxis.Color   = [0 0 0];
try
    ax_demo.YAxis.Axle.Visible   = 'on';
    ax_demo.YAxis.Axle.ColorType = 'rgb';
    ax_demo.YAxis.Axle.ColorData = uint8([0; 0; 0; 255]);
    ax_sz.XAxis.Axle.Visible     = 'on';
    ax_sz.XAxis.Axle.ColorType   = 'rgb';
    ax_sz.XAxis.Axle.ColorData   = uint8([0; 0; 0; 255]);
catch
end
exportgraphics(gcf, fullfile(fig_dir, 'fig4A_model_traces.pdf'), 'ContentType', 'vector')

% ═══════════════════════════════════════════════════════════════════════════
% MS Fig. 4G — RT quantiles per offset bin
% ═══════════════════════════════════════════════════════════════════════════
q_probs = [.10, .30, .50, .70, .90];

y_q_max = 0;
for mk = 1 : 2
    for ci = 1 : n_cats
        cat_m = monk_rt(monk_rt_id == mk & monk_off_rt == rt_cats(ci) & ~isnan(monk_rt));
        if numel(cat_m) >= 10
            y_q_max = max(y_q_max, quantile(cat_m, 0.95));
        end
    end
end
y_q_max = min(ceil(y_q_max / 100) * 100, RT_cutoff);

% This row of five narrow panels is the one that ends up smallest on the
% page, so it carries its own font size a few points above the shared fs and
% a slightly wider canvas to pay for it.  The rest of the script keeps fs.
% Do not push past +3: at +4 the y tick labels no longer fit and MATLAB
% silently drops the 200 ms ticks down to 0/500/1000.
fs_q = fs + 3;

figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 34 11])
fit_paper(gcf)
for ci = 1 : n_cats
    subplot(1, n_cats, ci)
    hold on

    cat_mod = rts(offsets == rt_cats(ci) & ~isnan(rts));

    for mk = 1 : 2
        cat_monk = monk_rt(monk_rt_id == mk & monk_off_rt == rt_cats(ci) & ~isnan(monk_rt));
        if numel(cat_monk) >= 10
            q_m = quantile(cat_monk, q_probs);
            plot(q_probs, q_m, 'o-', 'Color', clr_monk_ind{mk}, ...
                'MarkerFaceColor', clr_monk_ind{mk}, 'MarkerSize', 4, 'LineWidth', lw, ...
                'HandleVisibility', 'off')
        end
    end
    if numel(cat_mod) >= 10
        q_s = quantile(cat_mod, q_probs);
        plot(q_probs, q_s, 'o--', 'Color', clr_model, ...
            'MarkerFaceColor', clr_model, 'MarkerSize', 4, 'LineWidth', lw, ...
            'HandleVisibility', 'off')
    end

    xlim([0 1]);  ylim([0, y_q_max])
    % Three ticks, not five: at this label size five would not fit across a
    % fifth of the figure and MATLAB silently rotates them to vertical.  All
    % five quantiles are still plotted as markers.
    xticks([.1 .5 .9]);  xticklabels({'.1','.5','.9'});  xtickangle(0)
    xlabel('Quantile', 'FontSize', fs_q)
    if ci == 1
        ylabel('RT (ms)', 'FontSize', fs_q)
    elseif ci == n_cats
        % Invisible markers so the legend is complete even if a bin is empty
        for mk = 1 : 2
            plot(NaN, NaN, 'o-', 'Color', clr_monk_ind{mk}, ...
                'MarkerFaceColor', clr_monk_ind{mk}, 'MarkerSize', 4, 'LineWidth', lw, ...
                'DisplayName', monk_lbl{mk})
        end
        plot(NaN, NaN, 'o--', 'Color', clr_model, ...
            'MarkerFaceColor', clr_model, 'MarkerSize', 4, 'LineWidth', lw, ...
            'DisplayName', 'Model')
        legend('Location', 'northwest', 'Box', 'off', 'FontSize', fs_q)
    end
    if rt_cats(ci) == 0, lbl = 'correct'; else, lbl = sprintf('%+d', rt_cats(ci)); end
    title(lbl, 'FontSize', fs_q)
    apply_axis_style(gca, fs_q, lw)   % style_ax, but at this figure's fs_q
end
exportgraphics(gcf, fullfile(fig_dir, 'fig4G_rt_quantiles.pdf'), 'ContentType', 'vector')

% ═══════════════════════════════════════════════════════════════════════════
% MS Fig. 3E — median ± IQR RT per offset bin, per monkey
% ═══════════════════════════════════════════════════════════════════════════
% Pre-compute per-monkey RT statistics (no RT cutoff)
rt_med_mk  = NaN(2, n_cats);
rt_q25_mk  = NaN(2, n_cats);
rt_q75_mk  = NaN(2, n_cats);
for mk = 1 : 2
    for ci = 1 : n_cats
        cat_rt = monk_rt_nc(monk_rt_id_nc == mk & monk_off_rt_nc == rt_cats(ci) & ~isnan(monk_rt_nc));
        if numel(cat_rt) >= 5
            rt_med_mk(mk, ci) = median(cat_rt);
            rt_q25_mk(mk, ci) = quantile(cat_rt, 0.25);
            rt_q75_mk(mk, ci) = quantile(cat_rt, 0.75);
        end
    end
end

y_top_mk = max(rt_q75_mk(:), [], 'omitnan') * 1.2;
if isnan(y_top_mk) || y_top_mk == 0, y_top_mk = RT_cutoff; end

% ── Statistical tests: trend and asymmetry (session-level) ───────────────
% Unit of analysis = session, not trial, to avoid overpower from within-
% session trial correlations.
%
% Trend:     compute Spearman rho(offset, median_RT) per session, then test
%            whether the distribution of rho values is < 0 across sessions
%            (Wilcoxon signed-rank against 0).
% Asymmetry: compute (median_RT_neg - median_RT_pos) per session, then test
%            whether the distribution of differences is != 0 across sessions
%            (Wilcoxon signed-rank, paired by session).
min_n_sess = 5;   % min trials per offset group per session to include that session

p_trend_mk   = NaN(1, 2);
rho_med_mk   = NaN(1, 2);   % median per-session rho
p_asym_mk    = NaN(1, 2);
n_sess_mk    = NaN(1, 2);

for mk = 1 : 2
    sess_ids = unique(monk_sess_id_nc(monk_rt_id_nc == mk));
    n_sess   = numel(sess_ids);
    n_sess_mk(mk) = n_sess;

    rho_per_sess  = NaN(n_sess, 1);
    diff_per_sess = NaN(n_sess, 1);

    for si = 1 : n_sess
        in_s = monk_sess_id_nc == sess_ids(si);

        % Per-session Spearman rho across offset bins
        med_rt_s = NaN(1, n_cats);
        for ci = 1 : n_cats
            rt_sb = monk_rt_nc(in_s & monk_off_rt_nc == rt_cats(ci) & ~isnan(monk_rt_nc));
            if numel(rt_sb) >= min_n_sess
                med_rt_s(ci) = median(rt_sb);
            end
        end
        ok = ~isnan(med_rt_s);
        if sum(ok) >= 3
            rho_per_sess(si) = corr(rt_cats(ok)', med_rt_s(ok)', 'Type', 'Spearman');
        end

        % Per-session asymmetry: median RT at negative vs positive offsets
        rt_neg = monk_rt_nc(in_s & monk_off_rt_nc < 0 & monk_off_rt_nc >= -2 & ~isnan(monk_rt_nc));
        rt_pos = monk_rt_nc(in_s & monk_off_rt_nc > 0 & monk_off_rt_nc <=  2 & ~isnan(monk_rt_nc));
        if numel(rt_neg) >= min_n_sess && numel(rt_pos) >= min_n_sess
            diff_per_sess(si) = median(rt_neg) - median(rt_pos);
        end
    end

    rho_valid  = rho_per_sess(~isnan(rho_per_sess));
    diff_valid = diff_per_sess(~isnan(diff_per_sess));

    if numel(rho_valid) >= 5
        p_trend_mk(mk) = signrank(rho_valid);
        rho_med_mk(mk) = median(rho_valid);
    end
    if numel(diff_valid) >= 5
        p_asym_mk(mk) = signrank(diff_valid);
    end
end

fprintf('\n--- RT trend and asymmetry tests (session-level) ---\n');
fprintf('  Unit of analysis: session\n');
fprintf('  Trend:     Wilcoxon signed-rank of per-session Spearman rho vs 0.\n');
fprintf('  Asymmetry: Wilcoxon signed-rank of per-session (med_RT_neg - med_RT_pos) vs 0.\n\n');
for mk = 1 : 2
    fprintf('  Monkey %s (%d sessions):  median rho = %+.3f, p = %.8f  |  asym p = %.8f\n', ...
        monkeys{mk}, n_sess_mk(mk), rho_med_mk(mk), p_trend_mk(mk), p_asym_mk(mk));
end

figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 14 9])
fit_paper(gcf)
hold on

for mk = 1 : 2
    valid_ci = find(~isnan(rt_med_mk(mk, :)));
    x_vals   = rt_cats(valid_ci) + x_jitter(mk);

    % Connect dots
    plot(x_vals, rt_med_mk(mk, valid_ci), '-', ...
        'Color', clr_monk_ind{mk}, 'LineWidth', lw, 'HandleVisibility', 'off')

    % Error bars + markers
    for ci = valid_ci
        x_ci = rt_cats(ci) + x_jitter(mk);
        errorbar(x_ci, rt_med_mk(mk, ci), ...
            rt_med_mk(mk, ci) - rt_q25_mk(mk, ci), ...
            rt_q75_mk(mk, ci) - rt_med_mk(mk, ci), ...
            'o', 'Color', clr_monk_ind{mk}, ...
            'MarkerFaceColor', clr_monk_ind{mk}, ...
            'MarkerSize', 6, 'LineWidth', lw, 'CapSize', 5, ...
            'HandleVisibility', 'off')
    end

    % Invisible marker for legend
    plot(NaN, NaN, 'o-', 'Color', clr_monk_ind{mk}, ...
        'MarkerFaceColor', clr_monk_ind{mk}, 'MarkerSize', 6, 'LineWidth', lw, ...
        'DisplayName', monk_lbl{mk})
end

xline(0, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1, 'HandleVisibility', 'off')
xlim([-2.7 2.7]);  ylim([0, y_top_mk])
xticks(rt_cats);  xticklabels({'-2', '-1', '0', '+1', '+2'})
xlabel('Response offset', 'FontSize', fs)
ylabel('Median RT ± IQR (ms)', 'FontSize', fs)
legend('Location', 'northeast', 'Box', 'off', 'FontSize', fs)
style_ax(gca)
exportgraphics(gcf, fullfile(fig_dir, 'fig3E_rt_per_offset.pdf'), 'ContentType', 'vector')



% ═══════════════════════════════════════════════════════════════════════════
% MS Fig. 4F — median ± IQR RT per offset bin: monkeys vs model
% ═══════════════════════════════════════════════════════════════════════════
y_top_cmp = max([rt_q75_monk(:)', rt_q75_mod], [], 'omitnan') * 1.2;
if isnan(y_top_cmp) || y_top_cmp == 0, y_top_cmp = RT_cutoff; end

x_jitter_cmp = [-0.07, 0, 0.07];

figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 14 9])
fit_paper(gcf)
hold on

for src = 1 : 3   % 1 = monkey 1, 2 = monkey 2, 3 = model
    if src <= 2
        med_rt  = rt_med_monk(src, :);
        q25_rt  = rt_q25_monk(src, :);
        q75_rt  = rt_q75_monk(src, :);
        clr_src = clr_monk_ind{src};
        lbl_src = monk_lbl{src};
        sty_src = '-';
    else
        med_rt  = rt_med_mod;
        q25_rt  = rt_q25_mod;
        q75_rt  = rt_q75_mod;
        clr_src = clr_model;
        lbl_src = 'Model';
        sty_src = '--';
    end
    xj = x_jitter_cmp(src);

    valid_ci = find(~isnan(med_rt));
    x_vals   = rt_cats(valid_ci) + xj;

    plot(x_vals, med_rt(valid_ci), sty_src, ...
        'Color', clr_src, 'LineWidth', lw, 'HandleVisibility', 'off')

    for ci = valid_ci
        errorbar(rt_cats(ci) + xj, med_rt(ci), ...
            med_rt(ci) - q25_rt(ci), q75_rt(ci) - med_rt(ci), ...
            'o', 'Color', clr_src, ...
            'MarkerFaceColor', clr_src, ...
            'MarkerSize', 6, 'LineWidth', lw, 'CapSize', 5, ...
            'HandleVisibility', 'off')
    end

    plot(NaN, NaN, ['o' sty_src], 'Color', clr_src, ...
        'MarkerFaceColor', clr_src, 'MarkerSize', 6, 'LineWidth', lw, ...
        'DisplayName', lbl_src)
end

xline(0, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1, 'HandleVisibility', 'off')
xlim([-2.7 2.7]);  ylim([0, y_top_cmp])
xticks(rt_cats);  xticklabels({'-2', '-1', '0', '+1', '+2'})
xlabel('Response offset', 'FontSize', fs)
ylabel('Median RT ± IQR (ms)', 'FontSize', fs)
legend('Location', 'northeast', 'Box', 'off', 'FontSize', fs)
style_ax(gca)
exportgraphics(gcf, fullfile(fig_dir, 'fig4F_rt_model_vs_monkey.pdf'), 'ContentType', 'vector')



% =========================================================================
% Local functions
% =========================================================================
function apply_axis_style(ax, fs, lw)
% Kept identical to the helper of the same name in model_comparison_fig4.m,
% so the two figure sets cannot drift apart in tick size or colour.
set(ax, 'TickDir', 'out', 'LineWidth', lw, 'FontSize', fs, 'Box', 'off', ...
    'XColor', 'k', 'YColor', 'k', 'ZColor', 'k')

lbl = [ax.Title, ax.XLabel, ax.YLabel];
set(lbl(isgraphics(lbl)), 'Color', 'k')
end
