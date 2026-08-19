% schematic_row2_MS.m
%
% Row 2 of the model schematic: the OFFSET-INVARIANCE signature.
%
% One panel per account of the response distribution, all on the same axes:
% the predicted probability of each offset r - n, with target 3 and target 4
% drawn on top of each other.  The panel is built to be read as a yes/no
% question — do the two targets superimpose?
%
%   Stopping             target-invariant BY CONSTRUCTION.  One accumulation
%                        to a bound serves both targets, so the two curves
%                        are the same curve.  It is SKEWED: a start point
%                        drawn from a folded normal lets a trial arrive at
%                        the bound early far more easily than late.
%   Linear number line   mu_n = n with ONE sigma in count units, so the
%                        distribution is the same shape at every n and the
%                        two curves superimpose as well.  It is SYMMETRIC,
%                        which is what separates it from the stopping model.
%   Log number line      mu_n = log(n) with one sigma in LOG units, so the
%                        absolute width grows with n.  The two curves do NOT
%                        superimpose: the target-4 curve is wider, lower and
%                        pushed to the right.
%
% =========================================================================
% THIS IS A SCHEMATIC.  THE PARAMETERS ARE ILLUSTRATIVE, NOT FITS.
% =========================================================================
% Nothing here is read from results/.  The figure exists to show what the
% three accounts CLAIM, so the parameters are chosen to separate the claims
% as clearly as possible and the differences are deliberately exaggerated
% relative to the fitted values.  In particular the linear number line is
% drawn with NO bias (mu_n = n exactly), so it comes out perfectly symmetric;
% the fitted bias of about -0.07 count units tilts it very slightly and would
% blur the one feature this panel uses to tell it from the stopping model.
%
% For the same reason the two number lines are drawn as FULL DISTRIBUTIONS —
% a normal on the count axis, a normal in log units read back onto the count
% axis — rather than as the five discrete probabilities the likelihood
% actually scores.  The continuous curve is the model; the five points are
% only where it happens to be sampled.  The stopping model keeps joined
% markers because it is simulated and has no closed form, exactly as in MS Fig. 4C.
%
% The quantitative version of this comparison, on the real fits and with the
% discrete probabilities, are MS Fig. 4C and 4E in model_comparison_fig4.m.
%
% Figure written to figures/:
%   fig4B_model_predictions.pdf    idealised offset distributions, target 3 vs
%                              target 4, one panel per model

clear; clc;

project_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_dir, 'helpers'));

targets_schem = [3, 4];
n_targets     = numel(targets_schem);

% Offsets shown.  Wider than the -2:+2 the likelihood uses, because the log
% line's target-4 tail is part of what the panel is claiming and clipping it
% at +2 would hide it.
off_lim  = 3;
off_int  = -off_lim : off_lim;

% =========================================================================
% 1.  Illustrative parameters  (NOT fits — see the header)
% =========================================================================
% Stopping: the real mechanism from run_diffusion_sim, so the shape of the
% skew is genuine, but with a larger start-point spread than the fit gives.
% sigma_z is what produces the skew — a high start point crosses the bound
% early, and there is no matching way to arrive late — so enlarging it is
% what makes "skewed" legible at a glance.
sim_drift   = 0.30;
sim_sigma_z = 0.55;

% Number lines.  Both are drawn unbiased, so the linear one is exactly
% symmetric about the target and the log one is skewed only because of the
% logarithm, not because of a fitted offset.
%
% The widths are chosen so that all three models are equally wide AT TARGET 3
% (an SD of about 1.05 responses; the console prints all three to confirm).
% That matching is what makes the panel an argument: the three accounts start
% from the same spread on the smaller target, so the only differences left to
% see are the two the schematic is about — the SHAPE of the distribution, and
% whether target 4 looks any different from target 3.  Letting the widths
% drift apart would add a spurious third difference and invite the reader to
% conclude that one model predicts better accuracy than another.
sig_lin = 1.05;    % count units, the same at every target
sig_log = 0.30;    % log units,   the same at every target — exaggerated

% ---- stopping model: simulate once, use for both targets ----------------
% Fixed model constants, identical to the fitting scripts.
fix_drift = true;
noise     = 0.1;
dt        = 0.01;
bound     = 1;
period    = 1;
t_vec     = (0 : dt : 7 * period)';
n_t       = numel(t_vec);

rng(42);
n_sim = 400000;

% run_diffusion_sim also returns RTs, which need a scaling vector; this panel
% is about offsets only, so the scaling is a placeholder and its output is
% discarded.
[~, off_sim] = run_diffusion_sim([sim_drift, sim_sigma_z], fix_drift, bound, ...
    n_sim, t_vec, n_t, dt, noise, period, 1);

p_stop = histcounts(off_sim, [off_int - 0.5, off_lim + 0.5], ...
    'Normalization', 'probability');

% =========================================================================
% 2.  Number lines as full distributions on the offset axis
% =========================================================================
% Each is evaluated on a fine grid of responses r and normalised to unit area
% over r, so "wider" automatically reads as "lower" — which is the whole of
% the log line's claim.  Offset is r - n, a unit shift, so the area carries
% over to the offset axis unchanged.
%
% The log line is a normal in log units read back onto the count axis: the
% height of the Gaussian at log(r).  That is the convention this project
% uses throughout (see predict_gauss in fit_gaussian_mle_MS.m) and is what
% gives the count-axis curve its log-normal look.
% Both targets are normalised over the SAME wide support, not over their own
% slice of the offset window.  Normalising per window would divide the two
% log curves by different numbers — target 4's window cuts more of its right
% tail — and would flatter the target-4 curve upwards, working against the
% very difference the panel is drawing.
r_norm  = linspace(1e-4, 20, 40000);
n_fine  = 2000;
dens     = cell(2, n_targets);     % {1,:} linear, {2,:} log
off_fine = cell(2, n_targets);

for ti = 1 : n_targets
    n_tg = targets_schem(ti);

    kern_lin = @(x) exp(-0.5 * ((x      - n_tg)      / sig_lin) .^ 2);
    kern_log = @(x) exp(-0.5 * ((log(x) - log(n_tg)) / sig_log) .^ 2);

    Z_lin = trapz(r_norm, kern_lin(r_norm));
    Z_log = trapz(r_norm, kern_log(r_norm));

    % Responses must stay positive — the logarithm is undefined at r <= 0,
    % and at target 3 the offset window reaches r = 0 exactly.
    r = linspace(max(n_tg - off_lim, 1e-3), n_tg + off_lim, n_fine);

    dens{1, ti}     = kern_lin(r) / Z_lin;
    dens{2, ti}     = kern_log(r) / Z_log;
    off_fine{1, ti} = r - n_tg;
    off_fine{2, ti} = r - n_tg;
end

% Markers at the whole-number offsets, read off the same curves.
mark = cell(2, n_targets);
for li = 1 : 2
    for ti = 1 : n_targets
        mark{li, ti} = interp1(off_fine{li, ti}, dens{li, ti}, off_int, ...
            'linear', 0);
    end
end

% =========================================================================
% 3.  Style — kept identical to model_comparison_fig4.m
% =========================================================================
lw       = 1.6;
fs       = 14;
fs_title = fs - 2;

% The in-panel labels — each panel's claim and the legend — carry the whole
% message of a schematic, so they sit at the axis-label size rather than
% being shrunk to footnote size the way in-plot annotations are on the
% quantitative figures.

style_ax = @(ax) apply_axis_style(ax, fs, lw);

clr_model_ind = {[0.00 0.45 0.70], ...   % stopping           — blue
                 [0.84 0.37 0.00], ...   % linear number line — vermillion
                 [0.00 0.62 0.45]};      % log number line    — bluish green

model_disp = {'Stopping', 'Linear number line', 'Log number line'};

% The verdict each panel exists to deliver.  Two clauses: does it separate
% the targets, and what shape is it.
model_claim = {'invariant  ·  skewed', ...
               'invariant  ·  symmetric', ...
               'NOT invariant  ·  widens with n'};

% Within a panel the two targets are told apart by line style and marker
% fill, not by colour: the colour is the model's identity and has to stay
% constant across the panel so the three panels read as one figure.
%
% Target 3 gets the LARGER open marker and target 4 the smaller filled one,
% so that where the two curves coincide the square sits inside the circle and
% both are still visible.  Equal-sized markers would let the later one hide
% the earlier one, and a reader cannot tell "identical" from "not drawn".
sty_target = {'-', '--'};
mrk_target = {'o', 's'};
msz_target = [9.5, 5.5];

fit_paper = @(h) set(h, 'PaperUnits', 'centimeters', ...
    'PaperSize', h.Position(3:4), 'PaperPosition', [0 0 h.Position(3:4)]);
fig_dir = fullfile(project_dir, 'figures');
if ~isfolder(fig_dir), mkdir(fig_dir); end

% =========================================================================
% 4.  Figure — one panel per model, shared axes
% =========================================================================
% A common y-limit across the panels is what makes the log model's lower,
% flatter target-4 curve read as a difference rather than as a rescaling.
% The axis carries no numbers: these are illustrative parameters, and printed
% values would invite exactly the quantitative reading fig9 is for.
y_max = 1.30 * max([p_stop, cellfun(@max, dens(:)')]);

% Wider than the quantitative figures: at the axis-label size each panel's
% claim is a full line of text and needs the room.
fig0 = figure('Color', 'w', 'Units', 'centimeter', 'Position', [5 5 30 10.5]);
fit_paper(fig0)

for mi = 1 : 3
    ax = subplot(1, 3, mi);  hold on
    clr = clr_model_ind{mi};

    % Reference line at offset 0.  Symmetry and skew are both claims ABOUT
    % the target, so the target needs to be visible on the axis.
    plot([0 0], [0 y_max], ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off')

    % Nothing plotted here enters the legend directly.  On the number-line
    % panels the curve and its markers are two separate objects, so an
    % automatic legend would show a bare line and drop the marker that
    % actually carries the target identity; the proxy handles below carry
    % both.  This is the same device fig9 uses.
    for ti = 1 : n_targets
        face = ternary(ti == 1, 'w', clr);

        if mi == 1
            % Simulated, no closed form, so joined markers — as in MS Fig. 4C.
            plot(off_int, p_stop, [mrk_target{ti} sty_target{ti}], ...
                'Color', clr, 'MarkerFaceColor', face, ...
                'MarkerSize', msz_target(ti), 'LineWidth', lw, ...
                'HandleVisibility', 'off')
        else
            li = mi - 1;
            plot(off_fine{li, ti}, dens{li, ti}, sty_target{ti}, ...
                'Color', clr, 'LineWidth', lw, 'HandleVisibility', 'off')
            plot(off_int, mark{li, ti}, mrk_target{ti}, 'LineStyle', 'none', ...
                'Color', clr, 'MarkerFaceColor', face, ...
                'MarkerSize', msz_target(ti), 'LineWidth', lw, ...
                'HandleVisibility', 'off')
        end
    end

    % The legend belongs to the log panel and only to it: that is the one
    % panel whose two curves come apart, so it is the only one where telling
    % target 3 from target 4 tells the reader anything.  On the other two the
    % curves coincide and the legend would be labelling a single visible
    % curve twice — which reads as if one of the two had gone missing.
    if mi == 3
        h_leg = gobjects(1, n_targets);
        for ti = 1 : n_targets
            % "n = 3" rather than "target 3": at the axis-label size the
            % longer form runs into the rising flank of the target-3 curve,
            % and the x label already reads "r - n", so n needs no gloss.
            h_leg(ti) = plot(NaN, NaN, [mrk_target{ti} sty_target{ti}], ...
                'Color', clr, 'MarkerFaceColor', ternary(ti == 1, 'w', clr), ...
                'MarkerSize', msz_target(ti), 'LineWidth', lw, ...
                'DisplayName', sprintf('n = %d', targets_schem(ti)));
        end
        % Northwest, then nudged down clear of the claim, which owns the top
        % of the panel.  The left flank is where both curves are still near
        % zero, so this is the only corner with room for it.
        lgd = legend(h_leg, 'Location', 'northwest', 'Box', 'off', ...
            'FontSize', fs, 'TextColor', 'k');
        drawnow
        lgd.Position(2) = lgd.Position(2) - 0.13 * ax.Position(4);
    end

    text(-off_lim - 0.25, y_max, model_claim{mi}, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
        'FontSize', fs, 'Color', 'k')

    xticks(off_int)
    xlim([-off_lim - 0.4, off_lim + 0.4])
    ylim([0, y_max])
    yticks([])
    xlabel('Response offset  r - n', 'FontSize', fs)
    if mi == 1
        ylabel('Response probability', 'FontSize', fs)
    end
    title(model_disp{mi}, 'FontSize', fs_title)
    style_ax(ax)
end

sgtitle(['Schematic: what each account predicts for target 3 vs target 4 ' ...
    '(illustrative parameters, not fits)'], 'FontSize', fs_title, ...
    'FontWeight', 'bold', 'Color', 'k')

exportgraphics(fig0, fullfile(fig_dir, 'fig4B_model_predictions.pdf'), ...
    'ContentType', 'vector')

% =========================================================================
% 5.  Console note
% =========================================================================
fprintf('\nSchematic parameters (illustrative — NOT fitted values)\n');
fprintf('  stopping     drift = %.2f   sigma_z = %.2f   (%d simulated trials)\n', ...
    sim_drift, sim_sigma_z, n_sim);
fprintf('  linear line  mu = n         sigma   = %.2f   (count units)\n', sig_lin);
fprintf('  log line     mu = log(n)    sigma   = %.2f   (log units)\n\n', sig_log);

fprintf('Stopping model, simulated offset distribution (%%)\n  ');
fprintf('%6d', off_int);  fprintf('\n  ');
fprintf('%6.1f', 100 * p_stop);  fprintf('\n');
m_stop  = sum(off_int .* p_stop);
sd_stop = sqrt(sum((off_int - m_stop) .^ 2 .* p_stop));
fprintf('  skew check: p(-1) = %.1f%% against p(+1) = %.1f%%   SD = %.2f\n\n', ...
    100 * p_stop(off_int == -1), 100 * p_stop(off_int == 1), sd_stop);

% Measured over the full support, not the plotted window, so the numbers
% describe the distributions rather than the crop.
fprintf('Number lines, width on the COUNT axis (SD of the full density)\n');
name = {'linear', 'log   '};
for li = 1 : 2
    fprintf('  %s ', name{li});
    for ti = 1 : n_targets
        n_tg = targets_schem(ti);
        if li == 1
            d = exp(-0.5 * ((r_norm      - n_tg)      / sig_lin) .^ 2);
        else
            d = exp(-0.5 * ((log(r_norm) - log(n_tg)) / sig_log) .^ 2);
        end
        d  = d / trapz(r_norm, d);
        m  = trapz(r_norm, r_norm .* d);
        sd = sqrt(trapz(r_norm, (r_norm - m) .^ 2 .* d));
        fprintf('  target %d: SD = %.2f', n_tg, sd);
    end
    fprintf('\n');
end

fprintf('\nWritten: %s\n\n', fullfile(fig_dir, 'fig4B_model_predictions.pdf'));

% =========================================================================
% Local functions
% =========================================================================
function out = ternary(cond, a, b)
% Inline choice, so the plot call above stays one statement per target.
if cond, out = a; else, out = b; end
end

function apply_axis_style(ax, fs, lw)
% Tick labels at the same size as the axis labels, and every piece of text on
% the axes in true black.  MATLAB defaults the axis colour to [0.15 0.15 0.15]
% and the title to the same grey, which prints noticeably lighter than the
% data; journals generally want black.
set(ax, 'TickDir', 'out', 'LineWidth', lw, 'FontSize', fs, 'Box', 'off', ...
    'XColor', 'k', 'YColor', 'k', 'ZColor', 'k')

lbl = [ax.Title, ax.XLabel, ax.YLabel];
set(lbl(isgraphics(lbl)), 'Color', 'k')
end
