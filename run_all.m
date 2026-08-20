function run_all(varargin)
% run_all  Reproduce every figure panel and every number this repository is
%          responsible for in the manuscript
%
%   run_all           redraws every figure and reprints every table from the
%                     fitted parameters shipped in results.
%   run_all('refit')  refits everything from the raw data first, overwriting
%                     results/, then does the same (several hours — the
%                     cross-validation alone runs for hours)
%
% Every stage is an independent script and can be run on its own, in the
% order below.

known     = {'refit', 'fits', 'skipfits'};
bad       = varargin(~ismember(lower(varargin), known));
if ~isempty(bad)
    error('run_all:unknownOption', ...
        'Unknown option "%s".  Use run_all or run_all(''refit'').', bad{1});
end
skip_fits = ~any(ismember(lower(varargin), {'refit', 'fits'}));

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'scripts'), fullfile(here, 'helpers'));

t_all = tic;

if skip_fits
    fprintf(['\nUsing the fitted parameters in results/ — nothing is ' ...
        'refitted.\nPass run_all(''refit'') to fit everything from the raw ' ...
        'data instead.\n']);
else
    fprintf(['\nrefit: results/ will be overwritten with new fits.  The ' ...
        'shipped fits are\nthe ones the manuscript reports; a refit of the ' ...
        'RT model in particular\nlands on a different optimum.\n']);
end

% ---------------------------------------------------------------------
% 1.  Fits to the response frequencies
% ---------------------------------------------------------------------
if ~skip_fits
    run_stage('fit_diffusion_mle_MS', ...
        'Stopping model fitted to response frequencies (~80 min)');
    run_stage('fit_gaussian_mle_MS', ...
        'Number-line models fitted to response frequencies (seconds)');

    % -----------------------------------------------------------------
    % 2.  Leave-one-session-out cross-validation
    % -----------------------------------------------------------------
    run_stage('cv_models_MS', ...
        'Cross-validation over the 76 sessions (hours)');
    run_stage('cv_pooled_vs_animal_MS', ...
        'Cross-validation, pooled vs per-animal parameters (hours)');
end

% ---------------------------------------------------------------------
% 3.  Model comparison — MS Fig. 4C, 4D, 4E and the Results/Methods tables
% ---------------------------------------------------------------------
run_stage('model_comparison_fig4', ...
    'Model comparison — MS Fig. 4C, 4D, 4E');

% ---------------------------------------------------------------------
% 4.  Between-session variability — the Methods numbers that justify
%     treating the session as the unit of analysis
% ---------------------------------------------------------------------
run_stage('session_variability_MS', ...
    'Between-session variability (prints only)');

% ---------------------------------------------------------------------
% 5.  What the three accounts predict — MS Fig. 4B
% ---------------------------------------------------------------------
run_stage('schematic_row2_MS', ...
    'Predicted response distributions — MS Fig. 4B');

% ---------------------------------------------------------------------
% 6.  Forward simulation and the RT figures — MS Fig. 3E, 4A, 4F, 4G,
%     plus the session-level RT trend and asymmetry tests
% ---------------------------------------------------------------------
run_stage('diffusion_model_fig4', ...
    'Forward simulation and RT figures — MS Fig. 3E, 4A, 4F, 4G');

% ---------------------------------------------------------------------
% 7.  Reaction-time fit and MS Table 1
% ---------------------------------------------------------------------
if ~skip_fits
    run_stage('fit_diffusion_rt_MS', ...
        'Stopping model fitted to RT distributions (~20 min)');
end

run_stage('compare_rt_fit_MS', ...
    'RT goodness of fit, accuracy-fit vs RT-fit — MS Table 1');

fprintf('\n\nAll done in %.1f min.\n', toc(t_all) / 60);
fprintf('Figures: %s\n', fullfile(here, 'figures'));
fprintf('Results: %s\n', fullfile(here, 'results'));
end


function run_stage(script_name, description)
% Each stage is run from inside this function rather than from run_all's own
% workspace: every stage script begins with `clear`, which would otherwise
% wipe run_all's variables halfway through the pipeline.  Nothing after the
% eval may rely on this function's locals for the same reason.
fprintf('\n\n%s\n', repmat('=', 1, 78));
fprintf('  %s\n', description);
fprintf('  (%s.m)\n', script_name);
fprintf('%s\n\n', repmat('=', 1, 78));
eval(script_name);
end
