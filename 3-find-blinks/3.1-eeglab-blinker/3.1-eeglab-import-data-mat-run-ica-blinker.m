%% Script for running the BLINKER toolbox to extract blinks from the musicImagery dataset
%
% Pre-requisites:
% 1. Download the musicImagery dataset from https://www.data.cnspworkshop.net/data/datasetCND_musicImagery.zip
%   - This step is covered in 1-source-data/1-download-dataset.ipynb 
% 2. Clone the EEGLAB repository on your machine: https://github.com/sccn/eeglab
% 3. Install the ICLabel and BLINKER plugins using EEGLAB's plugin manager. 
%   - See EEGLAB documentation for installing plugins: https://eeglab.org/others/EEGLAB_Extensions.html#to-install-or-update-a-plugin
% 4. Verify that the installed BLINKER plugin inclues eyecatch mat files.
%   - Check under your path to eeglab/plugins/Blinker1.2.0/utilities/+pr/private/
%   - If these files are missing, they can be downloaded from BLINKER's repo: https://github.com/VisLab/EEG-Blinks/tree/master/blinker/utilities/%2Bpr/private
%   - If issues still persist, see the eyecatch repo: https://github.com/bigdelys/eye-catch/tree/master/private
% 
% This script is organized as follows:
% 1. Add paths to eeglab, BLINKER toolbox, and the CND musicImagery dataset files (.mat)
% 2. The musicImagery dataset is structured into trials. We load these trials for each subject as "epochs" for both listening/imagery conditions.
%   - Optionally, we can run ICA on epochs and visualize the ICs stacked over trials.
% 3. Merge all 88 trials (here epochs) from each subject into one contiguous dataset per subject. This is done as BLINKER doesn't run on epochs.
% 4. Run ICA and ICLabel on these merged trials per subject, and identify top eye-related ICs.
% 5. Although BLINKER supports running directly on ICs, we had trouble doing so. Instead we refactor the ICs into a new EEGLAB dataset.
%   - For more info on this issue, see https://github.com/VisLab/EEG-Blinks/issues/9
% 6. Run BLINKER and save blink summaries out for downstreaming processing.
%
% Please open an issue or email auppal [AT] ucsd [DOT] edu if you run into any problems.
%   - Pre-run BLINKER results can also be made available if needed. These files were ~MB so aren't committed to this repo.

%% Start MATLAB from the eye-blink-music repo's root directory
%cd /path/to/eye-blink-music

%% Setup paths
% Add eeglab to path
%addpath /path/to/eeglab 

% For example, if EEGLAB was cloned in eye-blink-music's parent directory
addpath ../eeglab

%% launch EEGLAB first to verify installation and install the BLINKER plugin
%eeglab

%% Once installed, add the BLINKER plugin to MATLAB's path
addpath ../eeglab/plugins/Blinker1.2.0/utilities/+pr/private/
% Verify that eyecatch mat files were correctly installed, or check instructions above for troubleshooting

%% Load stim mat file from dataset
stim_path = './1-source-data/datasetCND_musicImagery/dataCND/dataStim.mat';
load(stim_path)

% stimIdx to name mapping was missing in the stim struct
% Identified stimIdx names from matching sheet music to provided events
% 2: 'chor-038', 
% 1: 'chor-096', 
% 3: 'chor-101',
% 4: 'chor-019', 

stimNames = {};
stimNames{1} = 'chor-096';
stimNames{2} = 'chor-038';
stimNames{3} = 'chor-101';
stimNames{4} = 'chor-019';

%% Load one subject's data to start with, could eventually load multiple
for subject = 1 %:21
 
    data_path = ['./1-source-data/datasetCND_musicImagery/dataCND/dataSub', num2str(subject), '.mat'];
    data_path = join(data_path);
    load(data_path)
    
    %% Merge all trials into one 3D matrix
    data = []; % nbchan x points x trials, from https://sccn.ucsd.edu/~arno/eeglab/auto/eeglab.html
    trialsval = {}; % trial labels, trial and epoch appears to be used interchangeably in EEGLAB
    
    trialIdx = 1;
    for origTrialIdx = eeg.origTrialPosition
        
        % eeg.data{trial_idx} is 1803x64 (time x chans)
        % transpose to get chans x time 
        data(:, :, trialIdx) = eeg.data{origTrialIdx}';
    
        % Identify the corresponding stimulus and condition 
        stimIdx = stim.stimIdxs(origTrialIdx);
        stimName = stimNames{stimIdx};
        condIdx = stim.condIdxs(origTrialIdx);
        condName = stim.condNames{condIdx};
        condition = [condName, '_', stimName];
    
        trialsval{trialIdx, 1} = condName;
        trialsval{trialIdx, 2} = stimName;
    
        trialIdx = trialIdx + 1;
    
    end
    
    %% Import this data into EEGLAB as an epoched set
    
    EEG_EPS = pop_importdata('setname', 'musicImagery' ...
            , 'data', data ...
            , 'subject', subject ...
            , 'condition', 'all' ...
            , 'session', 1 ...
            , 'nbchan', 64 ...
            , 'chanlocs', eeg.chanlocs ...
            , 'srate', eeg.fs ...
            );
            %, 'ref', '', );
    
    EEG_EPS.epoch = trialsval'; % append trial descriptors (is EEG.trialsval not supported?)
    EEG_EPS.epochdescription = {'condName'; 'stimName'}; % EEG.epoch column descriptors
    EEG_EPS.filename = ['dataSub', num2str(subject), '.mat']; 
    EEG_EPS, changes = eeg_checkset(EEG_EPS); % check for errors?

    %% Optional: run ICA and examine ICs over trials
    if 0
        EEG_EPS = pop_runica(EEG_EPS, 'icatype', 'runica')
        % pop_eegplot(ALLEEG, 0)

        %% Now relaunch GUI to run ICLabel
        %eeglab redraw
        
        % Run ICLabel
        EEG_EPS = pop_iclabel(EEG_EPS, 'default'); % fails for epoched ICs?
        [pClass, eyeICidxs] = sortrows(...
            EEG_EPS.etc.ic_classification.ICLabel.classifications,...
            3, 'descend');
        topEyeICidxs = eyeICidxs(1:3); %(pClass(:, 3)>0.9) % >90% eye class

        % plot IC properties
        pop_prop(EEG_EPS, 0, topEyeICidxs(1))
    end

    %% Merge all trials into one dataset
    for origTrialIdx = eeg.origTrialPosition
        
        % eeg.data{trial_idx} is 1803x64 (time x chans)
        % transpose to get chans x time (for EEGLAB import below)
        data = eeg.data{origTrialIdx}';
    
        % Identify the corresponding stimulus and condition 
        stimIdx = stim.stimIdxs(origTrialIdx);
        stimName = stimNames{stimIdx};
        condIdx = stim.condIdxs(origTrialIdx);
        condName = stim.condNames{condIdx};
        condition = [condName, '_', stimName];
    
        EEG_TRIAL = pop_importdata('setname', 'musicImagery' ...
            , 'data', data ...
            , 'subject', subject ...
            , 'condition', condition ...
            , 'session', 1 ...
            , 'nbchan', 64 ...
            , 'chanlocs', eeg.chanlocs ...
            , 'srate', eeg.fs ...
        );
            %, 'ref', '', );
    
        if origTrialIdx == eeg.origTrialPosition(1)
            ALLEEG = EEG_TRIAL;
        else % append the new trial's data
            ALLEEG = pop_mergeset(ALLEEG, EEG_TRIAL); % looses condition info
        end
    
    end
    
    %% Run ICA
    ALLEEG = pop_runica(ALLEEG, 'icatype', 'runica')
    % pop_eegplot(ALLEEG, 0)

    %% Now relaunch GUI to run ICLabel
    %eeglab redraw
    ALLEEG = pop_iclabel(ALLEEG, 'default');
    [pClass, eyeICidxs] = sortrows(...
        ALLEEG.etc.ic_classification.ICLabel.classifications,...
        3, 'descend');
    topEyeICidxs = eyeICidxs(pClass(:, 3)>0.9) % >90% eye class

    %% Swap in ICA components for regular EEG channels
    icdata = eeg_getdatact(ALLEEG, 'component', [1:size(ALLEEG.icaweights,1)]);
    eyeICdata = icdata(topEyeICidxs, :);
    EEG_EYE_ICs = pop_importdata('setname', 'musicImagery' ...
        , 'data', eyeICdata ...
        , 'subject', subject ...
        , 'condition', condition ...
        , 'session', 1 ...
        , 'nbchan', length(topEyeICidxs) ...
        , 'chanlocs', [] ...
        , 'srate', eeg.fs ...
    );
    
    %% Run blinker (doesn't take epoched data, expects continuous input)
    params = checkBlinkerDefaults(struct(), getBlinkerDefaults(EEG_EYE_ICs));
    
    params.subjectID = num2str(subject);
    params.uniqueName = 'allTrials';
    params.experiment = 'musicImagery';
    params.task = 'allConditions';
    %params.startDate
    %params.startTime
    
    params.signalTypeIndicator = 'UseNumbers';
    params.signalNumbers = 1:length(topEyeICidxs);
    %params.signalTypeIndicator = 'UseICs';
    %params.signalNumbers = [3, 8]; % if override is needed to manually select eye ICs
    
    %params.signalLabels = {''};
    %params.excludeLabels = {''};
    
    params.dumpBlinkerStructures = true;
    params.showMaxDistribution = true;
    params.dumpBlinkImages = false;
    params.verbose = true;
    params.dumpBlinkPositions = true;
    
    params.fileName = ['dataSub', num2str(subject), '.mat']; 
    params.blinkerSaveFile = ['dataSub', num2str(subject), '_BlinkSummary.mat'];
    
    params.blinkerDumpDir = '3-find-blinks/3.1-eeglab-blinker/blinkerDumpDir';
    params.keepSignals = false;
    
    [OUTEEG, com, blinks, blinkFits, blinkProperties, ...
                         blinkStatistics, params] = pop_blinker(...
                         EEG_EYE_ICs ...
                         , params);
    
    %% Print blinker stats
    blinks
    blinkStatistics
    
    %% Plot the signals if needed
    %     figure()
    %     blinkSignal = blinks.signalData.signal;
    %     plot(blinkSignal); hold on
    %     
    %     blinkPositions1 = blinks.signalData.blinkPositions(1, :);
    %     blinkPositions2 = blinks.signalData.blinkPositions(2, :);
    %     
    %     plot(blinkPositions1, blinkSignal(blinkPositions1), 'rx')
    %     plot(blinkPositions2, blinkSignal(blinkPositions2), 'gx')
end