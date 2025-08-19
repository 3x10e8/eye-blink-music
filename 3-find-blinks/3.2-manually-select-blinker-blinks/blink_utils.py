import mne
import numpy as np

def make_blink_annots(
        merged_raws, 
        leftZeros, 
        rightZeros, 
        peakFrames,
        description = "",
    ):

    # Onset/offset for "good" blinks identified by blinker
    onset = []
    duration = []
    
    for peak in peakFrames:
        # Find the closest leftZero (which might be to the right even)
        peakIdx = np.argmin(np.abs(peak - leftZeros)) # & (peak < rightZeros))
        #print(leftZeros[peakIdx])
        t_onset = merged_raws.times[int(leftZeros[peakIdx])]
        t_offset = merged_raws.times[int(rightZeros[peakIdx])]

        onset.append(t_onset)
        duration.append(t_offset - t_onset)
        #print(leftZeros[goodBlinkIdx], blinkPeakIdx, rightZeros[goodBlinkIdx])

    annot = mne.Annotations(
        onset = onset,
        duration = duration,
        description = description,
    )

    return annot
def plot_annot(
    merged_raws,
    boundary_annot,
    blink_summary,
    SUBJECT,
    usedSignalIdx = None,
    t_manual_blinks = None,
    t_bad_blinker = None,
    t_bad_saccades = None,
    butterfly = False,
    lowpass = None,
    highpass = None,
):
    
    blinkFits = blink_summary[SUBJECT]['blinkFits']
    maxFrame = [int(x)-1 for x in blinkFits['maxFrame']] # convert to 0 index

    # All candidate blinks (includes bad blinks)
    if usedSignalIdx == None:
        blinkPositions = blink_summary[SUBJECT]['blinks']['signalData']['blinkPositions'] -1 # zero index
    else:
        blinkPositions = blink_summary[SUBJECT]['blinks']['signalData'][usedSignalIdx]['blinkPositions'] -1 # zero index

    leftZeros = blinkPositions[0, :]
    rightZeros = blinkPositions[1, :]
        
    blinker_annot = make_blink_annots(
        merged_raws, 
        leftZeros, 
        rightZeros, 
        maxFrame,
        description = "blinker",
    )
        
    if t_manual_blinks:
        manualFrame = np.array(t_manual_blinks) * merged_raws.info['sfreq']

        manual_annot = make_blink_annots(
            merged_raws, 
            leftZeros, 
            rightZeros, 
            manualFrame,
            description = "manual",
        )
    else:
        manual_annot = mne.Annotations([], [], [])

    if t_bad_blinker:
        badFrame = np.array(t_bad_blinker) * merged_raws.info['sfreq']

        bad_annot = make_blink_annots(
            merged_raws, 
            leftZeros, 
            rightZeros, 
            badFrame,
            description = "BAD blinker",
        )
    else:
        bad_annot = mne.Annotations([], [], [])

    if t_bad_saccades:
        badSaccadeFrame = np.array(t_bad_saccades) * merged_raws.info['sfreq']

        bad_saccade_annot = make_blink_annots(
            merged_raws, 
            leftZeros, 
            rightZeros, 
            badSaccadeFrame,
            description = "saccade",
        )
    else:
        bad_saccade_annot = mne.Annotations([], [], [])

    merged_raws.set_annotations(
        boundary_annot
        + blinker_annot 
        + manual_annot 
        + bad_annot
        + bad_saccade_annot
    )

    merged_raws.plot(
        n_channels = 64,
        duration = 60,
        scalings = {'eeg': 2.5e-3},
        #highpass = 1,
        #lowpass = 20,
        #group_by = 'position',
        butterfly = butterfly,
        lowpass = lowpass,
        highpass = highpass,
        picks = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'Afz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'midi', 'expectation', 'envelope', 'tactile_cue']
    )

# Make a misc. blinks channel using blink onsets
def make_blink_channel(merged_raws):

    events, event_id = mne.events_from_annotations(merged_raws, regexp = "") # gets "BAD"s too

    # zeros of correct shape
    blinks_misc = np.zeros_like(merged_raws.times)
    sweeps_misc = np.zeros_like(merged_raws.times)

    # mark blinks found by blinker
    blinks_misc[
        events[
            events[:, 2] == event_id["blinker"], 0
            ]
    ] = 1

    # remove any manually identified bad blinks
    blinks_misc[
        events[
            events[:, 2] == event_id["BAD blinker"], 0
            ]
    ] = 0

    # add any manually identified missed blinks
    blinks_misc[
        events[
            events[:, 2] == event_id["manual"], 0
            ]
    ] = 1

    try:
        # add any manually identified missed blinks
        sweeps_misc[
            events[
                events[:, 2] == event_id["sweep"], 0
                ]
        ] = 1
    except KeyError as e:
        print(e)

    print('Total valid blinks:', int(sum(blinks_misc)))
    print('Total sweeps:', int(sum(sweeps_misc)))

    return blinks_misc, sweeps_misc

def make_trials_channel(
    merged_raws,
    event_dict = {
        "Listening/chor-038": 1,
        "Listening/chor-096": 2,
        "Listening/chor-101": 3,
        "Listening/chor-019": 4,
        "Imagery/chor-038": 11,
        "Imagery/chor-096": 12,
        "Imagery/chor-101": 13,
        "Imagery/chor-019": 14,
    },
):
    trial_events = mne.make_fixed_length_events(merged_raws, duration=1803 / merged_raws.info['sfreq'])

    for cond in ['Listening', 'Imagery']:
        for chor in ['chor-038', 'chor-096', 'chor-101', 'chor-019']: 
            cond_chor_mask = [merged_raws.filenames.index(x) for x in merged_raws.filenames if f'{chor}_condition-{cond}' in str(x)]
            trial_events[cond_chor_mask, 2] = [event_dict[f"{cond}/{chor}"]]*len(cond_chor_mask)

    trial_stim = np.zeros_like(merged_raws.times)
    trial_stim[trial_events[:, 0]] = trial_events[:, 2]
    return trial_stim

def make_new_raw(
    merged_raws,
    signal,
    usedSignal,
    note_annot,
):
    
    blinks_misc, sweeps_misc = make_blink_channel(merged_raws)
    trial_stim = make_trials_channel(merged_raws)

    eeg_and_stim_data = np.vstack(
        (
            merged_raws.get_data(), 
            signal/1e6, # IC and polarity as used by blinker
            blinks_misc, # blink events after manual verification
            sweeps_misc, # (optional) if eye-sweep annotations are available
            trial_stim, # for neat handling of trial labels
        )
    )

    ch_names = merged_raws.ch_names + [f'ica_{int(usedSignal)}', 'blinks', 'sweeps', 'trials']
    ch_types = merged_raws.get_channel_types() + ['misc', 'misc', 'misc', 'stim']

    # Make an info object
    info = mne.create_info(
        ch_names = ch_names, 
        sfreq=merged_raws.info['sfreq'], 
        ch_types=ch_types,
    )

    info.set_montage('biosemi64', match_case=False) # resolves Afz case trouble

    final_raw = mne.io.RawArray(
        data = eeg_and_stim_data, # transpose for importing to MNE
        info = info,
        verbose = 'ERROR',
    )
    
    final_raw.set_annotations(note_annot)
    final_raw.info.description = merged_raws.filenames
    return final_raw