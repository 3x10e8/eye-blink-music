from decimal import DivisionByZero
import os, mne
from glob import glob
import numpy as np
#import pyspike as spk # pip install pyspike errors out in the eegeyenet-env...
# https://stackoverflow.com/a/21530768
from matplotlib.gridspec import GridSpec
import elephant
import quantities as pq
import matplotlib.pyplot as plt
import pandas as pd

# https://neo.readthedocs.io/en/0.7.2/api_reference.html#neo.core.SpikeTrain
from neo.core import SpikeTrain

# https://medium.com/@gubrani.sanya2/evaluating-multi-class-classification-model-using-confusion-matrix-in-python-4d9344084dfa
from sklearn.metrics import confusion_matrix
import seaborn as sns

def scraws_by_subject(
    DATA_PATH, 
    SUBJECT='*',
    FILENAME='_blinks_merged_raw',
):
    DATA_FILES = glob(DATA_PATH + os.path.sep + f'subject-{SUBJECT}{FILENAME}.fif')
    DATA_FILES = sorted(DATA_FILES) # to sort by run#
    print(f'Found {DATA_FILES}')

    RAWS_SUB = {}

    if len(DATA_FILES) > 0:
        for DATA_FILE in DATA_FILES: #[:2*5]:
            
            raw = mne.io.read_raw_fif(DATA_FILE, preload=True)
            fname = str(raw.filenames[0]).split(os.path.sep)[-1].rstrip('_raw.fif')
            
            toks = fname.split('_')
            subject = int(toks[0].split('-')[-1])

            if not subject in RAWS_SUB:
                RAWS_SUB[subject] = raw
            
            #print(fname, subject, condition, run)
        #print(RAWS_SUB)

    else:
        print(f'No FIF files found! Check path: {DATA_PATH}')            
    return RAWS_SUB

def trials_as_epochs(
    raw, 
    stim_channel = 'trials',
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
    picks = ['trials', 'blinks'], # excludes EEG channels
    tmin = 0, # first sample
    tmax = 1802 / 64, # 1803rd sample
):
    
    trial_events = mne.find_events(
        raw,
        stim_channel = stim_channel,
        initial_event = True,
    )

    #event_id_order = np.argsort(trial_events[:, -1])
    #print(trial_events[order, -1]) # sanity check

    # Sort trial order first by condition, then by experiment order (time)    
    df = pd.DataFrame(trial_events, columns = ['start', 'stop', 'event'])
    df_sorted = df.sort_values(by=['event', 'start'])
    event_id_order = df_sorted.index

    print(picks)

    epochs = mne.Epochs(
        raw,
        events = trial_events,
        event_id = event_dict,
        tmin = tmin, 
        tmax = tmax, # for 1803 total time points 
        baseline = None,
        picks = picks, 
        reject = None,
        flat = None,
        reject_by_annotation = False,
        preload = True,
    )

    return epochs, event_id_order

def plot_image_blinks(
    epochs,
    subjectID,
    order, # sorting order by trial event IDs
    v = [1]*int(64/(100/60)), # conv rectangle
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
    sweep_events = [], # use for overlay if available
    quantize_colorbar = False, 
):

    epochs_blink_sq = epochs.copy()
    epochs_blink_sq.apply_function(
        np.convolve, 
        picks = 'blinks',
        v = v,
        mode = 'same',
    )

    if sweep_events != []:
        overlay_times = [sweep_events[j, 0] - j*1803 for j in range(88)]
        overlay_times = np.array(overlay_times) / epochs_blink_sq.info['sfreq']
    else:
        overlay_times = None

    epochs_blink_sq.plot_image(
        picks=['blinks'],
        cmap = 'Reds',
        vmin = 0,
        vmax = 2,
        title = f"Subject {subjectID}: Trials in original experimental order",
        ts_args = {'ylim': {'eog': [0, 5]}},
        overlay_times = overlay_times,
        #order = sweep_order, #order,
    )

    figs = epochs_blink_sq.plot_image(
        picks=['blinks'],
        cmap = 'Reds',
        vmin = 0,
        vmax = 2,
        title = f"Subject {subjectID}: Trials sorted by condition/chorale",
        ts_args = {'ylim': {'eog': [0, 5]}},
        overlay_times = overlay_times,
        order = order[::-1],
        show = False,
    )
    # add epoch group boundaries
    ax_cloud = figs[0].get_axes()[0]
    img = ax_cloud.get_images()[0]
    for n in range(0, 88, 11):
        ax_cloud.axhline(y=n-0.5, c='gray', lw=1, ls='-')
    ax_cloud.set_yticks(
        ticks = np.arange(0, 88, 11) + 11/2,
        labels = list(event_dict.keys())[::-1],
    )
    ax_cloud.set_ylabel('Trials')

    ax_erp = figs[0].get_axes()[1]
    ax_erp.set_ylabel('Mean Blink Count')
    ax_erp.set_ylim([0, 0.5])
    
    ax_colorbar = figs[0].get_axes()[-1]
    ax_colorbar.set_title('Blinks')
    max_count = int(np.max(np.max(epochs_blink_sq.get_data(picks='blinks')[:, 0, :])))
    print(max_count)
    
    #figs[0].show()
    
    if quantize_colorbar:
        cmap = plt.get_cmap('Reds', max_count)
        try:
            plt.colorbar(img, cax = ax_colorbar, values = range(max_count), ticks = range(max_count))
        except ValueError as e:
            print(e)
        
    figs[0].show()

    if sweep_events != []:
        sweep_order = np.argsort(overlay_times)

        epochs_blink_sq.plot_image(
            picks=['blinks'],
            cmap = 'Reds',
            vmin = 0,
            vmax = 2,
            title = f"Subject {subjectID}: Trials sorted by end-of-line sweep",
            ts_args = {'ylim': {'eog': [0, 5]}},
            overlay_times = overlay_times,
            order = sweep_order, #order,
        )

    for chor in ['chor-038', 'chor-096', 'chor-101', 'chor-019']:
        for cond in ['Listening', 'Imagery']:
            epochs_blink_sq[f"{cond}/{chor}"].plot_image(
                picks=['blinks'],
                cmap = 'Reds',
                vmin = 0,
                vmax = 2,
                title = f"{cond}/{chor}/blinks",
                ts_args = {'ylim': {'misc': [0, 5]}}
            )

def bin_blinks(
        blinks,
        bin_width = int(64/(100/60)), # qtr note
    ):

    n_chans, n_times = blinks.shape
    #bins = [0]
    #bins.extend([e for e in range(int(bin_width/2), n_times, bin_width+1)]) # these bins will be uniformly sized
    bins = []
    bins.extend([e for e in range(0, n_times, bin_width+1)]) # these bins will be uniformly sized
    binned_blinks = np.add.reduceat(blinks, bins, axis=1) # https://stackoverflow.com/a/29391999
    
    return bins, binned_blinks

def fit_λ(binned_blinks_all_trials):

    # Get number of trials (rows)
    T, _ = binned_blinks_all_trials.shape
    #print(T)
    
    # Fit λ
    λ = np.sum(binned_blinks_all_trials, axis=0) / T # average blinks per bin
    
    # Probability of blinking atleast once during a bin interval
    #p_blinked = 1 - np.exp(-λ) # https://stats.stackexchange.com/q/306915
    #return p_blinked

    return λ

def plot_binned_blinks_prob(
    epochs,
    subjectID, # to get peaks
    event_dict,
    condLabel, stimLabel, 
    bin_width,
    imgs,
    bpm = 100,
):

    fig = plt.figure(figsize=(8, 4+2))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    ax = plt.subplot(gs[0]) #figsize=(8, 4))

    img_idx = 2*(event_dict[f'Listening/{stimLabel}']-1)
    img_h, img_w, _ = imgs[img_idx].shape
    
    if stimLabel == 'chor-038':
        time_sig_offset = -1.2
        num_bars = 9
    elif stimLabel == 'chor-096':
        time_sig_offset = -1.2
        num_bars = 9
    elif stimLabel == 'chor-101':
        time_sig_offset = -2.5
        num_bars = 8
    elif stimLabel == 'chor-019':
        time_sig_offset = -1.8
        num_bars = 9

    ax.imshow(imgs[img_idx], extent = [time_sig_offset, 
                                  num_bars/12*1803/64, 
                                  12, 
                                  12 + img_h * (num_bars/12*1803/64 - time_sig_offset)/img_w])
    img_h, img_w, _ = imgs[img_idx+1].shape
    ax.imshow(imgs[img_idx+1], extent = [num_bars/12*1803/64, 
                                  1803/64, 
                                  12, 
                                  12 + img_h * (num_bars/12*1803/64 - time_sig_offset)/img_w])
    
    # Plot notes and beats
    midi = epochs[f'{condLabel}/{stimLabel}'].get_data(
        picks=['midi'], copy=False)[0, 0, :]
    notes = np.where(midi != 0)[0]

    tactile_cue = epochs[f'{condLabel}/{stimLabel}'].get_data(
        picks=['tactile_cue'], copy=False)[0, 0, :]
    beats = np.where(tactile_cue != 0)[0]

    fs_Hz = epochs.info['sfreq']
    t = epochs.times

    blinks = epochs[f'{condLabel}/{stimLabel}'].get_data(
        picks = ['blinks'], 
        copy = True,
    )[:, 0, :] # reduce to two dims: n_epochs x n_times

    bins, agg_blinks = bin_blinks(blinks, bin_width)
    density_blinks = fit_λ(agg_blinks)
    N_trials = agg_blinks.shape[0]

    ax.plot(
        t[beats],
        0 * t[beats] + N_trials + 1,
        "o",
        label="Tactile Cue",
        c='C1',
    )

    ax.plot(
        t[notes],
        0 * t[notes] + N_trials + 1,
        ".",
        label="Note",
        c='C0'
    )

    trialCnt = 0
    
    for i in range(N_trials):
        agg_blinks_i = agg_blinks[i, :]
        peaks = np.where(blinks[i] != 0)[0]
        print(subjectID, i, peaks)
        
        # Plot the binned blink counts
        ## First add one more bin edge for the last step
        x = t[bins]
        x = np.append(x, t[-1])
        y = agg_blinks_i / 4 + trialCnt
        y = np.append(y, y[-1])

        line, = ax.step(  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.step.html
            x=x,
            y=y,
            where="post",  # [x[i], x[i+1]) has the value y[i]
        )

        # Draw a line representing the trial
        ax.axhline(
            y=trialCnt,
            ls="--",
            c="k",
            lw=0.5,
            alpha=0.3,
        )

        # Plot the actual blinks for reference
        ax.plot(
            t[peaks],
            0 * t[peaks] + trialCnt,
            "x",
            c=line.get_color(),
        )

        trialCnt += 1

    # Draw bin boundaries
    for t_bin in t[bins]:
        ax.axvline(
            x=t_bin,
            ls="--",
            c="k",
            lw=0.5,
            alpha=0.3,
        )

    # Twin axis for probability plot
    axp = ax.twinx()

    # Draw prob on a separate y-axis
    centers = list(t[bins] + bin_width / 2 / fs_Hz)
    #centers[0] = 0 # fix first center since its half a bin
    width = [np.diff(t[bins])[1]]*len(centers)
    bars = axp.bar(
        x=centers,
        height=density_blinks,
        width=width, #np.diff(t[bins]), #bin_width / fs_Hz,
        alpha=0.1,
        label="Probability of Blinking"
        # where = 'post', # [x[i], x[i+1]) has the value y[i]
    )

    # Formatting
    ax.set_xlabel("Time [seconds]")
    ax.set_yticks(range(0, 11))
    ax.set_yticklabels(epochs[f'{condLabel}/{stimLabel}'].selection)
    ax.set_ylabel("Trial #")
    ax.legend(
        # loc='upper right',
        ncols=2,
        bbox_to_anchor=[0.375, -0.05],
    )

    bps = bpm/60
    bin_qtr_notes = bin_width / (fs_Hz / bps)
    title_str = f"Subject {subjectID}, {condLabel}, {stimLabel}\n"
    title_str += "Bin Size = {:.2f} x 1/4 notes = {:.3f} seconds".format(
        bin_qtr_notes, bin_width / fs_Hz
    )
    ax.set_title(title_str)

    ax.set_xlim([0, t[-1]])

    #axp.set_ylim([0, 1])
    axp.legend(bbox_to_anchor=[1.01, -0.05])
    axp.tick_params(axis="y")  # , colors=bars[0].get_facecolor()) #, alpha=1)
    #axp.set_ylabel("Pr[$b_k > 0$]", color=bars[0].get_facecolor(), alpha=1)
    axp.set_ylabel("$\lambda_i$", color=bars[0].get_facecolor(), alpha=1)
    axp.set_ylim([0, 1.5])

    #fig2, axd = plt.subplots(figsize=(4, 4), sharex=ax)
    axd = plt.subplot(gs[1], sharex=ax)

    spike_trains = [spk.SpikeTrain(np.where(blinks[x,:]!=0)[0]/64, [epochs.tmin, epochs.tmax]) for x in range(blinks.shape[0])]

    for fun in [spk.isi_profile, spk.spike_profile, spk.spike_sync_profile]:
        profile = fun(spike_trains)
        x, y = profile.get_plottable_data()
        axd.plot(x, y, '-', label=fun.__name__)
        #axd.set_ylim([0, 1])
        axd.set_ylabel('Dissimilarity')
    #axd.legend()

    #fig.add_subplot()

def make_blink_trains(epochs, order):
    blinks = epochs.get_data(
        picks = 'blinks',
    )[:, 0, :][order]

    blink_trains = []
    for blink_train in blinks:
        blink_times = np.where(blink_train !=0)[0] / epochs.info['sfreq']
        blink_trains.append(
            SpikeTrain(
                times = blink_times, 
                units = 's',
                t_stop = epochs.tmax,
            )
        )

    return blink_trains

def make_confusion_matrix(y_true, y_pred):

    cm_labels = np.sort(np.unique(y_true))

    cm = confusion_matrix(
        y_true = y_true,
        y_pred = y_pred,
        labels = cm_labels,
    )

    return cm, cm_labels

def plot_confusion_matrix(
    cm, cm_labels, event_dict, ax,
    show_xticklabels = True, 
    show_yticklabels = True, 
    title = 'Confusion Matrix', 
    xlabel = 'Actual',
    ylabel = 'Prediction',
    vlim=[0, 11], 
    cbar=False,
):

    inv_event_dict = dict((v, k) for k, v in event_dict.items()) # assumes unique entries: https://stackoverflow.com/a/483685
    if show_xticklabels:
        xticklabels = [inv_event_dict[i].split('/')[-1] for i in cm_labels]
    else:
        xticklabels = []

    if show_yticklabels:
        yticklabels = [inv_event_dict[i].split('/')[-1] for i in cm_labels]
    else:
        yticklabels = []

    sns.heatmap(
        cm, 
        annot=True,
        fmt='d', 
        cmap='YlGnBu', 
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        square = True,
        ax = ax,
        vmin = vlim[0],
        vmax = vlim[1],
        cbar = cbar,
    )
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_title(title,fontsize=12)
    #plt.show()
    
def get_vp_distance(blink_trains, tau_s, y_true):

    if tau_s == 0:
        q_Hz = np.inf
    elif tau_s == np.inf:
        q_Hz = 0
    else:
        q_Hz = 1/tau_s
    
    # https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_dissimilarity.py#L50
    vp = elephant.spike_train_dissimilarity.victor_purpura_distance(
        spiketrains = blink_trains,
        cost_factor = q_Hz * pq.Hz,
        kernel = None,
        sort = True,
        algorithm = 'fast', # same accuracy as 'intuitive'
    )

    score = np.matrix.argsort(np.matrix(vp), axis=1) # first col is the (i, i)th element index

    y_pred = []
    true_positive = 0
    for i in range(len(blink_trains)):
        cat = y_true[i] #int(i/11) # 0-indexed
        #nearest = score[i, 0] # sanity check: 100%
        nearest = score[i, 1] # second column has the actual nearest index #score[1, i]
        nearest_cat = y_true[nearest] #int(nearest/11)
        y_pred.append(nearest_cat)

    cm, cm_labels = make_confusion_matrix(y_true, y_pred)
    true_positive = np.matrix.trace(cm)

    accuracy = true_positive / len(blink_trains) * 100
    try:
        print(f'{round(accuracy, 2)}%\t{true_positive}/{len(blink_trains)}\tq={round(q_Hz, 2)}/s\tt={round(1/q_Hz, 2)}s\n{cm}')
    except ZeroDivisionError as e:
        print(f'{round(accuracy, 2)}%\t{true_positive}/{len(blink_trains)}\tq={round(q_Hz, 2)}/s\tt=$\infty$ s\n{cm}')
    
    return {'vp': vp, 'acc': accuracy, 'cm': cm, 'cm_labels': cm_labels, 'y_pred': y_pred, 'y_true': y_true}

def sweep_tau(
        epochs, 
        order, 
        include_events = [],
        tau_list = []
    ):

    blink_trains = make_blink_trains(epochs, order)

    ev_mask = []
    for ev in include_events:
        idxs = np.where(epochs.events[order][:, -1] == ev)[0]
        ev_mask.extend(idxs)
    #print(ev_mask)
    
    y_true = epochs.events[order][ev_mask, -1]    
    results = {}
    for tau in tau_list: #[0, 0.15, 0.3, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2]:
        results[tau] = get_vp_distance(
            [blink_trains[n] for n in ev_mask], # can't seem to apply mask directly to list
            tau,
            y_true,
        )

    return results

def plot_results(subjectID, event_dict, results_listen, results_imagery):

    tau_list = list(results_imagery.keys())

    fig, ax = plt.subplots(2, 1)

    acc_listen = [results_listen[tau]['acc'] for tau in tau_list]
    tau_max_listen = tau_list[np.argmax(acc_listen)]

    acc_imagery = [results_imagery[tau]['acc'] for tau in tau_list]
    tau_max_imagery = tau_list[np.argmax(acc_imagery)]

    line_listen, = ax[0].plot(tau_list, acc_listen, '.-', label='Listen')
    line_imagery, = ax[0].plot(tau_list, acc_imagery, '.-', label='Imagery')

    ax[0].axvline(x=tau_max_listen, lw=1, c=line_listen.get_color(), ls='-')
    ax[0].axvline(x=tau_max_imagery, lw=1, c=line_imagery.get_color(), ls='-')

    ax[0].axhline(y=1/4*100, c='r', label='chance', ls='-')
    for n_bars in range(12):
        ax[0].axvline(x=0.6*4*n_bars, lw=1, c='k', ls=':')


    ax[0].legend(loc='upper right')
    ax[0].set_xlabel('Timescale (= 1/q) [s]')
    ax[0].set_ylabel('Accuracy %')

    ax[0].set_title(f'Subject {subjectID}: argmin(distance(1 test train, 43 ref trains))')
    ax[0].set_ylim([0, 100])

    ax[1].remove()
    ax_cm_listen = fig.add_subplot(2, 2, 3)

    plot_confusion_matrix(
        results_listen[tau_max_listen]['cm'],
        results_listen[tau_max_listen]['cm_labels'],
        event_dict,
        ax = ax_cm_listen,
        title = f'Listening Trials\nConfusion Matrix\nq={round(1/tau_max_listen, 2)}Hz, 1/q=' + f'{tau_max_listen}s'
    )

    ax_cm_imagery = fig.add_subplot(2, 2, 4)

    plot_confusion_matrix(
        results_imagery[tau_max_imagery]['cm'],
        results_imagery[tau_max_imagery]['cm_labels'],
        event_dict,
        ax = ax_cm_imagery,
        title = f'Imagery Trials\nConfusion Matrix\nq={round(1/tau_max_imagery, 2)}Hz, 1/q=' + f'{tau_max_imagery}s'
    )

    plt.tight_layout()
    plt.show()

def plot_info(results_matrix, ax, rowLabels, colLabels, title= "", vmin = None, vmax=None, fontsize=4, cmap='magma'):
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    # fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(results_matrix, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(colLabels)), labels=colLabels)
    ax.set_yticks(np.arange(len(rowLabels)), labels=rowLabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor", fontsize=fontsize)
    '''
    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            text = ax.text(j, i, results_matrix[i, j],
                        ha="center", va="center", color="w")
    '''

    for offset in range(0, results_matrix.shape[0], 11):
        plt.axvline(x = offset-0.5, c='w')
        plt.axhline(y = offset-0.5, c='w')

    ax.set_title(title)
    plt.colorbar(im) #, ax=ax)
    plt.tight_layout()
    #plt.show()