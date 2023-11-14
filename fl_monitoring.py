import numpy as np
import time
import asyncio
from matplotlib import colormaps, rcParams
import matplotlib.pyplot as plt

from IPython.display import clear_output, display_pretty

from fl_visualisation import plot_metrics, plot_series


class ProgressReporter:
    """a simple class for tracking loss/accuracy metrics of a model
    as it trains, and reporting them as a real-time display."""
    def __init__(self,
                 num_epochs: int,      # expected number of epochs the model will train for
                 steps_per_epoch: int, # expected number of batches processed in each epoch
                 time_interval:float = None, # time (in seconds) between displayed outputs
                ):

        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self.time_interval = time_interval
        
        # lists to be filled at each training step:
        self.metrics = {'train_losses': [],
                        'val_losses':   [],
                        'train_accs':   [],
                        'val_accs':     [],
                        'prox_losses':  [],
                        'lrs':          []}
        self.indices = {'epoch':        [],
                        'batch':        [],
                        'update_time':  []}

        # attributes that are updated at each report step:
        self.last_report_time = time.time()
        self.last_report_steps = 0
        self.last_report = None # will contain the last full display, so that it 
                                # can be shown again if called after training is complete
        
        self.epoch_start_time = time.time()
        self.epoch_end_time = time.time()
        self.current_epoch = 0

        self.active = False # flag used for asynchronous control flow
        
        self.aggregator = None # optionally, this may be used as one individual reporter
                               # in an aggregate collection of reporters each tracking a
                               # different model, for federated asynchronous training.
                               # see JointProgressReporter class. 
        

    def update(self, tloss, tacc, vloss, vacc, 
               prox_loss = None, lr = None):
        """we expect this to be called at every training step;
        updates internal metrics with latest loss/accuracy numbers"""
        
        self.active = True # flag that this reporter is currently receiving updates
        
        self.metrics['train_losses'].append(tloss)
        self.metrics['train_accs'].append(tacc)
        self.metrics['val_losses'].append(vloss)
        self.metrics['val_accs'].append(vacc)
        if prox_loss is not None:
            self.metrics['prox_losses'].append(prox_loss)
        if lr is not None:
            self.metrics['lrs'].append(lr)
            
        epoch, batch = divmod(self.step, self.steps_per_epoch)
        cur_time = time.time()
        self.indices['epoch'].append(epoch)
        self.indices['batch'].append(batch)
        self.indices['update_time'].append(cur_time)

        if self.aggregator is None:
            # automatically report if a certain amount of time has elapsed
            elapsed = cur_time - self.last_report_time
            if elapsed > self.time_interval:
                self.report()
            # or if this is the last batch of the last epoch:
            if epoch == self.num_epochs:
                self.report()
                print('(final reporter update)')
        else:
            # if an aggregator is defined, tell it that we've updated.
            
            # if learning rates are being supplied, display the learning rate as well:
            if len(self.metrics['lrs']) > 0:
                current_lr = self.metrics['lrs'][-1]
                lr_str = f'current learning rate: {current_lr:.0e}'
                temp_messages = [lr_str]
            else:
                temp_messages = []
                
            self.aggregator.notify(self, temp_messages)
            # this suppresses the individual reporter's report behaviour,
            # but may trigger the aggregator to report depending on the
            # total time elapsed between aggregator updates    
    
    def report(self, aggregated=None, width=82):
        """when called, displays current training progress numbers
        plus progress bar. total display width is given by 'width' argument.
        if aggregated is False, displays directly to stdout.
        if aggregated is True, gathers a more concise report and passes it upstream
            as a variable to the aggregator."""

        if aggregated is None:
            # by default: check if an aggregator is registered
            aggregated = self.aggregator is not None

        # unpack metrics:
        train_losses, val_losses, train_accs, val_accs, prox_losses, lrs = self.metrics.values()

        # figure out where we are in the training process:
        epochs, batches, update_times = self.indices.values()
        epoch = epochs[-1]
        batch = batches[-1]
        update_time = update_times[-1]

        # epoch, batch = divmod(self.step, self.steps_per_epoch)
        # if batch == 0:
        #     self.epoch_start_time = time.time()
        # elif batch+1 == self.steps_per_epoch:
        #     self.epoch_end_time = time.time()
            

        # final = False
        
        if (epoch == self.num_epochs):
            # catch special case: after final batch of epoch
            if self.active:
                # last 'proper' update of the epoch, before becoming inactive
                self.epoch_end_time = time.time()
            epoch = self.num_epochs-1      # display as final epoch (e.g. 5/5)
            batch = self.steps_per_epoch   # display as final batch (e.g. 256/256)
            self.active = False            # flag that this reporter is no longer updating
            # aggregate over whole epochs:
            interval = self.steps_per_epoch
            elapsed = self.epoch_end_time - self.epoch_start_time
        else:
            # regular case: updating during active training
            if epoch != self.current_epoch:
                # record start of new epoch
                self.epoch_start_time = time.time()
                self.current_epoch = epoch

            interval = self.step - self.last_report_steps
            elapsed = update_time - self.last_report_time
        
        if interval == 0:
            if self.step > 0:
                # report method called after the model has stopped training,
                # so overrule the averaging window and average across whole epoch instead
                interval = self.steps_per_epoch
                elapsed = self.epoch_end_time - self.epoch_start_time
            else:
                return # function called before first step; nothing to print
            
        # compute averages and format nice strings:
        avg_tloss = f'{(sum(train_losses[-interval:]) / interval):.4f}'
        avg_tacc =  f'{(sum(train_accs[-interval:])   / interval):.2%}'
        avg_vloss = f'{(sum(val_losses[-interval:])   / interval):.4f}'
        avg_vacc =  f'{(sum(val_accs[-interval:])     / interval):.2%}'
        
        train_str = f'train loss: {avg_tloss} (acc: {avg_tacc})'
        val_str = f'val loss: {avg_vloss} (acc: {avg_vacc})'

        if len(prox_losses) > 0:
            # report fedprox loss if we're training with it
            avg_prox_loss = f'{(sum(prox_losses[-interval:]) / interval):.6f}'
            prox_str = f'prox loss: {avg_prox_loss}'
            loss_str = f'{prox_str:^{width//3}}{train_str:^{(width)//3}}{val_str:^{(width)//3}}'
        else:
            loss_str = f'{train_str:^{width//2}}{val_str:^{width//2}}'


        # if learning rates are being supplied, display the learning rate as well:
        if (not aggregated) and (len(lrs) > 0):
            current_lr = lrs[-1]
            lr_str = f'\n    current learning rate: {current_lr:.0e}'
        else:
            # (but when aggregating, the LR is the same for all models, so no need to display it here)
            lr_str = ''

        elapsed_str = f'{(elapsed / interval):.4f}'
        time_str = f'    time per batch: {elapsed_str}s' if not aggregated else ''
        
        output = (f'{f"Epoch {epoch+1}/{self.num_epochs}":^15}'
                  f'-{f"batch {batch}/{self.steps_per_epoch}":^20}'
                  f'-{f"{((self.step) / self.total_steps):.1%}":^10}  \n'
                  f'{progress_bar(self.step, self.total_steps, epochs=self.num_epochs, bar_len=width-2)}  \n' +\
                  loss_str + lr_str + time_str)

        # log when this report was given, for tracking output timings:
        self.last_report_time = time.time()
        self.last_report_steps = self.step
        self.last_report = output        
        
        if aggregated:
            return output
        else:
            # carriage return and display relevant numbers:
            clear_output(wait=True)
            display_pretty(output,  raw=True)

    def plot_curves(self, rolling_window):
        """plots this instance's training metrics as line graphs,
        loss and accuracy side by side"""
        train_losses, val_losses, train_accs, val_accs, prox_losses, lrs = self.metrics.values()
        epochs = self.indices['epoch']
        plot_metrics(train_losses, val_losses, train_accs, val_accs, 
                     rolling_window=rolling_window, 
                     epochs=epochs, lrs=lrs)


    @property
    def step(self):
        return len(self.metrics['train_losses'])
    @property
    def total_steps(self):
        return self.steps_per_epoch * self.num_epochs
     

class JointProgressReporter:
    """an aggregator class that holds multiple ProgressReport objects
    and displays their outputs simultaneously at regular time intervals.
    intended to be used with asynchronous training of multiple models"""
    def __init__(self, reporters: list[ProgressReporter],
                 time_interval: float=0.5):
        
        self.reporters = reporters   # list of subscribed reporter objects
        self.time_interval = time_interval   # seconds between display updates
        
        print(f'Progress aggregator registered {len(self.reporters)} reporters')

        for reporter in self.reporters:
            # register pointers from reporters to this aggregator
            reporter.aggregator = self

        # last notification time for each reporter:
        self.last_notification = {id(x): time.time() for x in self.reporters}
        # time of last aggregated report:
        self.last_report_time = time.time()

        # control flow flags:
        self.active = False    # denotes that any active updates are still being received
        self.reporting = False # denotes that this object is currently processing a report

        self.messages = [] # permanent messages to be appended to each progress report

        self.global_val_losses = []
        self.global_val_accs = []

    def notify(self, reporter, temp_messages=[]):
        """listener hook that gets called whenever a subscribed reporter updates.
        if enough time has elapsed since the last aggregated report, will generate
        and display a new report."""
        
        # listener hook that gets called whenever a subscribed reporter updates
        cur_time = time.time()
        self.last_notification[id(reporter)] = cur_time
        elapsed = cur_time - self.last_report_time
        
        if (elapsed > self.time_interval) and not self.reporting:
            self.reporting = True
            self.report(temp_messages=temp_messages)
            self.reporting = False
    
    def report(self, temp_messages: list[str]=[]):
        """displays a report based on collecting together
            the individual reports of each subscribed reporter object.
        optionally, can receive additional temporary messages
            to display at this step."""
        
        clear_output(wait=True)
        self.last_report_time = time.time()

        # collect individual reports:
        reports = [f'\nClient #{r}: {reporter.report(aggregated=True)}' 
                   for r, reporter in enumerate(self.reporters)]
        
        # add any temporary messages that should be displayed at this step:
        report_width = len(reports[0].split('\n')[0])
        padded_temp_messages = [f'{msg:^{report_width}}' for msg in temp_messages]
        reports.extend(padded_temp_messages)
        # add any permanent messages that need to be displayed at all times:
        reports.extend([''] + self.messages)
        output_str = '\n'.join([out for out in reports])
        display_pretty(output_str, raw=True)

    def global_update(self, val_loss, val_acc):
        """receive evaluation metrics from central server and record for final loss curve"""
        self.global_val_losses.append(val_loss)
        self.global_val_accs.append(val_acc)
    
    def plot_curves(self, rolling_window=50, fig=None, show_lrs=True):
        """plot the metric curves for all of the reporters subscribed to this aggregator,
        as a square of 2x2 subplots across training/validation loss/accuracy,
        breaking them up by epoch and superimposing the global validation metrics on top."""
        
        #### this is a huge and very messy function, with a lot of untidy hacks and kludges,
        #### because I haven't had time to clean it up and document properly. 
        #### sorry about that, not proud of it.
        
        num_clients = len(self.reporters)
        
    
        global_epoch_vlosses = self.global_val_losses
        global_epoch_vaccs = self.global_val_accs
    
        # assume epoch and learning rate information is the same for all client reporters,
        # but most complete for the client with the largest dataset size, so we use that one
        # as our reference reporter:
        
        max_rep_len = max([rep.step for rep in self.reporters])
        ref_rep = [rep for rep in self.reporters if rep.step == max_rep_len][0]
        lrs, epochs = ref_rep.metrics['lrs'], ref_rep.indices['epoch']
        
        num_epochs = max(epochs)
        epoch_learning_rates = []
        ref_epoch_start_idxs = []
        for e in range(num_epochs):
            # establish the global step at which each epoch starts:
            ref_epoch_start = np.where(np.asarray(epochs)==e)[0][0]
            ref_epoch_start_idxs.append(ref_epoch_start)
            # establish the learning rate used at this epoch:
            epoch_lr = lrs[ref_epoch_start+1]
            epoch_learning_rates.append(epoch_lr)
    
        # x coordinates of global metric datapoints:
        global_xs = (np.arange(num_epochs)+1) * ref_rep.steps_per_epoch
    
        # get current figure size params:
        fig_width, fig_height = rcParams['figure.figsize']
        
        # get the figure, retrieving one if one already exists:
        if fig is None:
            # set height to full width for drawing 2x2 plots:
            rcParams['figure.figsize'] = fig_width, fig_width
            fig, axes = plt.subplots(2, 2)
        else:
            axes = fig.axes
    
        # train on left, validation on right
        # and loss on top, accuracy on bottom:
        (tloss_ax, vloss_ax), (tacc_ax, vacc_ax) = axes

        if len(ref_rep.metrics['prox_losses']) > 0:
            # plot prox loss on the training loss axis if available
            prox_ax = tloss_ax.twinx()
    
        client_shades = np.linspace(0.2, 0.5, num_clients)
        train_cmap = colormaps['Oranges']
        train_colors = [train_cmap(shade) for shade in client_shades]
    
        val_cmap = colormaps['Blues']
        val_colors = [val_cmap(shade) for shade in client_shades]
        
        # pad each client's loss list to that of the longest:
        # longest_list = max([len(losses) for losses in client_val_losses])
        # padded_losses = []
        # client_artists = []
        for r, rep in enumerate(self.reporters):       
    
            # we must plot each epoch separately, since different clients have
            # different length epochs, with gaps in between
    
            for e in range(num_epochs):
                # label the *middle* colour of the client lines.
                # this means we label only one of the ~25 lines drawn inside this loop:
                use_label = (r == int(num_clients // 2)) and (e == 0)
    
                # get metrics for the epoch we are drawing:
                epoch_start_idx = np.where(np.asarray(rep.indices['epoch']) == e)[0][0]
                epoch_end_idx = (np.where(np.asarray(rep.indices['epoch']) == e)[0][-1]) +1
                epoch_tloss = rep.metrics['train_losses'][epoch_start_idx : epoch_end_idx]
                epoch_vloss = rep.metrics['val_losses'][epoch_start_idx : epoch_end_idx]
                epoch_tacc = rep.metrics['train_accs'][epoch_start_idx : epoch_end_idx]
                epoch_vacc = rep.metrics['val_accs'][epoch_start_idx : epoch_end_idx]
    
                client_epoch_len = epoch_end_idx - epoch_start_idx
                # and get the x locations of those metrics, which are actually determined
                # by the epoch start indexes of the reporter with the largest dataset:
                x_steps = np.arange(ref_epoch_start_idxs[e], ref_epoch_start_idxs[e] + client_epoch_len)


                # prox loss: (on same axis as train loss)
                if len(rep.metrics['prox_losses']) > 0:
                    show_prox = True
                    use_legend = False
                    epoch_prox_loss = rep.metrics['prox_losses'][epoch_start_idx : epoch_end_idx]
                    prox_label = ['Client prox loss'] if use_label else None
                    # tloss_ax.bar(x_steps, epoch_prox_loss, label=label, 
                    #              color=colormaps['Reds'](r/num_clients))
                    prox_artist = plot_series(epoch_prox_loss,
                        colors=[colormaps['Reds'](r/num_clients)],
                        linestyles=['-.'],
                        names=prox_label,
                        ylabel=f'Prox loss (k={rolling_window})',   
                        steps=x_steps,
                        use_legend=False,
                        ax=prox_ax, rolling_window=rolling_window, show=False)
                else:
                    use_legend = True
                    show_prox = False
                
                # train loss:
                tloss_label = ['Client train loss'] if use_label else None
                tloss_artist = plot_series(epoch_tloss,
                    colors=[train_colors[r]],
                    linestyles=[':'],
                    names=tloss_label, use_legend=use_legend,
                    ylabel=f'Loss (rolling average, k={rolling_window})',
                    steps=x_steps,
                    ax=tloss_ax, rolling_window=rolling_window, show=False)
                

                if use_label and show_prox:
                    # create a shared legend for prox and training loss:
                    artists = [tloss_artist[0], prox_artist[0]]
                    labels = [a.get_label() for a in artists]
                    
                
                # val loss:
                label = ['Client val loss'] if use_label else None
                plot_series(epoch_vloss,
                    colors=[val_colors[r]],
                    linestyles=[':'],
                    names=label,
                    ylabel=f'Loss (rolling average, k={rolling_window})',                        
                    steps=x_steps,                        
                    ax=vloss_ax, rolling_window=rolling_window, show=False)
        
                # train acc:
                label = ['Client train accuracy'] if use_label else None
                plot_series(epoch_tacc,
                    colors=[train_colors[r]],
                    linestyles=['-'],
                    ylabel=f'Accuracy (rolling average, k={rolling_window})',
                    names=label,
                    steps=x_steps,                        
                    ax=tacc_ax, rolling_window=rolling_window, show=False)
                
                # val acc:
                label = ['Client val accuracy'] if use_label else None
                plot_series(epoch_vacc,
                    colors=[val_colors[r]],
                    linestyles=['-'],
                    ylabel=f'Accuracy (rolling average, k={rolling_window})',
                    names=label,
                    steps=x_steps,
                    ax=vacc_ax, rolling_window=rolling_window, show=False)
    
        # also on validation axes: plot the global validation too
        
        vloss_ax.plot(global_xs, global_epoch_vlosses, '--bo', label='Global val loss')
        vacc_ax.plot(global_xs, global_epoch_vaccs, '-bo', label='Global val accuracy')
        # add dotted lines across entire plot to show epoch / update boundaries:
        for ax_row in axes:
            for ax in ax_row:
                for x in global_xs:
                    ax.axvline(x, c=val_cmap(0.2), linestyle='--', zorder=-99)
                # as well as legends:
                ax.legend()
                # and epoch marks:
                epoch_marks = np.arange(num_epochs) * ref_rep.steps_per_epoch
                ax.set_xticks(epoch_marks, minor=False)
                ax.set_xticklabels(np.arange(num_epochs)+1, minor=False)
                ax.set_xlabel('Epoch')
    
                if show_lrs:
                    epoch_centre_marks = epoch_marks + (ref_rep.steps_per_epoch / 2)
                    ax.set_xticks(epoch_centre_marks, minor=True)
                    lr_strings = [f'{epoch_learning_rates[e]:.0e}' for e in range(num_epochs)]
                    ax.set_xticklabels(lr_strings, minor=True)
                    ax.tick_params(axis='x', which='minor', length=0, labelsize=7)
        
        # plot titles:
        tloss_ax.set_title('Training loss')
        vloss_ax.set_title('Validation loss')
        tacc_ax.set_title('Training accuracy')
        vacc_ax.set_title('Validation accuracy')

        if show_prox:
            # shared legend for training and prox loss:
            tloss_ax.legend(artists, labels)
            
            # if max(epoch_prox_loss) < 1e-2:
            #     # use scientific notation for prox loss:
            #     cur_ylim = prox_ax.get_ylim()
            #     # make sure bottom is at 0:
            #     # prox_ax.set_ylim([0, cur_ylim[1]])
            #     ticklabels = prox_ax.get_yticklabels()
            #     ticktext = [lab.get_text().replace('−', '-') for lab in ticklabels]
            #     ticktext_sci = [f'{float(txt):.0e}' if float(txt) != 0 else '0'  for txt in ticktext]
            #     # [lab.set_text(scitext) for lab, scitext in zip(ticklabels, ticktext_sci)]
            #     # prox_ax.set_yticks(prox_ax.get_yticks())
            #     prox_ax.set_yticklabels(ticktext_sci)
            #     # import ipdb; ipdb.set_trace()

        
        # display:
        fig.suptitle('Federated learning metrics')
        plt.tight_layout(h_pad=4, w_pad=2)
        plt.show()
    
        # restore existing figure size params:
        rcParams['figure.figsize'] = fig_width, fig_height



def progress_bar(current_step, 
                 total_steps,
                 epochs=None, # optional number of epochs to demarcate plot):
                 bar_len=80):
    """makes a pretty little progress bar based on current progress
    out of some total amount of progress, formatted as a string
    of length=bar_length. (plus padding characters at each side)"""

    # define characters to use for bar:
    progress_char,  progress_sep_char  = '═', '╪'
    remaining_char, remaining_sep_char = '─', '┼'
    pad_chars = '[]'
    
    total_progress = bar_len # subtract pad chars
    complete_frac = (current_step) / total_steps # +1 because 1 is the first step
    current_progress = int(np.floor(complete_frac * total_progress))
    remaining_progress = total_progress - current_progress # number of characters of progress
    lb, rb = pad_chars
    bar_chars = [progress_char]*current_progress + [remaining_char]*remaining_progress
    if epochs is not None and epochs != 1:
        # epoch_steps = np.linspace(0, total_steps, epochs+1)
        # epoch_fracs = [(s+1 / total_steps) for s in epoch_steps]
        progress_per_epoch = total_progress / epochs
        epoch_marks = [int(np.round(e))-1 for e in np.arange(epochs) * progress_per_epoch]
        for i in epoch_marks[1:]:
            if bar_chars[i] == progress_char:
                bar_chars[i] = progress_sep_char
            else:
                bar_chars[i] = remaining_sep_char
    bar = lb + ''.join(bar_chars) + rb
    return bar
       