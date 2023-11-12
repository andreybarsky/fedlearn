import numpy as np
import time
import asyncio
from matplotlib import rcParams

from IPython.display import clear_output, display_pretty

from fl_visualisation import plot_metrics


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
            # if an aggregator is defined, tell it that we've updated:
            self.aggregator.notify(self)
            # this suppresses the individual reporter's report behaviour,
            # but may trigger the aggregator to report depending on the
            # total time elapsed between aggregator updates
    
    @property
    def step(self):
        return len(self.metrics['train_losses'])
    @property
    def total_steps(self):
        return self.steps_per_epoch * self.num_epochs
    
    
    
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
            # elif batch == self.steps_per_epoch:
            #     # record end of current epoch
            #     self.epoch_end_time = time.time()

            interval = self.step - self.last_report_steps
            elapsed = update_time - self.last_report_time
        
        if interval == 0:
            if self.step > 0:
                # report method called after the model has stopped training,
                # so overrule the averaging window and average across whole epoch instead
                print('reporter catched special end case')
                interval = self.steps_per_epoch
                elapsed = self.epoch_end_time - self.epoch_start_time
            else:
                return # function called before first step; nothing to print
            
        # compute averages:
        avg_tloss = f'{(sum(train_losses[-interval:]) / interval):.4f}'
        avg_tacc =  f'{(sum(train_accs[-interval:])   / interval):.2%}'
        avg_vloss = f'{(sum(val_losses[-interval:])   / interval):.4f}'
        avg_vacc =  f'{(sum(val_accs[-interval:])     / interval):.2%}'
        
        train_str = f'train loss: {avg_tloss} (acc: {avg_tacc})'
        val_str = f'val loss: {avg_vloss} (acc: {avg_vacc})'

        if len(prox_losses) > 0:
            # report fedprox loss if we're training with it
            avg_prox_loss = f'{(sum(prox_losses[-interval:]) / interval):.4f}'
            prox_str = f'prox loss: {avg_prox_loss}'
            loss_str = f'{prox_str:^{width//3}} {train_str:^{width//3}} {val_str:^{width//3}}'
        else:
            loss_str = f'{train_str:^{width//2}} {val_str:^{width//2}}'

        elapsed_str = f'{(elapsed / interval):.4f}'
        time_str = f'    time per batch: {elapsed_str}s' if not aggregated else ''
            
        # if learning rates are being supplied, display the learning rate as well:
        if len(lrs) > 0:
            current_lr = lrs[-1]
            lr_str = f'    current learning rate: {current_lr:.0e}  \n'
        else:
            lr_str = ''

        output = (f'{f"Epoch {epoch+1}/{self.num_epochs}":^15}'
                  f'-{f"batch {batch}/{self.steps_per_epoch}":^20}'
                  f'-{f"{((self.step) / self.total_steps):.1%}":^10}  \n'
                  f'{progress_bar(self.step, self.total_steps, epochs=self.num_epochs, bar_len=width-2)}  \n' +\
                  loss_str + '\n' + time_str + lr_str)

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
    
    def report(self, temp_messages=[]):
        clear_output(wait=True)
        self.last_report_time = time.time()
        reports = [f'Client #{r}: {reporter.report(aggregated=True)}\n' for r, reporter in enumerate(self.reporters)]
        # add any temporary messages that should be displayed at this step:
        reports.extend(temp_messages)
        # add any permanent messages that need to be displayed:
        reports.extend(self.messages)
        output_str = '\n'.join([out for out in reports])
        display_pretty(output_str, raw=True)
