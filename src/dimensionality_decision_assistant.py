import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DimensionalityDecisionAssistant(object):
    def __init__(self, df, encoder_seq_col, decoder_seq_col,
                 len_encoder_seq_col, len_decoder_seq_col,
                 start_char, end_char, n_thresholds=10,
                 chosen_input_thresh=None, chosen_target_thresh=None):
        """
        df: dataframe, includes following columns:
            encoder_seq_txt_col
                input sequence text (encoder)
            decoder_seq_txt_col
                input sequence text (decoder)
            len_encoder_seq_col
                --> column values are the count of the length of the encoder text input for that row (generalizable to different units: char, words, etc.)
            len_decoder_seq_col
                --> same as above, but for decoder sequence
        """
        self.df = df
        self.encoder_seq_col = encoder_seq_col
        self.decoder_seq_col = decoder_seq_col
        self.input_text_arr = df[encoder_seq_col].values
        self.target_text_arr = df[decoder_seq_col].values
        self.len_encoder_seq_col = len_encoder_seq_col
        self.len_decoder_seq_col = len_decoder_seq_col
        self.start_char = start_char
        self.end_char = end_char
        self.initial_n_samples = df.shape[0]
        self.n_thresholds = n_thresholds
        self.chosen_input_thresh = chosen_input_thresh
        self.chosen_target_thresh = chosen_target_thresh
        self._get_initial_dimensions()
        self._make_threshold_combos_df()

    def _get_initial_dimensions(self):
        # Get initial counts + dimensions
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        for txt_idx in range(self.initial_n_samples):
            input_text = self.input_text_arr[txt_idx]
            target_text = self.target_text_arr[txt_idx]
            target_text = self.start_char + target_text + self.end_char
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        self.initial_input_characters = sorted(list(input_characters))
        self.initial_target_characters = sorted(list(target_characters))
        self.initial_num_encoder_tokens = len(input_characters)
        self.initial_num_decoder_tokens = len(target_characters)
        self.initial_max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.initial_max_decoder_seq_length = max([len(txt) for txt in target_texts])
        self.initial_enc_dim = self.initial_n_samples * self.initial_max_encoder_seq_length * self.initial_num_encoder_tokens
        self.initial_dec_dim = self.initial_n_samples * self.initial_max_decoder_seq_length * self.initial_num_decoder_tokens

    def calc_df_within_thresholds(self, input_thresh, target_thresh):
        is_within_thresholds = (self.df[self.len_encoder_seq_col] <= input_thresh)&(self.df[self.len_decoder_seq_col] <= target_thresh)
        new_df = self.df[is_within_thresholds]
        return new_df

    def calc_decision_stats(self, input_thresh, target_thresh):
        new_df = self.calc_df_within_thresholds(input_thresh, target_thresh)
        new_n_samples = new_df.shape[0]
        perc_samples_reduced = ((new_n_samples - self.initial_n_samples) / self.initial_n_samples)*100
        perc_input_reduced = ((input_thresh - self.initial_max_encoder_seq_length) / self.initial_max_encoder_seq_length)*100
        perc_target_reduced = ((target_thresh - self.initial_max_decoder_seq_length) / self.initial_max_decoder_seq_length)*100
        new_enc_dim = new_n_samples * input_thresh * self.initial_num_encoder_tokens
        perc_enc_dim_reduced = ((new_enc_dim - self.initial_enc_dim) / self.initial_enc_dim)*100
        new_dec_dim = new_n_samples * target_thresh * self.initial_num_decoder_tokens
        perc_dec_dim_reduced = ((new_dec_dim - self.initial_dec_dim) / self.initial_dec_dim)*100
        avg_perc_dim_red = (perc_enc_dim_reduced + perc_dec_dim_reduced)/2
        decision_stats = [input_thresh, target_thresh, new_n_samples, perc_samples_reduced, perc_input_reduced,
                    perc_target_reduced, perc_enc_dim_reduced, perc_dec_dim_reduced, avg_perc_dim_red]
        return decision_stats

    def _make_threshold_combos_df(self, lowest_thresh=50):
        dr_df = pd.DataFrame(columns=['input_thresh', 'target_thresh', 'new_sample_size',
                                      'perc_samples_reduced', 'perc_input_reduced',
                                      'perc_target_reduced', 'perc_enc_dim_reduced',
                                      'perc_dec_dim_reduced', 'avg_perc_dim_red'])
        self.input_txt_thresholds = np.linspace(lowest_thresh, self.initial_max_encoder_seq_length, self.n_thresholds)
        self.target_txt_thresholds = np.linspace(lowest_thresh, self.initial_max_decoder_seq_length, self.n_thresholds)
        idx = 0
        for input_thresh in self.input_txt_thresholds:
            for target_thresh in self.target_txt_thresholds:
                row_data = self.calc_decision_stats(input_thresh, target_thresh)
                dr_df.loc[idx] = row_data
                idx += 1
        self.dr_df = dr_df

    def get_user_estimate_of_threshold(self, chosen_thresh=None, for_input=True, save_fig=True):
        user_response = self._plot_seq_len_hist(chosen_thresh, for_input=for_input, save_fig=save_fig)
        cnt = 0
        while user_response != 'done' and cnt < 10:
            user_response = self._plot_seq_len_hist(chosen_thresh=int(user_response), for_input=for_input, save_fig=save_fig)
        print(f"Alright, we've settled on an estimated threshold!")

    def _plot_seq_len_hist(self, chosen_thresh=None, for_input=True, save_fig=True):
        col = self.len_encoder_seq_col if for_input else self.len_decoder_seq_col
        filename = 'input' if for_input else 'target'
        values = self.df[col]
        plot_title = 'Input' if for_input else 'Target'
        plt.title(f"Histogram of {plot_title} Sequence Lengths")
        plt.xlabel("Sequence Length")
        plt.ylabel("Count")
        if chosen_thresh:
            plt.axvline(x=chosen_thresh, c='green', alpha=0.5, linewidth=3, label='chosen_threshold')
            if for_input:
                self.chosen_input_thresh = chosen_thresh
            else:
                self.chosen_target_thresh = chosen_thresh
        values.hist(bins=100)
        if save_fig:
            plt.savefig(f'../images/{filename}_seq_len_hist.png')
        plt.show()
        user_response = input("Where do you think a good maximum sequence length threshold might be? (If best already identified, enter 'done')")
        return user_response

    def find_closest_threshold(self, given_thresh, for_input=True):
        # figure out how to find closest thresholds to the chosen one
        thresh_col = 'input_thresh' if for_input else 'target_thresh'
        closest_thresh_idx = (abs(self.dr_df[thresh_col].unique() - given_thresh)).argsort()[:1]
        closest_threshold = self.dr_df[thresh_col].unique()[closest_thresh_idx]
        return closest_threshold

    def plot_decision_curve(self, for_input=True, save_fig=True,
                            chosen_thresh=None, perc_data_loss_est=None,
                            perc_dim_red_est=None):
        print('''
        Goal is to use the green lines on the plot to 'corner' the
        threshold you'd like to examine.

        Rerun this code with different values for:
            -chosen_thresh
            -perc_data_loss_est
            -perc_dim_red_est

        until you think you've found the points where the vertical
        green line intersects with the red and blue curves.

        Generally, you'll want to pick a chosen_thresh so that
        the vertical green line intersects the red curve
        near the red curve's elbow.
        ''')
        if for_input:
            plot_title = 'Input'
            thresh_col = 'input_thresh'
            alt_thresh_col = 'target_thresh'
            perc_dim_reduced_col = 'perc_enc_dim_reduced'
            chosen_thresh = chosen_thresh if chosen_thresh else self.chosen_input_thresh
        else:
            plot_title = 'Target'
            thresh_col = 'target_thresh'
            alt_thresh_col = 'input_thresh'
            perc_dim_reduced_col = 'perc_dec_dim_reduced'
            chosen_thresh = chosen_thresh if chosen_thresh else self.chosen_target_thresh
        fig = plt.figure(figsize=(15, 7.5))
        ax = fig.add_subplot(111)
        ax.set_title(f"Maximum {plot_title} Sequence Length: Dimensionality Reduction vs. Data Loss", fontsize=18)
        if chosen_thresh:
            plt.axvline(x=chosen_thresh, c='green', alpha=0.75, linewidth=3, label='chosen_threshold')
        alt_thresholds = sorted(self.dr_df[alt_thresh_col].unique(), reverse=True)
        cnt = self.n_thresholds + 2
        for alt_thresh in alt_thresholds:
            same_alt_thresh = self.dr_df[alt_thresh_col] == alt_thresh
            x = self.dr_df[same_alt_thresh][thresh_col]
            y1 = self.dr_df[same_alt_thresh]['perc_samples_reduced']
            y2 = self.dr_df[same_alt_thresh][perc_dim_reduced_col]
            plt.plot(x, y1, alpha=cnt/self.n_thresholds, c='red', linewidth=2, label='data_loss')
            plt.plot(x, y2, alpha=cnt/self.n_thresholds, c='blue', linewidth=2, label='dimension_reduction')
            cnt -= 1
        if perc_data_loss_est:
            plt.axhline(y=perc_data_loss_est, c='green', alpha=0.3, linewidth=3, label='chosen_threshold')
        if perc_dim_red_est:
            plt.axhline(y=perc_dim_red_est, c='green', alpha=0.3, linewidth=3, label='chosen_threshold')
        ax.set_xlabel(f'Max. {plot_title} Sequence Length', fontsize=16)
        ax.set_ylabel('Percent Reduced (%)', fontsize=16)
        plot_labels = ['data_loss', 'dimension_reduction']
        if chosen_thresh:
            plot_labels = ['chosen_threshold', 'data_loss', 'dimension_reduction']
        ax.legend(labels=plot_labels, loc='lower right', ncol=1, markerscale=0.5)
        if save_fig:
            fig.savefig(f'../images/{plot_title.lower()}_dda_{self.n_thresholds}_thresholds.png')
        plt.show()

    def combined_plot(self, perc_data_loss_max=None, perc_dim_red_min=None, save_fig=True):
        print('''
        Goal is to use the green lines on the plot
        to 'corner' the points you'd like to examine.

        Rerun this code with different values for:
            - perc_data_loss_est
            - perc_dim_red_est

        until you've isolated the points you think would be
        best in the upper lefthand box created by the green lines.

        Generally, you'll want to pick points that occur
        near the elbow of the purple dots
        ''')
        x = self.dr_df['avg_perc_dim_red']
        y = self.dr_df['perc_samples_reduced']
        alpha=0.5
        fig = plt.figure(figsize=(15, 7.5))
        ax = fig.add_subplot(111)
        ax.set_title("Maximum Sequence Length Thresholds: Dimensionality Reduction vs. Data Loss", fontsize=18)
        ax.scatter(x, y, alpha=alpha, color='purple', label='data_reduction')
        if perc_dim_red_min:
            plt.axvline(x=perc_dim_red_min, c='green', alpha=0.3)
        if perc_data_loss_max:
            plt.axhline(y=perc_data_loss_max, c='green', alpha=0.3)
        ax.set_xlabel('Avg % Dimensions Reduced', fontsize=16)
        ax.set_ylabel('% Data Lost', fontsize=16)
        if save_fig:
            plt.savefig(f'../images/dim_red_decision_combined_plot_{self.n_thresholds}_lvls.png')
        plt.show()

        self.perc_data_loss_max = perc_data_loss_max
        self.perc_dim_red_min = perc_dim_red_min

    def view_within_requirement_stats(self):
        is_within_reqs = (self.dr_df['avg_perc_dim_red'] < self.perc_dim_red_min)&(self.dr_df['perc_samples_reduced'] >= self.perc_data_loss_max)
        trimmed_df = self.dr_df[is_within_reqs]
        print(trimmed_df[['input_thresh', 'target_thresh', 'avg_perc_dim_red', 'perc_samples_reduced']])

    def quick_calc_perc_for_new_thresholds(self, input_thresh, target_thresh):
        decision_stats = self.calc_decision_stats(input_thresh, target_thresh)
        print(f"Percent Data Loss: {decision_stats[3]} \nPercent Dimensionality Reduced: {decision_stats[8]}")

if __name__ = '__main__':
    pd.set_option("display.max_columns",100)

    df = pd.read_csv('../data/less_raw_data.csv')

    dda = DimensionalityDecisionAssistant(df, 'ref_text', 'tate_text',
                                     'chars_in_referent', 'chars_in_tate',
                                     '\v', '\b', n_thresholds=10,
                                     chosen_input_thresh=None, chosen_target_thresh=None)

    # Get initial estimates of good threshold from user input - asks until user indicates good decision has been made
    dda.get_user_estimate_of_threshold(chosen_thresh=None, for_input=True, save_fig=True)
    dda.get_user_estimate_of_threshold(chosen_thresh=None, for_input=False, save_fig=True)

    # look at decision curves for INPUT txt - run until satisfied
    dda.plot_decision_curve(for_input=True, save_fig=True,
                    chosen_thresh=None, perc_data_loss_est=None,
                    perc_dim_red_est=None)
    # look at decision curves for TARGET txt - run until satisfied
    dda.plot_decision_curve(for_input=False, save_fig=True,
                    chosen_thresh=None, perc_data_loss_est=None,
                    perc_dim_red_est=None)

    # find the closest threshold value in the calculated df to the one you've estimated
    closest_input_thresh = dda.find_closest_threshold(dda.chosen_input_thresh)
    closest_target_thresh = dda.find_closest_threshold(dda.chosen_target_thresh)

    # try to define maximum data loss and minimum dimension reduction acceptable
    # then continue to use/rerun this code to pinpoint "best" point within that range
    dda.combined_plot(perc_data_loss_max=None, perc_dim_red_min=None, save_fig=True)

    # view threshold combinations that satistfy these min/max requirements
    dda.view_within_requirement_stats()

    # try custom thresholds and see if we can get even better results
    dda.quick_calc_perc_for_new_thresholds(input_thresh, target_thresh)
