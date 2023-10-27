import matplotlib.pyplot as plt
from nn_metrics import learning_data_scratch, learning_data_pytorch, learning_data_zeroed, learning_data_uniform

def plot_learning_curves(datalist:list, plot_title:str=None, linewidth:float=1.0, strict_axes:bool=False, savefile:str=None):
    plt.rcParams['figure.figsize'] = 11, 4.25 # (width, height)
    # fig, axes = plt.subplots(*total_shape, constrained_layout=True, sharex=True, sharey=True)
    fig, axes = plt.subplots(1,2, constrained_layout=True)
    # fig.subplots_adjust(top=0.8, bottom=0.2) 
    subplots = (axes[1], axes[0])
    for idx,subplot in enumerate(subplots):
        data = datalist
        metric_type = 'loss' if idx % 2 != 0 else 'accuracy'
        if metric_type == 'loss':
            target_idx = (2,4)
        else:
            target_idx = (3,5)
        x_both = [x[1] for x in data]
        y_train = [x[target_idx[0]] for x in data]
        y_test = [x[target_idx[1]] for x in data]
        ### plot styling ###
        master_title_font_dict = {'fontname':'GE Inspira Sans','fontsize':12}
        subplot_font_dict = {'fontname':'GE Inspira Sans','fontsize':11}
        axes_label_fonts = {'fontname':'GE Inspira Sans','fontsize':11}
        color_dict = {'plum':'#c8a4c6', 'lb':'#afd6d2'}
        subplot.grid(linewidth=0.2) # decrease linewidth of grid to see plots better
        subplot.set_title(f'Model {metric_type.title()}',**subplot_font_dict) # set subplot specific title
        train_error = f"Train {metric_type.title()}"
        test_error = f"Test {metric_type.title()}"
        subplot.plot(x_both, y_train, label=train_error, color=color_dict['lb'],linewidth=linewidth)
        subplot.plot(x_both, y_test, label=test_error, color=color_dict['plum'],linewidth=linewidth) # increasing line width to see better
        legend = subplot.legend(facecolor='#fafbfd')
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor('#2b2d2e')
        for text in legend.get_texts():
            text.set_color('#fafbfd')
            text.set_fontfamily('GE Inspira Sans')
            text.set_fontsize(9)
        if strict_axes:
            # plt.xlim([-1, 51])
            # plt.ylim([0.00, 1.0])
            if metric_type[:3].lower() == 'los':
                subplot.set_ylim(bottom=0, top=2.7) # use top=2.7 for combined regular weights
            else:
                subplot.set_ylim(top=1.05,bottom=0)
        subplot.set_ylabel(f'{metric_type.title()}',**axes_label_fonts)
        subplot.set_xlabel(f'Batch',**axes_label_fonts)
        # ax = plt.gca()
        subplot.set_facecolor('#2b2d2e')
        subplot.tick_params(labelsize=8)
    if plot_title is None:
        plot_title = f'MNIST Neural Network Learning Curves, PyTorch'
    fig.suptitle(plot_title, **master_title_font_dict)
    if savefile is not None:
        plt.savefig(savefile, dpi=199)
    plt.show()

########################### plot learning curves ###########################
if __name__ == '__main__':
    plot_learning_curves(learning_data_scratch, strict_axes=True, linewidth=1.2, plot_title='MNIST Neural Network Learning Curves, Scratch Implementation', savefile=None) # scratch
    plot_learning_curves(learning_data_pytorch, strict_axes=True, linewidth=1.2, plot_title='MNIST Neural Network Learning Curves, PyTorch', savefile=None) # pytorch
    plot_learning_curves(learning_data_zeroed, strict_axes=True, linewidth=1.2, plot_title='MNIST Neural Network Learning Curves, Weights Zeroed', savefile=None) # zeroed
    plot_learning_curves(learning_data_uniform, strict_axes=True, linewidth=1.2, plot_title='MNIST Neural Network Learning Curves, Weights [-1, 1]', savefile=None) # uniform
