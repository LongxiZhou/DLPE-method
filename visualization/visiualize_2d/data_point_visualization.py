import matplotlib.pyplot as plt


def show_data_points(x, y, save_path=None, data_label='data points', x_name='x_axis', y_name='y_axis', title='scatter'):
    plt.close()

    if data_label is not None:
        # plot1 = plt.plot(x, y, 'r', label='data points')
        plot1 = plt.plot(x, y, '*', label='data points')
    else:
        # plot1 = plt.plot(x, y, 'r')
        plot1 = plt.plot(x, y, '*')

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
    plt.title(title)
    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600
        plt.savefig(save_path)
        plt.close()


def show_two_data_sets(x1, y1, x2, y2, save_path=None, show=True, data_label=True):

    if show:
        if data_label:
            plot1 = plt.plot(x1, y1, '.', label='y_1 values')
            plot2 = plt.plot(x2, y2, '*', label='y_2 values', color='#FFA500')
        else:
            plot1 = plt.plot(x1, y1, '.')
            plot2 = plt.plot(x2, y2, '*', color='#FFA500')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
        plt.title('scatter graph')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()


def show_multiple_observations(x_lists, y_lists, save_path=None, show=True, line=True):
    # x_lists = [[x for observation 1], [x for observation 2], ...]
    # y_lists = [[y for observation 1], [y for observation 2], ...]
    plt.close()
    num_observations = len(x_lists)
    if show:
        for observation in range(num_observations):
            x = x_lists[observation]
            y = y_lists[observation]
            plt.plot(x, y, '*')
            if line:
                plt.plot(x, y, 'r-')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        # plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
        # plt.title('scatter graph')
        if save_path is None:
            plt.show()
        else:
            # plt.imshow()
            plt.savefig(save_path)
            plt.close()


def show_two_dataset_multiple_observation(x1_lists, y1_lists, x2_lists, y2_lists, save_path=None, show=True, dot=False, line=True):
    # x1_lists = [[x for observation 1], [x for observation 2], ...]
    # y1_lists = [[y for observation 1], [y for observation 2], ...]
    plt.close()
    num_observations_1 = len(x1_lists)
    num_observations_2 = len(x2_lists)
    if show:
        for observation in range(num_observations_1):
            x = x1_lists[observation]
            y = y1_lists[observation]
            if dot:
                plt.plot(x, y, '*')
            if line:
                plt.plot(x, y, 'r-', alpha=0.5)
        for observation in range(num_observations_2):
            x = x2_lists[observation]
            y = y2_lists[observation]
            if dot:
                plt.plot(x, y, '*')
            if line:
                plt.plot(x, y, 'b-', alpha=0.5)
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        # plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
        # plt.title('scatter graph')
        '''
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_xticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='black', alpha=0.7, zorder=2))
        '''
        if save_path is None:
            plt.show()
        else:
            # plt.imshow()
            plt.savefig(save_path)
            plt.close()
